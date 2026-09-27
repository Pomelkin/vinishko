import math
from typing import ClassVar
from typing import cast

import torch
import torch.nn.functional as F
from kostyl.ml.integrations.lightning import LightningCheckpointLoader
from kostyl.ml.integrations.lightning import LightningConfigLoader
from torch import nn
from transformers import PreTrainedConfig
from transformers import initialization as init
from transformers.models.dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit import DINOv3ViTModel
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTPreTrainedModel


class DinoV3ForWineConfig(PreTrainedConfig, LightningConfigLoader):
    r"""
    backbone_config (`dict | DINOv3ViTConfig`, *optional*):
        Конфиг бэкбона DINOv3; словарь превращается в `DINOv3ViTConfig`, без значения берётся конфиг по умолчанию.
    embed_dim (`int`, *optional*, defaults to 512):
        Размерность итогового вектора.
    gem_p (`float`, *optional*, defaults to 3.0):
        Начальная степень GeM-пулинга по патч-токенам; параметр обучаемый.
    gem_eps (`float`, *optional*, defaults to 1e-06):
        Нижняя граница токенов перед возведением в степень.
    initializer_range (`float`, *optional*, defaults to 0.02):
        Стандартное отклонение инициализации Linear головы.
    """

    model_type = "dinov3_for_wine"
    sub_configs: ClassVar[dict[str, type[PreTrainedConfig]]] = {
        "backbone_config": DINOv3ViTConfig
    }

    backbone_config: dict | PreTrainedConfig | None = None
    embed_dim: int = 512
    gem_p: float | int = 3.0
    gem_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs) -> None:
        if isinstance(self.backbone_config, dict):
            self.backbone_config = self.sub_configs["backbone_config"](
                **self.backbone_config
            )
        elif self.backbone_config is None:
            self.backbone_config = self.sub_configs["backbone_config"]()
        super().__post_init__(
            **kwargs
        )  # после подконфига: здесь attn_implementation пробрасывается во вложенные конфиги

    @property
    def backbone(self) -> DINOv3ViTConfig:
        """Конфиг бэкбона: после __post_init__ это всегда объект, а не словарь."""
        return cast(DINOv3ViTConfig, self.backbone_config)

    @property
    def patch_size(self) -> int:
        """Сторона патча бэкбона: обе стороны входа обязаны делиться на неё нацело."""
        return int(self.backbone.patch_size)  # ty: ignore[invalid-argument-type]


class GeM(nn.Module):
    """Generalized mean по токенам с обучаемой степенью: p=1 — среднее, большие p тянут к максимуму.

    Степень хранится логарифмом: p = exp(log_p) всегда положительна, а шаг оптимизатора меняет её в разы, а не на единицы.
    Считается в float32: степень и корень в половинной точности теряют мелкие активации.
    Токены после LayerNorm бывают отрицательными, дробная степень от них не определена, поэтому они поджимаются к eps.
    """

    def __init__(self, p: float, eps: float) -> None:
        super().__init__()
        self.log_p = nn.Parameter(torch.full((1,), math.log(float(p))))
        self.eps = eps

    @property
    def p(self) -> torch.Tensor:
        """Степень, не ниже 1: меньше — уже не обобщённое среднее, а сжатие к геометрическому."""
        return self.log_p.float().clamp(min=0.0).exp()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        p = self.p
        return (
            tokens.float()
            .clamp(min=self.eps)
            .pow(p)
            .mean(dim=1)
            .pow(1.0 / p)
            .to(tokens.dtype)
        )


class DinoV3ForWine(DINOv3ViTPreTrainedModel, LightningCheckpointLoader):
    """DINOv3 с головой под метрик-лёрнинг: CLS ⊕ GeM по патч-токенам → BN → Linear → BN → вектор embed_dim.

    От DINOv3ViTPreTrainedModel наследуются признаки семейства: поддержка sdpa, flash attention и flex attention, неделимые при шардировании
    слои, gradient checkpointing, инициализация Linear и BatchNorm. Регистровые токены DINOv3 в пулинг не идут. Вход — прямоугольник любого
    размера, кратного патчу: позиции у DINOv3 считает RoPE по фактическому размеру картинки. L2-нормировку делает forward при normalize=True;
    лоссу ArcFace вектор отдаётся без неё, он нормирует сам.
    """

    config: DinoV3ForWineConfig
    base_model_prefix = "backbone"

    def __init__(self, config: DinoV3ForWineConfig) -> None:
        super().__init__(config)
        self.backbone = DINOv3ViTModel(config.backbone)
        self.num_prefix_tokens = 1 + config.backbone.num_register_tokens
        self.gem = GeM(config.gem_p, config.gem_eps)
        pooled = 2 * config.backbone.hidden_size
        self.head = nn.Sequential(
            nn.BatchNorm1d(pooled),
            nn.Linear(pooled, config.embed_dim, bias=True),
            nn.BatchNorm1d(config.embed_dim),
        )
        # маск-токен нужен только предобучению DINOv3; в прямом проходе его нет, и DDP счёл бы его параметром без градиента
        self.backbone.embeddings.mask_token.requires_grad_(False)
        self.post_init()

    @classmethod
    def from_backbone(
        cls,
        name_or_path: str,
        embed_dim: int = 512,
        gem_p: float = 3.0,
        gem_eps: float = 1e-6,
        device_map: torch.device | str | None = None,
        attn_implementation: str | None = None,
        drop_path_rate: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> "DinoV3ForWine":
        """Модель с предобученным бэкбоном DINOv3 и свежей головой. Репозитории DINOv3 на HF закрытые: token=True берёт токен из hf auth login.

        Загрузить чекпоинт бэкбона прямо через from_pretrained этого класса нельзя: его config.json плоский, а здесь конфиг бэкбона вложен.
        """
        backbone = DINOv3ViTModel.from_pretrained(
            name_or_path,
            token=True,
            device_map=device_map,
            attn_implementation=attn_implementation,
            drop_path_rate=drop_path_rate,
            attention_dropout=attention_dropout,
        )
        config = DinoV3ForWineConfig(
            backbone_config=backbone.config,
            embed_dim=embed_dim,
            gem_p=gem_p,
            gem_eps=gem_eps,
            attn_implementation=attn_implementation,
        )
        model = cls(config)
        model.backbone.load_state_dict(backbone.state_dict(), strict=True)
        return model.to(
            backbone.device  # ty: ignore[invalid-argument-type]
        )  # новая модель собирается на CPU, device_map переносил только временный бэкбон

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Linear и BatchNorm головы инициализирует родитель, здесь — только степень GeM. Модули бэкбона до этого метода не доходят: вложенная модель инициализирует их сама."""
        super()._init_weights(module)
        if isinstance(module, GeM):
            init.constant_(module.log_p, math.log(float(self.config.gem_p)))

    def check_input_size(self, height: int, width: int) -> None:
        """Размер входа обязан делиться на патч нацело: иначе свёртка патчей молча отбросит край картинки."""
        patch = self.config.patch_size
        if height % patch or width % patch:
            raise ValueError(
                f"размер входа {height}×{width} не делится нацело на patch_size={patch}"
            )

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """Последовательность бэкбона (B, 1 + регистры + патчи, C) → CLS ⊕ GeM по патчам, (B, 2C)."""
        cls_token = tokens[:, 0]
        patches = tokens[:, self.num_prefix_tokens :]
        return torch.cat([cls_token, self.gem(patches)], dim=1)

    @property
    def head_dtype(self) -> torch.dtype:
        """Тип весов головы: у модели, загруженной в bf16, в нём же и голова. В fp32 на CPU голова тоже в fp32."""
        return next(self.head.parameters()).dtype

    def forward(
        self,
        pixel_values: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Эмбеддинг картинок (B, embed_dim) в float32."""
        self.check_input_size(pixel_values.shape[-2], pixel_values.shape[-1])
        tokens = self.backbone(pixel_values=pixel_values).last_hidden_state

        tokens_dtype = tokens.dtype
        with torch.autocast(device_type=pixel_values.device.type, enabled=False):
            tokens = tokens.float()

            pooled = self.pool(tokens)
            embedding = self.head(pooled)

        embedding = F.normalize(embedding, dim=-1) if normalize else embedding
        return embedding.to(tokens_dtype)
