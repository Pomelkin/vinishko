from pathlib import Path
from typing import Any
from typing import cast
from typing import override

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from kostyl.ml.configs.structs.training_settings import FSDP1StrategyConfig
from kostyl.ml.configs.structs.training_settings import FSDP2StrategyConfig
from kostyl.ml.dist_utils import scale_lrs_by_world_size
from kostyl.ml.dist_utils.fsdp import get_fsdp1_policies
from kostyl.ml.dist_utils.fsdp import get_fsdp2_policies
from kostyl.ml.dist_utils.fsdp import get_transformer_shard_modules
from kostyl.ml.dist_utils.fsdp import select_wrap_policy
from kostyl.ml.integrations.lightning import KostylLightningModule
from kostyl.ml.integrations.lightning import estimate_total_steps
from kostyl.ml.optim import create_optimizer
from kostyl.ml.optim import create_scheduler
from kostyl.ml.optim.schedulers import BaseScheduler
from kostyl.ml.optim.schedulers import CompositeScheduler
from kostyl.utils import setup_logger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.strategies import ModelParallelStrategy
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint.state_dict import StateDictOptions
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.checkpoint.state_dict import set_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import fully_shard
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from transformers import PreTrainedConfig
from transformers import PreTrainedModel

from vis_seacher_training.configs import ModelConfig
from vis_seacher_training.configs import TrainingConfig
from vis_seacher_training.datamodule import VAL_LOADERS
from vis_seacher_training.datamodule import WineDataModule
from vis_seacher_training.dino_modeling import DinoV3ForWine
from vis_seacher_training.dino_modeling import DinoV3ForWineConfig
from vis_seacher_training.losses import SubCenterArcFace
from vis_seacher_training.metrics import TOP
from vis_seacher_training.metrics import Misses
from vis_seacher_training.metrics import Panel
from vis_seacher_training.metrics import evaluate_retrieval
from vis_seacher_training.metrics import panels_figure
from vis_seacher_training.metrics import scores_figure
from vis_seacher_training.optimization import create_param_groups


logger = setup_logger(fmt="detailed")

Batch = dict[str, torch.Tensor]
MONITOR_METRIC = "wine/oneshot_noisy/recall@1"
"""Главная метрика: вино, одно фото класса в галерее, в галерею подмешано не-вино. Под именем val_recall1 за ней следит чекпоинтер."""
TRAIN_IMAGES_IN_GRID = 32
HIT_COLOR, MISS_COLOR, NOISE_COLOR = "#1baf7a", "#eb6834", "#eda100"
"""Рамки в сетке промахов: фото класса запроса, чужое вино, не-вино."""


def build_model(cfg: ModelConfig, device: torch.device) -> DinoV3ForWine:
    """Модель по конфигу: готовая из init_from, например из init_model.py, либо предобученный бэкбон со свежей головой.

    drop_path_rate и attention_dropout — регуляризация бэкбона; у DINOv3 в чекпоинтах они нулевые, поэтому задаются здесь до сборки модели:
    DropPath создаётся в конструкторе слоя только при ненулевой доле. Модель возвращается в eval, как её отдаёт from_pretrained;
    режим обучения включает configure_optimizers.
    """
    regularization = {
        "drop_path_rate": cfg.drop_path_rate,
        "attention_dropout": cfg.attention_dropout,
    }
    if cfg.init_from is not None:
        config = DinoV3ForWineConfig.from_pretrained(cfg.init_from)
        for name, value in regularization.items():
            setattr(config.backbone, name, value)
        model = DinoV3ForWine.from_pretrained(
            cfg.init_from,
            config=config,
            device_map=device,
            attn_implementation=cfg.attn_implementation,
        )
        if model.config.embed_dim != cfg.embed_dim:
            raise ValueError(
                f"model.embed_dim={cfg.embed_dim} в конфиге, а у модели из {cfg.init_from} — {model.config.embed_dim}"
            )
        model.backbone.embeddings.mask_token.requires_grad_(
            False
        )  # from_pretrained создаёт параметры заново и снимает заморозку из конструктора
        return model
    return DinoV3ForWine.from_backbone(
        cfg.backbone,
        embed_dim=cfg.embed_dim,
        gem_p=cfg.gem_p,
        device_map=device,
        attn_implementation=cfg.attn_implementation,
        **regularization,
    )


class WineTrainingModule(KostylLightningModule):
    """Обучение DinoV3ForWine лоссом sub-center ArcFace и проверка ретривом.

    На шаге обучения логируется только лосс. На валидации копятся эмбеддинги пяти даталоадеров, в конце эпохи по ним считаются
    recall@k вина, отказ по запросам без ответа и синтетический замер каталога; распределения косинуса top-1 уходят в TensorBoard
    гистограммами по группам запросов и общей картинкой, а несколько промахов главного протокола — сеткой картинок.
    """

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.hyperparams = config.hyperparams
        self.model: DinoV3ForWine | FSDP | None = None
        self.loss_fn: SubCenterArcFace | FSDP | None = None
        self._val_embeddings: list[list[torch.Tensor]] = [[] for _ in VAL_LOADERS]
        self._val_indices: list[list[torch.Tensor]] = [[] for _ in VAL_LOADERS]

    @override
    def configure_model(self) -> None:  # noqa: C901
        if self.model is not None:
            return
        cfg, datamodule = self.config, self.datamodule
        model = build_model(cfg.model, self.trainer.strategy.root_device)
        if model.config.patch_size != cfg.model.patch_size:
            raise ValueError(
                f"model.patch_size={cfg.model.patch_size} в конфиге, а у бэкбона {cfg.model.backbone} патч {model.config.patch_size}"
            )
        model.check_input_size(*cfg.data.input_size)
        if cfg.model.freeze_backbone:
            model.backbone.requires_grad_(False)
        if cfg.model.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        self.model = model
        self.loss_fn = SubCenterArcFace(
            dim=cfg.model.embed_dim,
            n_classes=datamodule.num_classes,
            class_counts=datamodule.class_counts,
            k=cfg.loss.k,
            s=cfg.loss.s,
            m_a=cfg.loss.m_a,
            m_b=cfg.loss.m_b,
            m_lambda=cfg.loss.m_lambda,
            centers=self._initial_centers(),
        )

        if isinstance(self.trainer.strategy, FSDPStrategy):
            if not isinstance(self.config.trainer_params.strategy, FSDP1StrategyConfig):
                raise ValueError("FSDPStrategy is required for this training setup.")
            policies = get_fsdp1_policies(self.config.trainer_params.strategy)
            wrap_policy = select_wrap_policy(model=self.model)

            self.model.backbone = FSDP(  # ty: ignore[invalid-assignment]
                module=self.model.backbone,
                auto_wrap_policy=wrap_policy,
                device_id=self.trainer.strategy.root_device,
                use_orig_params=True,
                **policies,
            )

            fp32_policies = {
                **policies,
                "mixed_precision": MixedPrecision(
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32,
                ),
            }

            self.model.gem = FSDP(  # ty: ignore[invalid-assignment]
                module=self.model.gem,
                sharding_strategy=ShardingStrategy.NO_SHARD,
                device_id=self.trainer.strategy.root_device,
                use_orig_params=True,
                **fp32_policies,  # ty: ignore[invalid-argument-type]
            )
            self.model.head = FSDP(  # ty: ignore[invalid-assignment]
                module=self.model.head,
                sharding_strategy=ShardingStrategy.NO_SHARD,
                device_id=self.trainer.strategy.root_device,
                use_orig_params=True,
                **fp32_policies,  # ty: ignore[invalid-argument-type]
            )
            self.model = FSDP(
                module=self.model,
                device_id=self.trainer.strategy.root_device,
                use_orig_params=True,
                **policies,
            )
            # лосс в fp32, как и голова: training_step считает ArcFace без autocast, а bf16-политика сделала бы центры классов bf16
            self.loss_fn = FSDP(
                module=self.loss_fn,
                sharding_strategy=ShardingStrategy.NO_SHARD,
                device_id=self.trainer.strategy.root_device,
                use_orig_params=True,
                **fp32_policies,  # ty: ignore[invalid-argument-type]
            )
            self.trainer.strategy.model = FSDP(
                module=self,
                device_id=self.trainer.strategy.root_device,
                use_orig_params=True,
                **policies,
            )
        elif isinstance(self.trainer.strategy, ModelParallelStrategy):
            if not isinstance(self.config.trainer_params.strategy, FSDP2StrategyConfig):
                raise ValueError(
                    "ModelParallelStrategy is required for this training setup."
                )
            policies = get_fsdp2_policies(self.config.trainer_params.strategy)
            dp_mesh = self.trainer.strategy.device_mesh["data_parallel"]
            modules_to_shard = get_transformer_shard_modules(self.model)
            if modules_to_shard is None:
                raise ValueError(
                    "No modules to shard found in the model. Check the model architecture."
                )
            for module in self.model.backbone.modules():
                if type(module) in modules_to_shard:
                    fully_shard(module, mesh=dp_mesh, **policies)
            fp32_policies = {
                **policies,
                "mp_policy": MixedPrecisionPolicy(
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                ),
            }
            for module in (self.model.gem, self.model.head):
                fully_shard(
                    module,
                    mesh=dp_mesh,
                    **fp32_policies,
                )  # ty: ignore[no-matching-overload]
            fully_shard(self.model, mesh=dp_mesh, **policies)
            replicate(self.loss_fn, mesh=dp_mesh, **fp32_policies)  # ty: ignore[no-matching-overload]
        return

    def _initial_centers(self) -> torch.Tensor | None:
        """Стартовые центры ArcFace из loss.centers_init в порядке классов датамодуля; классам без записи достаётся случайный вектор."""
        path = self.config.loss.centers_init
        if path is None:
            return None
        saved = torch.load(path, map_location="cpu", weights_only=True)
        by_name = dict(zip(saved["class_names"], saved["centers"], strict=True))
        centers = torch.empty(self.datamodule.num_classes, self.config.model.embed_dim)
        torch.nn.init.xavier_uniform_(centers)
        found = 0
        for row, name in enumerate(self.datamodule.class_names):
            if name in by_name:
                centers[row] = by_name[name]
                found += 1
        logger.log_rank_zero(
            "INFO",
            f"Центры ArcFace из {path}: {found} из {len(centers)} классов, остальные случайные",
        )
        return centers

    def load_weights(self, path: Path) -> None:
        """Веса чекпоинта в живую, возможно шардированную, модель на всех рангах: директория — DCP от FSDP2, файл — полный state dict.

        load_checkpoint стратегии не годится: после fit он тянет и состояние оптимизатора, которого при save_weights_only в чекпоинте нет.
        set_model_state_dict сам раскладывает полный state dict по шардам FSDP1 и FSDP2 либо кладёт в обычный модуль.
        """
        if path.is_dir():
            state = {"state_dict": get_model_state_dict(self)}
            dcp_load(state, checkpoint_id=str(path))
            set_model_state_dict(
                self, state["state_dict"], options=StateDictOptions(strict=True)
            )
        else:
            state_dict = torch.load(
                path, map_location="cpu", mmap=True, weights_only=False
            )["state_dict"]
            set_model_state_dict(
                self,
                state_dict,
                options=StateDictOptions(full_state_dict=True, strict=True),
            )

    def best_model(self) -> DinoV3ForWine | None:
        """Лучший чекпоинт собранной с рангов моделью в float32 на CPU: на ранге 0 модель, на остальных None. Зовётся после fit на всех рангах.

        Под FSDP веса лежат шардами, поэтому лучшие веса сначала грузятся в живую модель на всех рангах, затем полный state dict снимается
        через DCP API: он одинаково собирает FSDP1, FSDP2 и обычный модуль. Без лучшего чекпоинта — валидация не прошла ни разу — уходят
        веса на момент остановки.
        """
        checkpoint = self.trainer.checkpoint_callback
        path = (
            checkpoint.best_model_path
            if isinstance(checkpoint, ModelCheckpoint) and checkpoint.best_model_path
            else None
        )
        if path:
            logger.info(f"Собираю лучший чекпоинт {path}")
            self.load_weights(Path(path))
        else:
            logger.warning(
                "Лучшего чекпоинта нет: валидация не прошла ни разу, собираю веса на момент остановки"
            )
        full_state = get_model_state_dict(
            self.model,  # ty: ignore[invalid-argument-type]
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        if not self.trainer.is_global_zero:
            return None
        model = DinoV3ForWine._from_config(
            self.model_config, attn_implementation="sdpa", dtype=torch.float32
        )  # flash attention на CPU не собирается, а веса от него не зависят
        model.load_state_dict(full_state, strict=True)
        return model.eval()

    @property
    def datamodule(self) -> WineDataModule:
        """Датамодуль трейнера."""
        if self.trainer is None or self.trainer.datamodule is None:  # ty: ignore[unresolved-attribute]
            raise ValueError(
                "Trainer and datamodule must be set before accessing datamodule."
            )
        return cast(WineDataModule, self.trainer.datamodule)  # ty: ignore[unresolved-attribute]

    @property
    def writer(self) -> SummaryWriter | None:
        """SummaryWriter TensorBoard, если такой логгер подключён."""
        for lightning_logger in self.loggers:
            if isinstance(lightning_logger, TensorBoardLogger):
                return lightning_logger.experiment
        return None

    @override
    @property
    def model_instance(self) -> PreTrainedModel:
        if self.model is None:
            raise ValueError("Model is not configured. Call `configure_model()` first.")
        if isinstance(self.model, FSDP):
            return self.model.module
        return self.model

    @override
    @property
    def model_config(self) -> PreTrainedConfig:
        return self.model_instance.config

    @property
    def data_parallel_group(self) -> dist.ProcessGroup | None:
        """Группа процессов параллелизма по данным."""
        return dist.group.WORLD if dist.is_initialized() else None

    @override
    def configure_optimizers(self) -> dict[str, Any] | torch.optim.Optimizer:  # ty: ignore[invalid-method-override]
        self.model.train()  # from_pretrained отдаёт модель в eval, а Lightning режим обучения сам не включает: иначе BatchNorm головы учился бы по замороженным статистикам  # ty: ignore[unresolved-attribute]

        hp = self.hyperparams
        if dist.is_initialized():
            lrs = {
                name: value
                for name in ("base_value", "final_value", "warmup_value")
                if (value := getattr(hp.lr, name)) is not None
            }
            for name, value in scale_lrs_by_world_size(
                lrs=lrs,
                verbosity_level="rank-zero-only",
                group=self.data_parallel_group,
            ).items():
                setattr(hp.lr, name, value)
        total_steps = estimate_total_steps(
            trainer=self.trainer,
            dp_process_group=self.data_parallel_group,
            verbosity_level="rank-zero-only",
        )
        freeze_ratio = hp.backbone_freeze_ratio or 0.0
        groups = create_param_groups(
            self,
            lr=hp.lr.base_value,
            weight_decay=hp.weight_decay.base_value,
            backbone_prefix="model.backbone.",
            backbone_lr_multiplier=hp.backbone_lr_multiplier,
            backbone_frozen=freeze_ratio > 0,
        )
        optim = create_optimizer(
            parameters_groups=groups,
            optimizer_config=hp.optimizer,
            lr=hp.lr.base_value,
            weight_decay=hp.weight_decay.base_value,
        )

        schedulers: dict[str, BaseScheduler] = {}
        if hp.lr.scheduler_type is not None and freeze_ratio > 0:
            # бэкбон стоит первые freeze_ratio шагов, затем свой разогрев и косинус; голова и центры идут по обычному расписанию с первого шага
            schedulers["backbone_lr"] = create_scheduler(
                config=hp.lr,
                optim=optim,
                num_iters=total_steps,
                param_group_field="lr",
                apply_if_field="is_backbone",
                multiplier_field="lr_multiplier",
                freeze_ratio=freeze_ratio,
            )
            schedulers["head_lr"] = create_scheduler(
                config=hp.lr,
                optim=optim,
                num_iters=total_steps,
                param_group_field="lr",
                ignore_if_field="is_backbone",
                multiplier_field="lr_multiplier",
            )
        elif hp.lr.scheduler_type is not None:
            scheduler = create_scheduler(
                config=hp.lr,
                optim=optim,
                num_iters=total_steps,
                param_group_field="lr",
                multiplier_field="lr_multiplier",
            )
            schedulers[scheduler.param_name] = scheduler
        if hp.weight_decay.scheduler_type is not None:
            scheduler = create_scheduler(
                config=hp.weight_decay,
                optim=optim,
                num_iters=total_steps,
                param_group_field="weight_decay",
                skip_if_zero=True,
            )
            schedulers[scheduler.param_name] = scheduler
        if not schedulers:
            return optim
        scheduler = (
            CompositeScheduler(optimizer=optim, **schedulers)
            if len(schedulers) > 1
            else next(iter(schedulers.values()))
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @override
    def lr_scheduler_step(self, scheduler: BaseScheduler, metric: Any | None) -> None:  # ty: ignore[invalid-method-override]
        scheduler.step(self.global_step)

    def _log_train_images(self, pixel_values: torch.Tensor) -> None:
        """Первый батч эпохи картинкой в TensorBoard: так видно, что аугментации делают с входом на самом деле."""
        every = self.config.data.log_train_images_every_n_epochs
        writer = self.writer
        if (
            every is None
            or writer is None
            or self.current_epoch % every
            or not self.trainer.is_global_zero
        ):
            return
        mean = torch.tensor(self.config.data.img_mean, device=pixel_values.device)[
            :, None, None
        ]
        std = torch.tensor(self.config.data.img_std, device=pixel_values.device)[
            :, None, None
        ]
        images = (pixel_values[:TRAIN_IMAGES_IN_GRID].float() * std + mean).clamp(0, 1)
        writer.add_image(
            "train/augmented_batch", make_grid(images, nrow=8), self.global_step
        )

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        if self.model is None or self.loss_fn is None:
            raise ValueError("Model is not configured. Call `configure_model()` first.")
        if batch_idx == 0:
            self._log_train_images(batch["pixel_values"])
        embeddings = self.model(batch["pixel_values"], normalize=False)
        with torch.autocast(
            device_type=self.device.type, enabled=False
        ):  # acos и экспонента со scale 30 в половинной точности ломаются
            loss = self.loss_fn(embeddings.float(), batch["label"])
        self.log(
            "train/loss",
            loss.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "train/gem_p",
            self.model.gem.p.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "train_loss",
            loss.detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
            sync_dist=True,
            sync_dist_group=self.data_parallel_group,
        )
        return loss

    @override
    def validation_step(
        self, batch: Batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if self.model is None:
            raise ValueError("Model is not configured. Call `configure_model()` first.")
        self._val_embeddings[dataloader_idx].append(
            self.model(batch["pixel_values"], normalize=True)
        )
        self._val_indices[dataloader_idx].append(batch["index"])

    def _gathered(self, loader: int, total: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Эмбеддинги даталоадера со всех рангов и номера картинок, которым они принадлежат, без повторов.

        DistributedSampler добивает куски рангов повторами до равной длины, поэтому all_gather здесь законен, а повторы снимаются по номеру картинки.
        С limit_val_batches часть картинок не приходит вовсе: метрики считаются по тем, что пришли.
        """
        if not self._val_embeddings[loader]:
            return torch.empty(
                0, self.config.model.embed_dim, device=self.device
            ), torch.empty(0, dtype=torch.long, device=self.device)
        embeddings = cast(
            torch.Tensor, self.all_gather(torch.cat(self._val_embeddings[loader]))
        ).flatten(0, -2)
        indices = cast(
            torch.Tensor, self.all_gather(torch.cat(self._val_indices[loader]))
        ).flatten()
        table = torch.zeros(total, embeddings.shape[1], device=embeddings.device)
        seen = torch.zeros(total, dtype=torch.bool, device=embeddings.device)
        table[indices], seen[indices] = embeddings, True
        rows = torch.nonzero(seen).squeeze(1)
        return table[rows], rows

    @override
    def on_validation_epoch_end(self) -> None:
        datamodule = self.datamodule
        gathered = {
            name: self._gathered(n, len(datamodule.val_items[name]))
            for n, name in enumerate(VAL_LOADERS)
        }
        for store in (*self._val_embeddings, *self._val_indices):
            store.clear()
        if self.trainer.sanity_checking:
            return
        catalog, catalog_rows = gathered["catalog"]
        queries, query_rows = gathered["catalog_queries"]
        both = (
            torch.isin(catalog_rows, query_rows),
            torch.isin(query_rows, catalog_rows),
        )  # пары «чистое фото — его аугментированный вид»
        report = evaluate_retrieval(
            val=gathered["val"][0],
            labels=datamodule.val_labels.to(self.device)[gathered["val"][1]],
            distractors=gathered["distractors"][0],
            negatives=gathered["negatives"][0],
            catalog=catalog[both[0]],
            catalog_queries=queries[both[1]],
        )
        # значения на всех рангах одинаковы: метрики посчитаны по собранным со всех рангов эмбеддингам, усреднение их не меняет
        self.log_dict(
            {f"val/{name}": value for name, value in report.scalars.items()},
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "val_recall1",
            report.scalars.get(MONITOR_METRIC, 0.0),
            prog_bar=True,
            logger=False,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        writer = self.writer
        if writer is not None and self.trainer.is_global_zero and report.top1:
            for name, values in report.top1.items():
                if len(values):
                    writer.add_histogram(
                        f"cos_top1/{name}", values.float().cpu(), self.global_step
                    )
            figure = scores_figure(
                report.top1,
                f"Косинус top-1 по группам запросов · эпоха {self.current_epoch}",
            )
            writer.add_figure("cos_top1/groups", figure, self.global_step)
            plt.close(figure)
            self._log_misses(
                report.misses, gathered["val"][1], gathered["negatives"][1]
            )
        if dist.is_initialized():
            dist.barrier()

    def _log_misses(
        self, misses: Misses | None, val_rows: torch.Tensor, negative_rows: torch.Tensor
    ) -> None:
        """Несколько случайных промахов wine/oneshot_noisy картинкой: запрос, фото его класса из галереи и top-5 найденного с косинусами.

        Строки в misses — по собранным эмбеддингам, val_rows и negative_rows переводят их в номера картинок датасетов. Картинки читаются
        заново с диска ровно так, как их видела модель. Выборка своя в каждую эпоху, чтобы не смотреть один и тот же десяток фото.
        """
        limit = self.config.data.eval.log_misses
        writer = self.writer
        if misses is None or not limit or writer is None or not len(misses.queries):
            return
        if self.datamodule.val_datasets is None:
            raise RuntimeError("setup не вызывался")
        val, negatives = (
            self.datamodule.val_datasets[VAL_LOADERS.index(name)]
            for name in ("val", "negatives")
        )
        labels = val.items.labels
        val_rows, negative_rows = val_rows.cpu(), negative_rows.cpu()
        generator = torch.Generator().manual_seed(self.current_epoch)
        picked = torch.randperm(len(misses.queries), generator=generator)[
            :limit
        ].tolist()
        rows: list[list[Panel]] = []
        for i in picked:
            query, expected = (
                int(val_rows[misses.queries[i]]),
                int(val_rows[misses.expected[i]]),
            )
            ranks = torch.nonzero(misses.hits[i]).flatten()
            rank = f"ранг {int(ranks[0]) + 1}" if len(ranks) else f"ранг >{TOP}"
            panels = [
                Panel(val.image(query), f"запрос · {labels[query]}"),
                Panel(val.image(expected), f"в галерее · {rank}"),
            ]
            for k in range(misses.found.shape[1]):
                score = f"{float(misses.scores[i, k]):.3f}"
                if bool(misses.found_is_noise[i, k]):
                    panels.append(
                        Panel(
                            negatives.image(int(negative_rows[misses.found[i, k]])),
                            f"{score} · не вино",
                            NOISE_COLOR,
                        )
                    )
                else:
                    row = int(val_rows[misses.found[i, k]])
                    panels.append(
                        Panel(
                            val.image(row),
                            f"{score} · {labels[row]}",
                            HIT_COLOR if bool(misses.hits[i, k]) else MISS_COLOR,
                        )
                    )
            rows.append(panels)
        figure = panels_figure(
            rows,
            f"Промахи wine/oneshot_noisy: {len(misses.queries)} из {misses.total} запросов · эпоха {self.current_epoch}",
        )
        writer.add_figure("misses/oneshot_noisy", figure, self.global_step)
        plt.close(figure)
