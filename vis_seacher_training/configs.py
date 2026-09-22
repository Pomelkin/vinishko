from pathlib import Path
from typing import Literal

from kostyl.ml.configs import CheckpointConfig
from kostyl.ml.configs import EarlyStoppingConfig
from kostyl.ml.configs import HyperparamsConfig as KostylHyperparamsConfig
from kostyl.ml.configs import LightningTrainerParameters
from kostyl.ml.configs.mixins import ConfigLoadingMixin
from kostyl.utils import dump_to_file
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

DATASETS = ("winesensed", "off", "products10k")
"""Что обязано лежать в data.datasets_dir: раскладка scripts/prepare_datasets.py плюс разметка scripts/normalize_dataset.py."""
REQUIRED_FILES = {
    "winesensed": ("train.json", "val.json", "val_distractors.json", "normalization.jsonl"),
    "off": ("catalog.json", "normalization.jsonl"),
    "products10k": ("negatives.json", "normalization.jsonl"),
}
Interpolation = Literal["nearest", "linear", "cubic", "area", "lanczos"]


class StrictModel(BaseModel):
    """Опечатка в имени параметра конфига — ошибка, а не молча проигнорированная строка."""

    model_config = ConfigDict(extra="forbid")


class ViewAugConfig(StrictModel):
    """Геометрия вырезки: то, как нормализатор мог бы вырезать эту же бутылку иначе."""

    pad_y_range: tuple[float, float] = (0.0, 0.6)
    """Запас окна над и под этикеткой в долях её высоты, сверху и снизу независимо. Пайплайн режет ровно 0.2."""
    full_bottle_prob: float = Field(default=0.05, ge=0, le=1)
    """Доля примеров с окном во всю бутылку."""
    edge_jitter_prob: float = Field(default=0.8, ge=0, le=1)
    edge_jitter: float = Field(default=0.08, ge=0, lt=0.4)
    """Шум окна по высоте: верхняя и нижняя грани сдвигаются на ±долю его высоты, наружу либо внутрь."""
    edge_jitter_x: float = Field(default=0.04, ge=0, lt=0.4)
    """Шум окна по ширине: боковые грани сдвигаются только наружу, до этой доли ширины. Внутрь нельзя: пайплайн берёт ширину по маске и бутылку по бокам не срезает."""
    rotate_prob: float = Field(default=0.6, ge=0, le=1)
    rotate_deg: float = Field(default=10.0, ge=0, le=45)
    """Ошибка выравнивания: к углу из разметки добавляется ±столько градусов."""
    keep_background_prob: float = Field(default=0.2, ge=0, le=1)
    """Доля примеров с оригинальным фоном вместо заливки."""
    mask_jitter: tuple[float, float] = (0.5, 2.0)
    """Во сколько раз меняются расширение маски и растушёвка её края."""


class PerspectiveAugConfig(StrictModel):
    """Съёмка не строго анфас."""

    prob: float = Field(default=0.4, ge=0, le=1)
    scale: tuple[float, float] = (0.02, 0.07)


class OcclusionAugConfig(StrictModel):
    """Перекрытия: палец, ценник, соседняя бутылка."""

    prob: float = Field(default=0.35, ge=0, le=1)
    max_occluders: int = Field(default=3, ge=1)
    size_range: tuple[float, float] = (0.08, 0.35)
    """Сторона перекрытия в долях стороны кропа."""
    min_label_visible: float = Field(default=0.6, gt=0, le=1)
    min_bottle_visible: float = Field(default=0.6, gt=0, le=1)
    """Перекрытие, после которого видимая доля этикетки или бутылки упала бы ниже порога, не ставится."""
    cut_from_mask_prob: float = Field(default=0.5, ge=0, le=1)
    """Доля перекрытий, которые сегментация от бутылки отрезала: на их месте после заливки фон."""


class LabelNoiseAugConfig(StrictModel):
    """Порча самой этикетки: расфокус, смаз, шум только в её маске."""

    prob: float = Field(default=0.4, ge=0, le=1)


class MaskDefectsAugConfig(StrictModel):
    """Огрехи сегментации: маска бутылки неполная либо с неровным краем. Проявляются через заливку фона, поэтому идут прямо перед ней."""

    prob: float = Field(default=0.3, ge=0, le=1)
    truncate_prob: float = Field(default=0.4, ge=0, le=1)
    max_truncate: float = Field(default=0.6, ge=0, le=1)
    """Маска обрывается сверху либо снизу: до этой доли её высоты в кропе уходит в фон, но не глубже min_label_visible."""
    min_label_visible: float = Field(default=0.5, ge=0, le=1)
    side_cut_prob: float = Field(default=0.5, ge=0, le=1)
    """Проверяется, только если обрыва не случилось: вместе они оставляли от бутылки лоскут."""
    max_side_cuts: int = Field(default=2, ge=1)
    side_cut_size: tuple[float, float] = (0.05, 0.25)
    """Выкусы с боков маски: эллипсы с центром на её границе, размер в долях высоты и ширины кропа."""
    rough_prob: float = Field(default=0.5, ge=0, le=1)
    rough_sigma: tuple[float, float] = (0.01, 0.03)
    rough_amp: tuple[float, float] = (0.15, 0.4)
    """Неровный край: граница маски гуляет по низкочастотному шуму; sigma — ширина полосы в долях большей стороны, amp — сила шума."""


class GlareAugConfig(StrictModel):
    """Блик на стекле и глянцевой этикетке."""

    prob: float = Field(default=0.3, ge=0, le=1)
    max_blobs: int = Field(default=2, ge=1)
    intensity: tuple[float, float] = (0.35, 0.95)
    size_range: tuple[float, float] = (0.05, 0.25)
    strip_prob: float = Field(default=0.4, ge=0, le=1)
    """Доля бликов в виде узкой полосы вдоль бутылки: отражение ламп магазина в стекле; остальные — мягкие пятна."""
    strip_width: tuple[float, float] = (0.02, 0.07)
    """Ширина полосы в долях ширины кропа."""


class ExposureAugConfig(StrictModel):
    """Съёмка в полутьме либо пересвет: кадр меняет яркость равномерно или с перепадом к одному краю."""

    prob: float = Field(default=0.25, ge=0, le=1)
    dark_prob: float = Field(default=0.65, ge=0, le=1)
    """Доля затемнений среди срабатываний; остальные — пересвет."""
    dark_gain: tuple[float, float] = (0.25, 0.7)
    """Какая доля яркости остаётся в самом тёмном месте кадра."""
    bright_gain: tuple[float, float] = (1.3, 2.2)
    """Во сколько раз ярче самое светлое место кадра; светлое выбивается в белое."""
    bright_lift: float = Field(default=40.0, ge=0)
    """Подъём чёрного при пересвете, до стольких уровней: тени выцветают, контраст падает."""
    gradient_prob: float = Field(default=0.5, ge=0, le=1)
    """Доля срабатываний с перепадом к краю; у остальных кадр меняется равномерно."""


class PhotometricAugConfig(StrictModel):
    """Свет, баланс белого, шум матрицы, смаз всего кадра."""

    color_prob: float = Field(default=0.8, ge=0, le=1)
    white_balance_prob: float = Field(default=0.3, ge=0, le=1)
    shadow_prob: float = Field(default=0.15, ge=0, le=1)
    shake_prob: float = Field(default=0.05, ge=0, le=1)
    """Сильный смаз дрогнувшей руки; лёгкий смаз входит в blur_prob."""
    rare_color_prob: float = Field(default=0.03, ge=0, le=1)
    """Сильный сдвиг тона и насыщенности. Редко: тон отличает вина одной серии, но баланс белого телефона иногда промахивается сильно."""
    blur_prob: float = Field(default=0.25, ge=0, le=1)
    noise_prob: float = Field(default=0.25, ge=0, le=1)
    sharpen_prob: float = Field(default=0.1, ge=0, le=1)


class DegradeAugConfig(StrictModel):
    """Потеря качества: далёкая бутылка, растянутая обратно, и JPEG."""

    downscale_prob: float = Field(default=0.35, ge=0, le=1)
    downscale_range: tuple[float, float] = (0.12, 0.8)
    """Во сколько раз кроп уменьшается перед растяжением обратно; масштаб берётся лог-равномерно, чтобы сильный шакал не был редкостью."""
    small_jpeg_prob: float = Field(default=0.5, ge=0, le=1)
    small_jpeg_quality: tuple[int, int] = (30, 80)
    """JPEG на уменьшенном кадре: так пережимает камера, когда бутылка занимает в кадре сотню пикселей."""
    jpeg_prob: float = Field(default=0.4, ge=0, le=1)
    jpeg_quality: tuple[int, int] = (25, 90)


class AugmentationsConfig(StrictModel):
    """Аугментации обучения. Порядок применения — в TrainAugmenter."""

    enabled: bool = True
    view: ViewAugConfig = ViewAugConfig()
    perspective: PerspectiveAugConfig = PerspectiveAugConfig()
    occlusion: OcclusionAugConfig = OcclusionAugConfig()
    label_noise: LabelNoiseAugConfig = LabelNoiseAugConfig()
    glare: GlareAugConfig = GlareAugConfig()
    mask_defects: MaskDefectsAugConfig = MaskDefectsAugConfig()
    exposure: ExposureAugConfig = ExposureAugConfig()
    photometric: PhotometricAugConfig = PhotometricAugConfig()
    degrade: DegradeAugConfig = DegradeAugConfig()
    resize_interpolations: list[Interpolation] = ["nearest", "linear", "cubic", "area", "lanczos"]
    """Чем кроп приводится к размеру входа; в обучении выбирается случайно, на валидации area при уменьшении и cubic при увеличении."""
    random_pad_position: bool = True
    """Случайно сдвигать картинку внутри полей входа, а не держать по центру."""


class EvalConfig(StrictModel):
    """Сколько данных идёт в валидацию: полная val на 512 px — это 110 тыс. прямых проходов на каждую проверку."""

    max_val_images: int | None = Field(default=20000, gt=0)
    """Картинок winesensed/val.json; классы выбираются случайно, но воспроизводимо, и берутся целиком."""
    max_images_per_class: int | None = Field(default=8, gt=1)
    """Сколько картинок брать от класса val. Без потолка несколько вин с сотнями фото дали бы большую часть запросов."""
    max_distractors: int | None = Field(default=5000, gt=0)
    max_negatives: int | None = Field(default=5000, gt=0)
    """Картинок products10k; половина подмешивается шумом в галерею, половина идёт запросами без ответа."""
    off_augment_seed: int = 0
    """Сид синтетических запросов OFF: у каждой картинки каталога одна и та же аугментация от эпохи к эпохе."""


class DataConfig(StrictModel):
    """Данные и вход модели."""

    datasets_dir: Path
    normalize_config: Path | None = None
    """Конфиг нормализации с параметрами кропа и фона; по умолчанию normalize.toml пайплайна."""
    input_size: tuple[int, int] = (512, 512)
    """Высота и ширина входа. Кроп вписывается с сохранением пропорций, поля — нули после нормировки."""
    img_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    img_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    batch_size: int = Field(gt=0)
    eval_batch_size: int | None = Field(default=None, gt=0)
    num_workers: int = Field(ge=0)
    max_train_images: int | None = Field(default=None, gt=0)
    """Для отладки: первые N картинок train.json, классы целиком."""
    augmentations: AugmentationsConfig = AugmentationsConfig()
    eval: EvalConfig = EvalConfig()
    log_train_images_every_n_epochs: int | None = Field(default=1, gt=0)
    """Раз в столько эпох первый батч обучения уходит картинкой в TensorBoard: так видно, что делают аугментации."""

    @model_validator(mode="after")
    def _validate_datasets(self) -> "DataConfig":
        for name in DATASETS:
            root = self.datasets_dir / name
            if not (root / "images").is_dir():
                raise ValueError(f"в {self.datasets_dir} нет датасета {name}: ожидается директория {root / 'images'}, раскладка scripts/prepare_datasets.py")
            missing = [f for f in REQUIRED_FILES[name] if not (root / f).is_file()]
            if missing:
                raise ValueError(f"в {root} нет {missing}; normalization.jsonl пишет scripts/normalize_dataset.py")
        return self


class ModelConfig(StrictModel):
    """Архитектура DinoV3ForWine."""

    backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    patch_size: int = 16
    """Патч бэкбона; по нему data.input_size проверяется при чтении конфига, а при сборке модели сверяется с её настоящим патчем."""
    embed_dim: int = Field(default=512, gt=0)
    gem_p: float = Field(default=3.0, ge=1)
    attn_implementation: Literal["sdpa", "flash_attention_2", "flex_attention", "eager"] | None = None
    """Реализация внимания бэкбона; без значения transformers берёт sdpa. flash_attention_2 работает только в половинной точности, то есть с precision bf16-mixed либо 16-mixed."""
    freeze_backbone: bool = False
    gradient_checkpointing: bool = False


class LossConfig(StrictModel):
    """Sub-center ArcFace с отступом по размеру класса: m_c = m_a · n_c^(−m_lambda) + m_b."""

    k: int = Field(default=3, ge=1)
    s: float = Field(default=30.0, gt=0)
    m_a: float = 0.5
    m_b: float = 0.05
    m_lambda: float = 0.25


class HyperparamsConfig(KostylHyperparamsConfig):
    """Гиперпараметры обучения."""

    backbone_lr_multiplier: float | None = Field(default=None, gt=0)
    """Во сколько раз скорость обучения бэкбона отличается от скорости головы и лосса."""


class TrainingConfig(BaseModel, ConfigLoadingMixin):
    """Конфиг эксперимента: <experiment-dir>/config.yaml."""

    model_config = ConfigDict(extra="forbid")

    desc: str | None = None
    seed: int = 42
    trainer_params: LightningTrainerParameters
    early_stopping: EarlyStoppingConfig | None = None
    checkpointing: CheckpointConfig
    hyperparams: HyperparamsConfig
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    data: DataConfig

    @model_validator(mode="after")
    def _validate_input_size(self) -> "TrainingConfig":
        height, width = self.data.input_size
        patch = self.model.patch_size
        if height % patch or width % patch:
            raise ValueError(f"data.input_size={height}×{width} не делится нацело на model.patch_size={patch}")
        return self

    def dump_to_file(self, path: Path) -> None:
        """Сохраняет конфиг рядом с результатами прогона."""
        path.parent.mkdir(parents=True, exist_ok=True)
        dump_to_file(self.model_dump(mode="json"), path)
