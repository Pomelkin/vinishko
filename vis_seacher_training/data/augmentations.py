import albumentations as A
import cv2
import numpy as np

from vis_seacher_training.configs import AugmentationsConfig
from vis_seacher_training.data.kernels import apply_gain
from vis_seacher_training.data.kernels import blend_images
from vis_seacher_training.data.kernels import gain_gradient
from vis_seacher_training.data.kernels import screen_gauss
from vis_seacher_training.data.kernels import screen_map
from vis_seacher_training.data.views import View
from vis_seacher_training.data.views import ViewParams
from vis_seacher_training.data.views import mute_background
from vis_seacher_training.data.views import render_view

INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}
FULL_BOTTLE_PAD = 100.0  # запас заведомо больше бутылки: окно упирается в её маску
OCCLUDER_ATTEMPTS = 6
# чем уменьшать и чем растягивать обратно: area — как сенсор камеры, nearest и linear — быстрые и с алиасингом
DOWNSCALE_PAIRS = [
    (cv2.INTER_NEAREST, cv2.INTER_NEAREST),
    (cv2.INTER_AREA, cv2.INTER_LINEAR),
    (cv2.INTER_LINEAR, cv2.INTER_CUBIC),
    (cv2.INTER_AREA, cv2.INTER_NEAREST),
    (cv2.INTER_AREA, cv2.INTER_CUBIC),
    (cv2.INTER_LINEAR, cv2.INTER_LINEAR),
]
MIN_SMALL_PX = 8


def fit_size(height: int, width: int, target: tuple[int, int]) -> tuple[int, int]:
    """Размер картинки, вписанной в target с сохранением пропорций: высота и ширина."""
    scale = min(target[0] / height, target[1] / width)
    return min(target[0], max(1, round(height * scale))), min(
        target[1], max(1, round(width * scale))
    )


def resize_to_fit(
    image: np.ndarray, target: tuple[int, int], interpolation: int | None = None
) -> np.ndarray:
    """Вписывает картинку в target, пропорции сохраняются. Без interpolation — как в пайплайне: area при уменьшении, cubic при увеличении."""
    height, width = fit_size(image.shape[0], image.shape[1], target)
    if interpolation is None:
        interpolation = cv2.INTER_AREA if height < image.shape[0] else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interpolation)


class TrainAugmenter:
    """Аугментации обучения: фото и разметка бутылки → кроп, вписанный в размер входа.

    Порядок повторяет жизнь кадра. Сначала то, что случается при съёмке: как нормализатор вырезал бутылку (запас вокруг этикетки,
    шум бокса, ошибка выравнивания), перспектива, перекрытия, порча этикетки, блик, полутьма или пересвет, свет и шум, потеря качества. Затем то, что пайплайн
    делает с уже снятым кадром: заглушение фона по маске с огрехами сегментации, у части примеров пропускается, и ресайз случайной интерполяцией.
    Фон заливается после цветовых аугментаций, иначе он перестал бы быть нулём на входе модели.
    """

    def __init__(
        self,
        cfg: AugmentationsConfig,
        render_cfg: dict,
        input_size: tuple[int, int],
        seed: int = 0,
    ) -> None:
        self.cfg, self.render_cfg, self.input_size = cfg, render_cfg, input_size
        self.default_pad = float(render_cfg["crop"]["padding_y"])
        fill = tuple(render_cfg["background"]["color"])
        ph, dg = cfg.photometric, cfg.degrade
        # fit_output: без него Perspective увеличивает кадр и срезает бутылку прямой гранью окна, чего пайплайн по бокам не делает
        self.perspective = A.Compose(
            [
                A.Perspective(
                    scale=cfg.perspective.scale,
                    keep_size=True,
                    fit_output=True,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=fill,
                    fill_mask=0,
                    p=cfg.perspective.prob,
                )
            ]
        )
        self.label_noise = A.Compose(
            [
                A.OneOf(
                    [
                        A.GaussianBlur(sigma_limit=(1.0, 4.0), p=1),
                        A.MotionBlur(blur_limit=(5, 15), p=1),
                        A.Defocus(radius=(2, 5), p=1),
                        A.GaussNoise(std_range=(0.03, 0.09), p=1),
                        A.ISONoise(
                            color_shift=(0.01, 0.04), intensity=(0.1, 0.45), p=1
                        ),
                        A.ImageCompression(quality_range=(10, 40), p=1),
                        A.SaltAndPepper(amount=(0.01, 0.06), p=1),
                        # брызги и грязь на бумаге: порог задаёт плотность, от густой сыпи до редких капель; выше 0.76 пятен уже нет
                        A.Spatter(mode="mud", cutout_threshold=(0.68, 0.76), p=1),
                    ],
                    p=1,
                )
            ]
        )
        self.photometric = A.Compose(
            [
                # тон почти не трогаем: красная и белая этикетки одной серии — разные вина
                A.ColorJitter(
                    brightness=(0.6, 1.4),
                    contrast=(0.6, 1.4),
                    saturation=(0.6, 1.4),
                    hue=(-0.03, 0.03),
                    p=ph.color_prob,
                ),
                A.RandomGamma(gamma_limit=(70, 140), p=ph.color_prob * 0.4),
                # диапазон уже стандартного 3000–15000 К: на его краях оранжевая этикетка уходила в зелёную
                A.PlanckianJitter(
                    mode="blackbody",
                    temperature_limit=(4500, 8500),
                    p=ph.white_balance_prob,
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    shadow_intensity_range=(0.3, 0.6),
                    p=ph.shadow_prob,
                ),
                # редкий сильный промах баланса белого; яркость не трогает, ею занимается expose
                A.HueSaturationValue(
                    hue_shift_limit=(-25, 25),
                    sat_shift_limit=(-60, 60),
                    val_shift_limit=(0, 0),
                    p=ph.rare_color_prob,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(sigma_limit=(0.5, 2.5), p=1),
                        A.MotionBlur(blur_limit=(3, 11), p=1),
                        A.Defocus(radius=(1, 3), p=1),
                    ],
                    p=ph.blur_prob,
                ),
                A.MotionBlur(blur_limit=(13, 31), allow_shifted=True, p=ph.shake_prob),
                A.OneOf(
                    [A.GaussNoise(std_range=(0.02, 0.1), p=1), A.ISONoise(p=1)],
                    p=ph.noise_prob,
                ),
                A.Sharpen(p=ph.sharpen_prob),
            ]
        )
        self.degrade = A.Compose(
            [A.ImageCompression(quality_range=dg.jpeg_quality, p=dg.jpeg_prob)]
        )
        self.rng = np.random.default_rng(seed)
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        """Свой поток случайности. Зовётся в каждом воркере даталоадера: A.Compose держит собственный генератор, и после fork все воркеры выдавали бы одни и те же аугментации."""
        self.rng = np.random.default_rng(seed)
        for n, pipeline in enumerate(
            (self.perspective, self.label_noise, self.photometric, self.degrade)
        ):
            pipeline.set_random_seed((seed + n) % (1 << 32))

    def chance(self, prob: float) -> bool:
        """Сработает ли аугментация с вероятностью prob."""
        return bool(self.rng.random() < prob)

    def sample_view(self) -> ViewParams:
        """Случайная геометрия вырезки: запас над и под этикеткой, шум граней бокса, ошибка выравнивания."""
        view = self.cfg.view
        if self.chance(view.full_bottle_prob):
            pad_top = pad_bottom = FULL_BOTTLE_PAD
        else:
            pad_top, pad_bottom = self.rng.uniform(*view.pad_y_range, size=2)
        jitter = np.zeros(4)
        if self.chance(view.edge_jitter_prob):
            jitter[[1, 3]] = self.rng.uniform(
                -view.edge_jitter, view.edge_jitter, size=2
            )
            jitter[[0, 2]] = self.rng.uniform(0, view.edge_jitter_x, size=2)
        angle = (
            self.rng.uniform(-view.rotate_deg, view.rotate_deg)
            if self.chance(view.rotate_prob)
            else 0.0
        )
        return ViewParams(
            float(pad_top),
            float(pad_bottom),
            (float(jitter[0]), float(jitter[1]), float(jitter[2]), float(jitter[3])),
            float(angle),
        )

    def occluder_texture(
        self, image: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        """Чем закрашено перекрытие: сплошной цвет, шум либо кусок этого же кадра — он похож на соседний предмет."""
        kind = self.rng.integers(3)
        if kind == 0:
            return np.full(
                (height, width, 3), self.rng.integers(0, 256, size=3), np.uint8
            )
        if kind == 1:
            base = self.rng.integers(0, 256, size=3)
            return np.clip(
                base + self.rng.normal(0, 35, size=(height, width, 3)), 0, 255
            ).astype(np.uint8)
        h, w = image.shape[:2]
        ph, pw = max(2, min(h, height)), max(2, min(w, width))
        y, x = self.rng.integers(0, h - ph + 1), self.rng.integers(0, w - pw + 1)
        return cv2.resize(
            image[y : y + ph, x : x + pw],
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    def occlude(self, view: View) -> None:
        """Перекрытия поверх бутылки. Перекрытие не ставится, если после него видимая доля этикетки или бутылки упала бы ниже порога.

        Часть перекрытий вырезается и из масок: сегментация такой предмет от бутылки отделяет, и после заглушения на его месте фон.
        """
        cfg = self.cfg.occlusion
        H, W = view.image.shape[:2]
        label_area, bottle_area = (
            max(1, int((view.label > 0).sum())),
            max(1, int((view.bottle > 0).sum())),
        )
        ys, xs = np.nonzero(view.bottle)
        if not len(ys):
            return
        covered = np.zeros((H, W), bool)
        for _ in range(int(self.rng.integers(1, cfg.max_occluders + 1))):
            for _ in range(OCCLUDER_ATTEMPTS):
                h, w = (
                    max(2, int(s * side))
                    for s, side in zip(
                        self.rng.uniform(*cfg.size_range, size=2), (H, W), strict=True
                    )
                )
                at = self.rng.integers(len(ys))
                shape = np.zeros((H, W), np.uint8)
                if self.chance(0.5):
                    cv2.ellipse(
                        shape,
                        (int(xs[at]), int(ys[at])),
                        (max(1, w // 2), max(1, h // 2)),
                        float(self.rng.uniform(0, 180)),
                        0,
                        360,
                        255,
                        -1,
                    )
                else:
                    box = cv2.boxPoints(
                        (
                            (float(xs[at]), float(ys[at])),
                            (float(w), float(h)),
                            float(self.rng.uniform(-30, 30)),
                        )
                    )
                    cv2.fillPoly(shape, [np.round(box).astype(np.int32)], 255)
                trial = covered | (shape > 0)
                if (
                    (view.label > 0) & ~trial
                ).sum() / label_area < cfg.min_label_visible or (
                    (view.bottle > 0) & ~trial
                ).sum() / bottle_area < cfg.min_bottle_visible:
                    continue
                covered = trial
                y1, y2, x1, x2 = (
                    *np.nonzero(shape.any(axis=1))[0][[0, -1]],
                    *np.nonzero(shape.any(axis=0))[0][[0, -1]],
                )
                texture = self.occluder_texture(
                    view.image, int(y2 - y1 + 1), int(x2 - x1 + 1)
                )
                region = shape[y1 : y2 + 1, x1 : x2 + 1] > 0
                view.image[y1 : y2 + 1, x1 : x2 + 1][region] = texture[region]
                if self.chance(cfg.cut_from_mask_prob):
                    view.bottle[shape > 0] = 0
                    view.label[shape > 0] = 0
                break

    def noise_label(self, view: View) -> None:
        """Расфокус, смаз, шум либо JPEG только на этикетке: порченая версия кадра вмешивается по её растушёванной маске."""
        if not view.label.any():
            return
        spoiled = self.label_noise(image=view.image)["image"]
        alpha = cv2.GaussianBlur(view.label.astype(np.float32) / 255, (0, 0), 2.0)
        view.image = blend_images(
            np.ascontiguousarray(spoiled), np.ascontiguousarray(view.image), alpha
        )

    def add_glare(self, view: View) -> None:
        """Блики с центром на бутылке, смешение «экран»: мягкие пятна, вытянутые вдоль неё, либо узкие полосы — отражения ламп магазина в стекле."""
        cfg = self.cfg.glare
        ys, xs = np.nonzero(view.bottle)
        if not len(ys):
            return
        H, W = view.image.shape[:2]
        image = view.image.astype(np.float32)
        for _ in range(int(self.rng.integers(1, cfg.max_blobs + 1))):
            at = self.rng.integers(len(ys))
            intensity = float(self.rng.uniform(*cfg.intensity))
            if self.chance(cfg.strip_prob):
                w = max(2.0, float(self.rng.uniform(*cfg.strip_width)) * W)
                h = float(self.rng.uniform(0.3, 1.2)) * H
                strip = np.zeros((H, W), np.float32)
                box = cv2.boxPoints(
                    (
                        (float(xs[at]), float(ys[at])),
                        (w, h),
                        float(self.rng.uniform(-8, 8)),
                    )
                )
                cv2.fillPoly(strip, [np.round(box).astype(np.int32)], 1.0)
                screen_map(
                    image,
                    cv2.GaussianBlur(strip, (0, 0), max(0.5, w * 0.3)) * intensity,
                )
            else:
                sx = max(1.0, float(self.rng.uniform(*cfg.size_range)) * W * 0.5)
                sy = sx * float(self.rng.uniform(1.0, 4.0))
                screen_gauss(image, int(xs[at]), int(ys[at]), sx, sy, intensity)
        view.image = image.round().astype(np.uint8)

    def shrink(self, image: np.ndarray) -> np.ndarray:
        """Далёкая бутылка: кроп уменьшается быстрым методом, пережимается JPEG на малом размере и растягивается обратно, как это сделал бы пайплайн с мелким кропом."""
        cfg = self.cfg.degrade
        h, w = image.shape[:2]
        scale = float(np.exp(self.rng.uniform(*np.log(cfg.downscale_range))))
        down, up = DOWNSCALE_PAIRS[int(self.rng.integers(len(DOWNSCALE_PAIRS)))]
        small = cv2.resize(
            image,
            (max(MIN_SMALL_PX, round(w * scale)), max(MIN_SMALL_PX, round(h * scale))),
            interpolation=down,
        )
        if self.chance(cfg.small_jpeg_prob):
            quality = int(
                self.rng.integers(
                    cfg.small_jpeg_quality[0], cfg.small_jpeg_quality[1] + 1
                )
            )
            _, encoded = cv2.imencode(
                ".jpg",
                cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError(
                    f"JPEG {small.shape[1]}×{small.shape[0]} с качеством {quality} не раскодировался"
                )
            small = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return cv2.resize(small, (w, h), interpolation=up)

    def spoil_mask(self, view: View) -> None:
        """Огрехи сегментации в маске бутылки: обрыв сверху или снизу либо выкусы с боков, и независимо от них неровный край. Заливка фона потом съест эти места."""
        cfg = self.cfg.mask_defects
        H, W = view.bottle.shape
        rows = np.nonzero(view.bottle.any(axis=1))[0]
        if not len(rows):
            return
        if self.chance(cfg.truncate_prob):
            # маска покрыла бутылку не целиком; срез не глубже, чем оставляет min_label_visible этикетки
            from_top = self.chance(0.5)
            label_rows = (view.label > 0).sum(axis=1).astype(np.float64)
            cut = int(self.rng.uniform(0, cfg.max_truncate) * len(rows))
            if label_rows.sum() > 0:
                cum = np.cumsum(label_rows if from_top else label_rows[::-1])
                allowed = int(
                    np.searchsorted(
                        cum,
                        (1 - cfg.min_label_visible) * label_rows.sum(),
                        side="right",
                    )
                )
                cut = min(
                    cut, max(0, allowed - (rows[0] if from_top else H - 1 - rows[-1]))
                )
            if cut > 0:
                edge = rows[0] + cut if from_top else rows[-1] - cut
                tilt = np.tan(np.radians(self.rng.uniform(-10, 10))) * W / 2
                far = -1 if from_top else H
                cv2.fillPoly(
                    view.bottle,
                    [
                        np.array(
                            [
                                [0, far],
                                [W, far],
                                [W, round(edge + tilt)],
                                [0, round(edge - tilt)],
                            ],
                            np.int32,
                        )
                    ],
                    0,
                )
        elif self.chance(cfg.side_cut_prob):
            # обрыв и выкусы вместе оставляли от бутылки лоскут, поэтому либо одно, либо другое
            for _ in range(int(self.rng.integers(1, cfg.max_side_cuts + 1))):
                y = int(self.rng.choice(rows))
                xs = np.nonzero(view.bottle[y])[0]
                if not len(xs):
                    continue
                x = int(xs[0] if self.chance(0.5) else xs[-1])
                h, w = (
                    int(self.rng.uniform(*cfg.side_cut_size) * side) for side in (H, W)
                )
                cv2.ellipse(
                    view.bottle,
                    (x, y),
                    (max(1, w // 2), max(1, h // 2)),
                    float(self.rng.uniform(0, 180)),
                    0,
                    360,
                    0,
                    -1,
                )
        if self.chance(cfg.rough_prob):
            # граница гуляет по низкочастотному шуму; внутрь и наружу дальше полосы размытия шум не достаёт
            sigma = float(self.rng.uniform(*cfg.rough_sigma)) * max(H, W)
            cells = max(2, round(max(H, W) / (4 * sigma)))
            noise = cv2.resize(
                self.rng.normal(
                    size=(
                        max(2, round(H * cells / max(H, W))),
                        max(2, round(W * cells / max(H, W))),
                    )
                ).astype(np.float32),
                (W, H),
                interpolation=cv2.INTER_CUBIC,
            )
            soft = cv2.GaussianBlur(view.bottle.astype(np.float32) / 255, (0, 0), sigma)
            band = (soft > 0.001) & (soft < 0.999)
            rough = soft + float(self.rng.uniform(*cfg.rough_amp)) * noise > 0.5
            view.bottle = np.where(band, rough, soft >= 0.999).astype(np.uint8) * 255

    def expose(self, view: View) -> None:
        """Полутьма либо пересвет: коэффициент яркости один на кадр или растёт к случайному краю. Идёт до шума матрицы: в тёмном кадре шум остаётся заметным."""
        cfg = self.cfg.exposure
        dark = self.chance(cfg.dark_prob)
        if dark:
            low = float(self.rng.uniform(*cfg.dark_gain))
            high = float(
                self.rng.uniform(low, 1.0)
            )  # светлый край тоже не обязан остаться как в оригинале
        else:
            high = float(self.rng.uniform(*cfg.bright_gain))
            low = float(self.rng.uniform(1.0, high))
        H, W = view.image.shape[:2]
        if self.chance(cfg.gradient_prob):
            angle = float(self.rng.uniform(0, 2 * np.pi))
            gain = gain_gradient(
                H, W, float(np.cos(angle)), float(np.sin(angle)), low, high
            )
        else:
            gain = np.full((H, W), low if dark else high, np.float32)
        lift = 0.0 if dark else float(self.rng.uniform(0, cfg.bright_lift))
        view.image = apply_gain(
            np.ascontiguousarray(view.image),
            gain,
            np.array(lift, gain.dtype),
            np.array(max(high - 1, 1e-6), gain.dtype),
        )

    def __call__(
        self, rgb: np.ndarray, cand: dict
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """Кроп uint8, вписанный в размер входа, и его положение внутри полей: доли свободного места сверху и слева."""
        cfg = self.cfg
        view = render_view(rgb, cand, self.render_cfg, self.sample_view())
        warped = self.perspective(
            image=view.image, masks=[view.bottle, view.label, view.inside]
        )
        view = View(
            warped["image"], *(np.ascontiguousarray(mask) for mask in warped["masks"])
        )
        if self.chance(cfg.occlusion.prob):
            self.occlude(view)
        if self.chance(cfg.label_noise.prob):
            self.noise_label(view)
        if self.chance(cfg.glare.prob):
            self.add_glare(view)
        if self.chance(cfg.exposure.prob):
            self.expose(view)
        image = self.photometric(image=view.image)["image"]
        if self.chance(cfg.degrade.downscale_prob):
            image = self.shrink(image)
        image = self.degrade(image=image)["image"]
        if self.chance(cfg.view.keep_background_prob):
            # за краем исходного кадра настоящего фона нет: там растянутые крайние пиксели, в бою их закрыла бы заливка
            image[view.inside == 0] = self.render_cfg["background"]["color"]
        else:
            if self.chance(cfg.mask_defects.prob):
                self.spoil_mask(view)
            image = mute_background(
                image,
                view.bottle,
                self.render_cfg,
                float(self.rng.uniform(*cfg.view.mask_jitter)),
            )
        image = resize_to_fit(
            image,
            self.input_size,
            INTERPOLATIONS[str(self.rng.choice(cfg.resize_interpolations))],
        )
        offset = (
            (float(self.rng.random()), float(self.rng.random()))
            if cfg.random_pad_position
            else (0.5, 0.5)
        )
        return image, offset
