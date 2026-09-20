import albumentations as A
import cv2
import numpy as np

from vis_seacher_training.configs import AugmentationsConfig
from vis_seacher_training.data.views import View
from vis_seacher_training.data.views import ViewParams
from vis_seacher_training.data.views import mute_background
from vis_seacher_training.data.views import render_view

INTERPOLATIONS = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA, "lanczos": cv2.INTER_LANCZOS4}
FULL_BOTTLE_PAD = 100.0  # запас заведомо больше бутылки: окно упирается в её маску
OCCLUDER_ATTEMPTS = 6
DOWNSCALE_PAIRS = [(cv2.INTER_NEAREST, cv2.INTER_NEAREST), (cv2.INTER_AREA, cv2.INTER_LINEAR), (cv2.INTER_LINEAR, cv2.INTER_CUBIC), (cv2.INTER_AREA, cv2.INTER_NEAREST)]


def fit_size(height: int, width: int, target: tuple[int, int]) -> tuple[int, int]:
    """Размер картинки, вписанной в target с сохранением пропорций: высота и ширина."""
    scale = min(target[0] / height, target[1] / width)
    return min(target[0], max(1, round(height * scale))), min(target[1], max(1, round(width * scale)))


def resize_to_fit(image: np.ndarray, target: tuple[int, int], interpolation: int | None = None) -> np.ndarray:
    """Вписывает картинку в target, пропорции сохраняются. Без interpolation — как в пайплайне: area при уменьшении, cubic при увеличении."""
    height, width = fit_size(image.shape[0], image.shape[1], target)
    if interpolation is None:
        interpolation = cv2.INTER_AREA if height < image.shape[0] else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interpolation)


class TrainAugmenter:
    """Аугментации обучения: фото и разметка бутылки → кроп, вписанный в размер входа.

    Порядок повторяет жизнь кадра. Сначала то, что случается при съёмке: как нормализатор вырезал бутылку (запас вокруг этикетки,
    шум бокса, ошибка выравнивания), перспектива, перекрытия, порча этикетки, блик, свет и шум, потеря качества. Затем то, что пайплайн
    делает с уже снятым кадром: заглушение фона по маске, у части примеров пропускается, и ресайз случайной интерполяцией.
    Фон заливается после цветовых аугментаций, иначе он перестал бы быть нулём на входе модели.
    """

    def __init__(self, cfg: AugmentationsConfig, render_cfg: dict, input_size: tuple[int, int], seed: int = 0) -> None:
        self.cfg, self.render_cfg, self.input_size = cfg, render_cfg, input_size
        self.default_pad = float(render_cfg["crop"]["padding_y"])
        fill = tuple(render_cfg["background"]["color"])
        ph, dg = cfg.photometric, cfg.degrade
        self.perspective = A.Compose([A.Perspective(scale=cfg.perspective.scale, keep_size=True, border_mode=cv2.BORDER_CONSTANT, fill=fill, fill_mask=0, p=cfg.perspective.prob)])
        self.label_noise = A.Compose(
            [
                A.OneOf(
                    [
                        A.GaussianBlur(sigma_limit=(1.0, 4.0), p=1),
                        A.MotionBlur(blur_limit=(5, 15), p=1),
                        A.Defocus(radius=(2, 5), p=1),
                        A.GaussNoise(std_range=(0.03, 0.09), p=1),
                        A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.1, 0.45), p=1),
                        A.ImageCompression(quality_range=(10, 40), p=1),
                    ],
                    p=1,
                )
            ]
        )
        self.photometric = A.Compose(
            [
                # тон почти не трогаем: красная и белая этикетки одной серии — разные вина
                A.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=(-0.03, 0.03), p=ph.color_prob),
                A.RandomGamma(gamma_limit=(70, 140), p=ph.color_prob * 0.4),
                # диапазон уже стандартного 3000–15000 К: на его краях оранжевая этикетка уходила в зелёную
                A.PlanckianJitter(mode="blackbody", temperature_limit=(4500, 8500), p=ph.white_balance_prob),
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), shadow_intensity_range=(0.3, 0.6), p=ph.shadow_prob),
                A.OneOf([A.GaussianBlur(sigma_limit=(0.5, 2.5), p=1), A.MotionBlur(blur_limit=(3, 11), p=1), A.Defocus(radius=(1, 3), p=1)], p=ph.blur_prob),
                A.OneOf([A.GaussNoise(std_range=(0.02, 0.1), p=1), A.ISONoise(p=1)], p=ph.noise_prob),
                A.Sharpen(p=ph.sharpen_prob),
            ]
        )
        self.degrade = A.Compose(
            [
                A.OneOf([A.Downscale(scale_range=dg.downscale_range, interpolation_pair={"downscale": down, "upscale": up}, p=1) for down, up in DOWNSCALE_PAIRS], p=dg.downscale_prob),
                A.ImageCompression(quality_range=dg.jpeg_quality, p=dg.jpeg_prob),
            ]
        )
        self.rng = np.random.default_rng(seed)
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        """Свой поток случайности. Зовётся в каждом воркере даталоадера: A.Compose держит собственный генератор, и после fork все воркеры выдавали бы одни и те же аугментации."""
        self.rng = np.random.default_rng(seed)
        for n, pipeline in enumerate((self.perspective, self.label_noise, self.photometric, self.degrade)):
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
        jitter = self.rng.uniform(-view.edge_jitter, view.edge_jitter, size=4) if self.chance(view.edge_jitter_prob) else np.zeros(4)
        angle = self.rng.uniform(-view.rotate_deg, view.rotate_deg) if self.chance(view.rotate_prob) else 0.0
        return ViewParams(float(pad_top), float(pad_bottom), (float(jitter[0]), float(jitter[1]), float(jitter[2]), float(jitter[3])), float(angle))

    def occluder_texture(self, image: np.ndarray, height: int, width: int) -> np.ndarray:
        """Чем закрашено перекрытие: сплошной цвет, шум либо кусок этого же кадра — он похож на соседний предмет."""
        kind = self.rng.integers(3)
        if kind == 0:
            return np.full((height, width, 3), self.rng.integers(0, 256, size=3), np.uint8)
        if kind == 1:
            base = self.rng.integers(0, 256, size=3)
            return np.clip(base + self.rng.normal(0, 35, size=(height, width, 3)), 0, 255).astype(np.uint8)
        h, w = image.shape[:2]
        ph, pw = max(2, min(h, height)), max(2, min(w, width))
        y, x = self.rng.integers(0, h - ph + 1), self.rng.integers(0, w - pw + 1)
        return cv2.resize(image[y : y + ph, x : x + pw], (width, height), interpolation=cv2.INTER_LINEAR)

    def occlude(self, view: View) -> None:
        """Перекрытия поверх бутылки. Перекрытие не ставится, если после него видимая доля этикетки или бутылки упала бы ниже порога.

        Часть перекрытий вырезается и из масок: сегментация такой предмет от бутылки отделяет, и после заглушения на его месте фон.
        """
        cfg = self.cfg.occlusion
        H, W = view.image.shape[:2]
        label_area, bottle_area = max(1, int((view.label > 0).sum())), max(1, int((view.bottle > 0).sum()))
        ys, xs = np.nonzero(view.bottle)
        if not len(ys):
            return
        covered = np.zeros((H, W), bool)
        for _ in range(int(self.rng.integers(1, cfg.max_occluders + 1))):
            for _ in range(OCCLUDER_ATTEMPTS):
                h, w = (max(2, int(s * side)) for s, side in zip(self.rng.uniform(*cfg.size_range, size=2), (H, W), strict=True))
                at = self.rng.integers(len(ys))
                shape = np.zeros((H, W), np.uint8)
                if self.chance(0.5):
                    cv2.ellipse(shape, (int(xs[at]), int(ys[at])), (max(1, w // 2), max(1, h // 2)), float(self.rng.uniform(0, 180)), 0, 360, 255, -1)
                else:
                    box = cv2.boxPoints(((float(xs[at]), float(ys[at])), (float(w), float(h)), float(self.rng.uniform(-30, 30))))
                    cv2.fillPoly(shape, [np.round(box).astype(np.int32)], 255)
                trial = covered | (shape > 0)
                if ((view.label > 0) & ~trial).sum() / label_area < cfg.min_label_visible or ((view.bottle > 0) & ~trial).sum() / bottle_area < cfg.min_bottle_visible:
                    continue
                covered = trial
                y1, y2, x1, x2 = *np.nonzero(shape.any(axis=1))[0][[0, -1]], *np.nonzero(shape.any(axis=0))[0][[0, -1]]
                texture = self.occluder_texture(view.image, int(y2 - y1 + 1), int(x2 - x1 + 1))
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
        alpha = cv2.GaussianBlur(view.label.astype(np.float32) / 255, (0, 0), 2.0)[..., None]
        view.image = (alpha * spoiled + (1 - alpha) * view.image).round().astype(np.uint8)

    def add_glare(self, view: View) -> None:
        """Блики: вытянутые вдоль бутылки светлые пятна с центром на ней, смешение «экран»."""
        cfg = self.cfg.glare
        ys, xs = np.nonzero(view.bottle)
        if not len(ys):
            return
        H, W = view.image.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        image = view.image.astype(np.float32)
        for _ in range(int(self.rng.integers(1, cfg.max_blobs + 1))):
            at = self.rng.integers(len(ys))
            sx = max(1.0, float(self.rng.uniform(*cfg.size_range)) * W * 0.5)
            sy = sx * float(self.rng.uniform(1.0, 4.0))
            blob = np.exp(-(((xx - xs[at]) / sx) ** 2 + ((yy - ys[at]) / sy) ** 2) / 2) * float(self.rng.uniform(*cfg.intensity))
            image += blob[..., None] * (255 - image)
        view.image = image.round().astype(np.uint8)

    def __call__(self, rgb: np.ndarray, cand: dict) -> tuple[np.ndarray, tuple[float, float]]:
        """Кроп uint8, вписанный в размер входа, и его положение внутри полей: доли свободного места сверху и слева."""
        cfg = self.cfg
        view = render_view(rgb, cand, self.render_cfg, self.sample_view())
        warped = self.perspective(image=view.image, masks=[view.bottle, view.label, view.inside])
        view = View(warped["image"], *(np.ascontiguousarray(mask) for mask in warped["masks"]))
        if self.chance(cfg.occlusion.prob):
            self.occlude(view)
        if self.chance(cfg.label_noise.prob):
            self.noise_label(view)
        if self.chance(cfg.glare.prob):
            self.add_glare(view)
        image = self.degrade(image=self.photometric(image=view.image)["image"])["image"]
        if self.chance(cfg.view.keep_background_prob):
            # за краем исходного кадра настоящего фона нет: там растянутые крайние пиксели, в бою их закрыла бы заливка
            image[view.inside == 0] = self.render_cfg["background"]["color"]
        else:
            image = mute_background(image, view.bottle, self.render_cfg, float(self.rng.uniform(*cfg.view.mask_jitter)))
        image = resize_to_fit(image, self.input_size, INTERPOLATIONS[str(self.rng.choice(cfg.resize_interpolations))])
        offset = (float(self.rng.random()), float(self.rng.random())) if cfg.random_pad_position else (0.5, 0.5)
        return image, offset
