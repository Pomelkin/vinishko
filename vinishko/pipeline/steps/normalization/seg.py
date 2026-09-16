from collections.abc import Callable
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from PIL import Image, ImageOps
from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.sam import SAM3SemanticPredictor

BOTTLE = 39  # класс bottle в COCO
MIN_CONTOUR_AREA = 10  # контуры меньше этого в пикселях кадра модели — шум маски


def to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
    """Массив с GPU или уже numpy — в стабах ultralytics это объединение."""
    return x.cpu().numpy() if isinstance(x, Tensor) else x


def open_image(path: Path | str) -> Image.Image:
    """RGB-картинка с явно применённым EXIF-поворотом, иначе координаты разойдутся с тем, как её видит человек."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


class Segmenter:
    """Сегментация бутылок: SAM3 по текстовому промпту или YOLO-seg по классу bottle.

    Результат вызова: {"w", "h", "model", "prompt", "imgsz", "scale", "cands": [{"id", "conf", "box", "area", "polys"}]}.
    Боксы и полигоны — в пикселях оригинала после EXIF-поворота, кандидаты отсортированы по conf.
    """

    def __init__(
        self,
        model: Path | str,
        prompt: str = "wine bottle",
        conf: float | None = None,
        imgsz: int | None = None,
        max_side: int = 1600,
    ) -> None:
        """Путь к весам model: SAM3 узнаётся по имени файла и требует весов на диске, для YOLO промпт не используется."""
        self.model = str(model)
        self.is_sam3 = "sam3" in Path(model).name
        if self.is_sam3 and not Path(model).is_file():
            raise FileNotFoundError(f"нет весов {model}, как скачать: см. README.md")
        self.prompt = prompt if self.is_sam3 else "class bottle"
        self.imgsz = imgsz or (1008 if self.is_sam3 else 640)
        # SAM3 даёт бокалам 0.1–0.37, бутылкам 0.8+
        self.conf = conf if conf is not None else (0.3 if self.is_sam3 else 0.1)
        self.max_side = max_side  # маски 6000px на 20 кандидатов не влезут в VRAM
        self._run: Callable[[np.ndarray], Results] | None = None

    @property
    def tag(self) -> str:
        """Идентификатор настроек сегментации; калибровка записывает, под какие настройки обучена."""
        suffix = f"_{self.prompt.replace(' ', '-')}" if self.is_sam3 else ""
        return f"{Path(self.model).stem}_{self.imgsz}_c{self.conf}{suffix}"

    def _load(self) -> Callable[[np.ndarray], Results]:
        if self.is_sam3:
            predictor = SAM3SemanticPredictor(
                overrides={
                    "model": self.model,
                    "conf": self.conf,
                    "imgsz": self.imgsz,
                    "quantize": 16,
                    "task": "segment",
                    "mode": "predict",
                    "save": False,
                    "verbose": False,
                }
            )

            def run_sam3(bgr: np.ndarray) -> Results:
                predictor.set_image(bgr)
                return next(iter(predictor(text=[self.prompt])))

            return run_sam3

        yolo = YOLO(self.model)

        def run_yolo(bgr: np.ndarray) -> Results:
            results = yolo.predict(bgr, classes=[BOTTLE], conf=self.conf, imgsz=self.imgsz, retina_masks=True, verbose=False)
            return cast(Results, next(iter(results)))

        return run_yolo

    def __call__(self, img: Image.Image) -> dict:
        """Не потокобезопасно: вызывающий держит лок, если зовёт из нескольких потоков."""
        w, h = img.size
        k = min(1.0, self.max_side / max(w, h))
        small = img.resize((round(w * k), round(h * k)), Image.Resampling.BILINEAR) if k < 1 else img
        bgr = np.ascontiguousarray(np.asarray(small)[:, :, ::-1])
        if self._run is None:
            self._run = self._load()
        r = self._run(bgr)
        cands = []
        if r.masks is not None and r.boxes is not None:
            masks = to_numpy(r.masks.data) > 0.5  # маски уже в размере поданного кадра
            boxes = to_numpy(r.boxes.xyxy)
            confs = to_numpy(r.boxes.conf)
            for m, box, conf in zip(masks, boxes, confs, strict=True):
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polys = [
                    np.round(cv2.approxPolyDP(c, 1.0, True)[:, 0] / k).astype(int).tolist()
                    for c in contours
                    if cv2.contourArea(c) > MIN_CONTOUR_AREA
                ]
                cands.append(
                    {
                        "conf": round(float(conf), 3),
                        "box": [round(float(v) / k) for v in box],
                        "area": round(float(m.sum()) / m.size, 5),
                        "polys": polys,
                    }
                )
        cands.sort(key=lambda c: -c["conf"])
        for i, c in enumerate(cands, 1):
            c["id"] = i
        return {
            "w": w,
            "h": h,
            "model": Path(self.model).name,
            "prompt": self.prompt,
            "imgsz": self.imgsz,
            "scale": round(k, 4),
            "cands": cands,
        }
