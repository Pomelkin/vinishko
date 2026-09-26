import cv2
import numpy as np
from PIL import Image

FEAT_SIDE = 1024  # резкость и салиентность считаем на уменьшенной копии

BASE_FEATURES = ["conf", "conf_rel", "conf_rank", "log_area", "area_rel", "height_frac", "fill", "log_elong",
                 "edge_touch", "log_edge_margin", "center_dist", "log_sharp", "log_saliency", "log_n_cands",
                 "log_n_strong", "log_crowd"]
LABEL_FEATURES = ["label_conf", "label_cover", "label_fill", "log_label_sharp", "log_label_px", "log_bottle_px", "label_px_rel", "log_n_readable"]
FEATURES = BASE_FEATURES + LABEL_FEATURES


def spectral_saliency(gray: np.ndarray) -> np.ndarray:
    """Карта салиентности по спектральному остатку (Hou & Zhang 2007) на миниатюре 64x64, растянутая до размера входа."""
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    f = np.fft.fft2(small)
    log_amp = np.log(np.abs(f) + 1e-8)
    sal = np.abs(np.fft.ifft2(np.exp(log_amp - cv2.blur(log_amp, (3, 3)) + 1j * np.angle(f)))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (9, 9), 2.5)
    return cv2.resize(sal, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)


def label_short_side(polys_list: list[dict]) -> float:
    """Короткая сторона самой крупной этикетки корпуса в пикселях оригинала; 0, если этикеток нет."""
    best, best_area = 0.0, 0.0
    for lab in polys_list:
        pts = [np.asarray(p, np.float32) for p in lab["polys"] if len(p) >= 3]
        area = sum(cv2.contourArea(p) for p in pts)
        if pts and area > best_area:
            best_area, best = area, min(cv2.minAreaRect(np.concatenate(pts))[1])
    return float(best)


def candidate_features(gray_small: np.ndarray, seg: dict) -> list[dict]:
    """Признаки всех кандидатов одной картинки. gray_small: уменьшенная серая копия оригинала."""
    w, h, cands = seg["w"], seg["h"], seg["cands"]
    if not cands:
        return []
    s = gray_small.shape[1] / w
    lap2 = cv2.Laplacian(gray_small.astype(np.float32), cv2.CV_32F) ** 2
    lap_img = float(lap2.mean()) + 1e-6
    sal = spectral_saliency(gray_small)
    sal_img = float(sal.mean()) + 1e-9
    max_conf = max(c["conf"] for c in cands)
    max_area = max(c["area"] for c in cands)
    order = {c["id"]: r for r, c in enumerate(sorted(cands, key=lambda c: -c["conf"]))}
    n_strong = sum(c["conf"] >= 0.7 for c in cands)
    kernel = np.ones((3, 3), np.uint8)
    lab_px = [label_short_side(c.get("label_polys", [])) for c in cands]
    max_lab_px = max(lab_px) + 1e-6
    n_readable = sum(1 for c in cands if c.get("label", {}).get("cover", 0) >= 0.1)
    out = []
    for idx, c in enumerate(cands):
        x1, y1, x2, y2 = c["box"]
        mask = np.zeros(gray_small.shape, np.uint8)
        polys = [np.round(np.array(p) * s).astype(np.int32) for p in c["polys"] if len(p) >= 3]
        if polys:
            cv2.fillPoly(mask, polys, 1)
        else:
            mask[int(y1 * s):int(y2 * s) + 1, int(x1 * s):int(x2 * s) + 1] = 1
        inner = cv2.erode(mask, kernel, iterations=2)
        region = inner if inner.any() else mask
        pts = cv2.findNonZero(mask)
        rw, rh = cv2.minAreaRect(pts)[1] if pts is not None and len(pts) >= 3 else (x2 - x1, y2 - y1)
        margin = min(x1, y1, w - x2, h - y2) / max(w, h)
        crowd = sum(1 for o in cands if o is not c and o["conf"] >= 0.5 and 0.5 <= o["area"] / (c["area"] + 1e-9) <= 2)
        box_area = max(1, (x2 - x1) * (y2 - y1))
        out.append({
            "conf": c["conf"],
            "conf_rel": c["conf"] / max_conf,
            "conf_rank": order[c["id"]] / len(cands),
            "log_area": float(np.log(c["area"] + 1e-6)),
            "area_rel": c["area"] / (max_area + 1e-9),
            "height_frac": (y2 - y1) / h,
            "fill": min(1.0, c["area"] * w * h / box_area),
            "log_elong": float(np.log(max(rw, rh) / (min(rw, rh) + 1e-6) + 1e-6)),
            "edge_touch": float(margin < 0.005),
            "log_edge_margin": float(np.log(max(margin, 0) + 1e-3)),
            "center_dist": float(np.hypot((x1 + x2) / 2 / w - 0.5, (y1 + y2) / 2 / h - 0.5)),
            "log_sharp": float(np.log((lap2[region > 0].mean() + 1e-6) / lap_img)),
            "log_saliency": float(np.log((sal[mask > 0].mean() + 1e-9) / sal_img)),
            "log_n_cands": float(np.log(len(cands))),
            "log_n_strong": float(np.log1p(n_strong)),
            "log_crowd": float(np.log1p(crowd)),
            **label_feats(c, lab_px[idx], max_lab_px, n_readable, max(rw, rh) / s),
        })
    return out


def label_feats(c: dict, lab_px: float, max_lab_px: float, n_readable: int, bottle_px: float) -> dict:
    """Признаки этикетки кандидата по статистике SAM3: уверенность, покрытие, заполнение, резкость и размер в пикселях."""
    lb = c.get("label", {})
    return {
        "label_conf": lb.get("conf", 0.0),
        "label_cover": lb.get("cover", 0.0),
        "label_fill": lb.get("fill", 0.0),
        "log_label_sharp": float(np.log1p(lb.get("sharpness", 0.0))),
        "log_label_px": float(np.log1p(lab_px)),
        "log_bottle_px": float(np.log1p(bottle_px)),
        "label_px_rel": lab_px / max_lab_px,
        "log_n_readable": float(np.log1p(n_readable)),
    }


def gray_small(img: Image.Image, side: int = FEAT_SIDE) -> np.ndarray:
    """Серая уменьшенная копия RGB-картинки: одинаково для калибровки и нормализации."""
    g = img.convert("L")
    g.thumbnail((side, side), Image.Resampling.BILINEAR)
    return np.asarray(g)
