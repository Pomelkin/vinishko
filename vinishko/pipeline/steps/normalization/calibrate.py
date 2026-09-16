import json
import re
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import rich_click as click
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from vinishko.pipeline.steps.normalization.features import FEAT_SIDE, FEATURES, candidate_features, gray_small
from vinishko.pipeline.steps.normalization.seg import Segmenter, open_image

HERE = Path(__file__).resolve().parent
IOU_MATCH = 0.7  # кандидат считается попаданием в цель от этого IoU

Box = list[int]
Row = dict[str, Any]
ImageStats = dict[str, int]


def iou(a: Box, b: Box) -> float:
    """IoU двух боксов [x1, y1, x2, y2]."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def image_rows(name: str, label: dict, seg: dict, gray: np.ndarray) -> tuple[str, list[Row], int, int, int]:
    """Строки признаков кандидатов одной картинки с целевой меткой: (имя, строки, целей, промахов SAM3, кандидатов).

    Кандидат положительный, если совпал по IoU с целой целевой бутылкой; обрезанные цели по ТЗ нормализовать нельзя,
    поэтому совпадение с ними отрицательное и помечается truncated_target.
    """
    feats = candidate_features(gray, seg)
    good = [t["box"] for t in label["targets"] if not t["truncated"]]
    bad = [t["box"] for t in label["targets"] if t["truncated"]]
    rows: list[Row] = []
    for c, f in zip(seg["cands"], feats, strict=True):
        best_good = max((iou(c["box"], b) for b in good), default=0)
        best_bad = max((iou(c["box"], b) for b in bad), default=0)
        rows.append(
            {
                "name": name,
                "cand": c["id"],
                "box": c["box"],
                "y": int(best_good >= IOU_MATCH),
                "truncated_target": int(best_bad >= IOU_MATCH),
                **f,
            }
        )
    # цели, которых нет среди кандидатов (SAM3 пропустил): ловить их нечем, но в метриках они считаются промахом
    missed = sum(1 for b in good if max((iou(c["box"], b) for c in seg["cands"]), default=0) < IOU_MATCH)
    return name, rows, len(good), missed, len(seg["cands"])


def evaluate(
    names: list[str],
    rows: list[Row],
    probs: np.ndarray,
    thr: float,
    n_good: ImageStats,
    n_missed: ImageStats,
    subset: Callable[[str], bool] | None = None,
) -> dict[str, float]:
    """Метрики уровня бутылок и картинок для правила «берём кандидатов с p >= thr»."""
    by_img: dict[str, list[tuple[int, bool]]] = {}
    for r, p in zip(rows, probs, strict=True):
        by_img.setdefault(str(r["name"]), []).append((int(r["y"]), bool(p >= thr)))
    tp = fp = fn = exact = no_target_imgs = no_target_fp = imgs = 0
    for n in names:
        if subset and not subset(n):
            continue
        imgs += 1
        items = by_img.get(n, [])
        t = sum(y and s for y, s in items)
        f = sum(s and not y for y, s in items)
        miss = sum(y and not s for y, s in items) + n_missed[n]
        tp += t
        fp += f
        fn += miss
        exact += f == 0 and miss == 0
        if n_good[n] == 0:
            no_target_imgs += 1
            no_target_fp += f > 0
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    return {
        "images": imgs,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec + 1e-9),
        "exact_images": exact / max(imgs, 1),
        "fp_on_empty": no_target_fp / max(no_target_imgs, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def fbeta(m: dict[str, float], beta: float) -> float:
    """F-бета по точности и полноте из evaluate."""
    p, r = m["precision"], m["recall"]
    return (1 + beta**2) * p * r / (beta**2 * p + r + 1e-9)


def best_threshold(
    names: list[str], rows: list[Row], probs: np.ndarray, n_good: ImageStats, n_missed: ImageStats, beta: float = 1.0
) -> float:
    """Порог, максимизирующий F-бета: beta > 1 ценит полноту выше точности."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [fbeta(evaluate(names, rows, probs, t, n_good, n_missed), beta) for t in grid]
    return float(grid[int(np.argmax(scores))])


@click.command()
@click.option("--images", type=click.Path(file_okay=False, path_type=Path), default=HERE / "images", show_default=True)
@click.option("--labels", type=click.Path(dir_okay=False, path_type=Path), default=HERE / "labels.json", show_default=True)
@click.option("-c", "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=HERE / "normalize.toml", show_default=True)
@click.option("--beta", type=float, default=1.5, show_default=True, help="Вес полноты при выборе порога: 1 = F1, 2 = полнота вдвое важнее")
def main(images: Path, labels: Path, config: Path, beta: float) -> None:
    """Калибровка отбора целевых бутылок по разметке из labels.json.

    Сегментация запускается с настройками секции [segmentation] конфига нормализации, поэтому калибровка и нормализация
    видят одних и тех же кандидатов. Признаки пишутся в features.jsonl, веса логрега и порог — в calibration.json.
    """
    marked = {n: v for n, v in json.loads(labels.read_text()).items() if v["decision"] != "ambiguous"}
    todo = [(n, v) for n, v in marked.items() if (images / n).is_file()]
    click.echo(f"размеченных картинок: {len(marked)}, найдено в {images}: {len(todo)}")
    sc = tomllib.loads(config.read_text())["segmentation"]
    model = Path(sc["model"]) if Path(sc["model"]).is_absolute() else config.resolve().parent / sc["model"]
    segmenter = Segmenter(model, sc["prompt"], sc["conf"], sc["imgsz"], sc["max_side"])
    results = []
    for i, (n, v) in enumerate(todo, 1):
        img = open_image(images / n)
        results.append(image_rows(n, v, segmenter(img), gray_small(img)))
        if i % 50 == 0 or i == len(todo):
            click.echo(f"  сегментация и признаки: {i}/{len(todo)}")
    names = [r[0] for r in results]
    rows = [row for r in results for row in r[1]]
    n_good = {r[0]: r[2] for r in results}
    n_missed = {r[0]: r[3] for r in results}
    n_cands = {r[0]: r[4] for r in results}
    with (HERE / "features.jsonl").open("w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    X = np.array([[r[k] for k in FEATURES] for r in rows])
    y = np.array([r["y"] for r in rows])
    # серии кадров с одним исходным именем не должны попадать и в обучение, и в проверку
    groups = [re.sub(r"_[0-9a-f]{10}$", "", str(r["name"]).rsplit(".", 1)[0]) for r in rows]
    click.echo(
        f"кандидатов: {len(rows)}, положительных: {y.sum()}, целей без кандидата (промах SAM3): {sum(n_missed.values())}, "
        f"обрезанных среди кандидатов: {sum(int(r['truncated_target']) for r in rows)}"
    )

    oof = {"lr": np.zeros(len(y)), "gb": np.zeros(len(y))}
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        lr_fold = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(X[tr]), y[tr])
        oof["lr"][te] = lr_fold.predict_proba(scaler.transform(X[te]))[:, 1]
        gb_fold = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05).fit(X[tr], y[tr])
        oof["gb"][te] = gb_fold.predict_proba(X[te])[:, 1]

    conf = X[:, FEATURES.index("conf")]
    edge = X[:, FEATURES.index("edge_touch")]
    methods = {
        "порог по conf SAM3": conf,
        "conf + отказ при касании края": conf * (1 - edge),
        "логистическая регрессия": oof["lr"],
        "градиентный бустинг (справочно)": oof["gb"],
    }

    def hard(n: str) -> bool:
        return n_cands[n] >= 2

    report = {}
    click.echo(
        f"\n{'метод':34s} {'порог':>5s} | {'все: P':>6s} {'R':>5s} {'F1':>5s} {'точн.карт':>9s} {'FP пуст':>7s} | {'трудн: F1':>9s} {'точн.карт':>9s}"
    )
    for m, probs in methods.items():
        thr = best_threshold(names, rows, probs, n_good, n_missed, beta)
        a = evaluate(names, rows, probs, thr, n_good, n_missed)
        hd = evaluate(names, rows, probs, thr, n_good, n_missed, subset=hard)
        report[m] = {"threshold": thr, "all": a, "hard": hd}
        click.echo(
            f"{m:34s} {thr:5.2f} | {a['precision']:6.3f} {a['recall']:5.3f} {a['f1']:5.3f} {a['exact_images']:9.3f} {a['fp_on_empty']:7.3f} |"
            f" {hd['f1']:9.3f} {hd['exact_images']:9.3f}   tp={a['tp']} fp={a['fp']} fn={a['fn']}"
        )

    click.echo("\nлогистическая регрессия, порог под разные beta:")
    for b in (1.0, 1.25, 1.5, 2.0):
        t = best_threshold(names, rows, oof["lr"], n_good, n_missed, b)
        m = evaluate(names, rows, oof["lr"], t, n_good, n_missed)
        hd = evaluate(names, rows, oof["lr"], t, n_good, n_missed, subset=hard)
        click.echo(f"  beta={b:<4} порог={t:.2f}  P={m['precision']:.3f} R={m['recall']:.3f}  трудных верно={hd['exact_images']:.3f}")

    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(X), y)
    thr = report["логистическая регрессия"]["threshold"]
    click.echo("\nвеса признаков (на стандартизованных значениях):")
    for k, wgt in sorted(zip(FEATURES, lr.coef_[0], strict=True), key=lambda t: -abs(t[1])):
        click.echo(f"  {k:16s} {wgt:+.2f}")
    calib = {
        "features": FEATURES,
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "threshold": thr,
        "beta": beta,
        "feat_side": FEAT_SIDE,
        "segmentation": segmenter.tag,
        "n_images": len(names),
        "n_candidates": len(rows),
        "cv_report": report,
    }
    (HERE / "calibration.json").write_text(json.dumps(calib, ensure_ascii=False, indent=1))
    click.echo("\nсохранено: calibration.json, features.jsonl")


if __name__ == "__main__":
    main()
