#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["ultralytics", "pillow", "numpy", "scikit-learn", "timm", "clip @ git+https://github.com/ultralytics/CLIP.git"]
# ///
"""
Калибровка отбора целевых бутылок по разметке из labels.json.

  calibrate.py [--images images] [--labels labels.json] [-c normalize.toml] [--beta 1.5]

Сегментация SAM3 запускается с настройками из секции [segmentation] конфига нормализации,
поэтому калибровка и нормализация видят одних и тех же кандидатов.
Кандидат положительный, если его бокс совпал с целевой бутылкой, которая не обрезана.
Обрезанные цели по ТЗ нормализовать нельзя, поэтому для отбора они отрицательные.
Признаки пишутся в features.jsonl, итоговые веса и порог в calibration.json.
"""
import argparse, json, re, tomllib
from pathlib import Path

import numpy as np

from features import FEAT_SIDE, FEATURES, candidate_features, gray_small

HERE = Path(__file__).resolve().parent


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def image_rows(name, label, seg, gray):
    feats = candidate_features(gray, seg)
    good = [t["box"] for t in label["targets"] if not t["truncated"]]
    bad = [t["box"] for t in label["targets"] if t["truncated"]]
    rows = []
    for c, f in zip(seg["cands"], feats):
        best_good = max((iou(c["box"], b) for b in good), default=0)
        best_bad = max((iou(c["box"], b) for b in bad), default=0)
        rows.append({"name": name, "cand": c["id"], "box": c["box"], "y": int(best_good >= 0.7),
                     "truncated_target": int(best_bad >= 0.7), **f})
    # цели, которых нет среди кандидатов (SAM3 пропустил): ловить их нечем, но в метриках они считаются промахом
    missed = sum(1 for b in good if max((iou(c["box"], b) for c in seg["cands"]), default=0) < 0.7)
    return name, rows, len(good), missed, len(seg["cands"])


def evaluate(names, rows, probs, thr, n_good, n_missed, subset=None):
    """Метрики уровня бутылок и картинок для правила «берём кандидатов с p >= thr»."""
    by_img = {}
    for r, p in zip(rows, probs):
        by_img.setdefault(r["name"], []).append((r["y"], p >= thr))
    tp = fp = fn = exact = no_target_imgs = no_target_fp = imgs = 0
    for n in names:
        if subset and not subset(n):
            continue
        imgs += 1
        items = by_img.get(n, [])
        t = sum(y and s for y, s in items); f = sum(s and not y for y, s in items)
        miss = sum(y and not s for y, s in items) + n_missed[n]
        tp += t; fp += f; fn += miss
        exact += (f == 0 and miss == 0)
        if n_good[n] == 0:
            no_target_imgs += 1; no_target_fp += f > 0
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    tp, fp, fn = int(tp), int(fp), int(fn)
    return {"images": imgs, "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec + 1e-9),
            "exact_images": exact / max(imgs, 1), "fp_on_empty": no_target_fp / max(no_target_imgs, 1),
            "tp": tp, "fp": fp, "fn": fn}


def fbeta(m, beta):
    p, r = m["precision"], m["recall"]
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-9)


def best_threshold(names, rows, probs, n_good, n_missed, beta=1.0):
    """Порог, максимизирующий F-бета: beta > 1 ценит полноту выше точности."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [fbeta(evaluate(names, rows, probs, t, n_good, n_missed), beta) for t in grid]
    return float(grid[int(np.argmax(scores))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=HERE / "images")
    ap.add_argument("--labels", type=Path, default=HERE / "labels.json")
    ap.add_argument("-c", "--config", type=Path, default=HERE / "normalize.toml")
    ap.add_argument("--beta", type=float, default=1.5, help="вес полноты при выборе порога: 1 = F1, 2 = полнота вдвое важнее")
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    from seg import Segmenter, open_image

    labels = {n: v for n, v in json.loads(args.labels.read_text()).items() if v["decision"] != "ambiguous"}
    todo = [(n, v) for n, v in labels.items() if (args.images / n).is_file()]
    print(f"размеченных картинок: {len(labels)}, найдено в {args.images}: {len(todo)}")
    seg_cfg = tomllib.loads(args.config.read_text())["segmentation"]
    segmenter = Segmenter(args.config.resolve().parent / seg_cfg["model"], seg_cfg["prompt"], seg_cfg["conf"],
                          seg_cfg["imgsz"], seg_cfg["max_side"])
    results = []
    for i, (n, v) in enumerate(todo, 1):
        img = open_image(args.images / n)
        results.append(image_rows(n, v, segmenter(img), gray_small(img)))
        if i % 50 == 0 or i == len(todo):
            print(f"  сегментация и признаки: {i}/{len(todo)}", flush=True)
    names = [r[0] for r in results]
    rows = [row for r in results for row in r[1]]
    n_good = {r[0]: r[2] for r in results}; n_missed = {r[0]: r[3] for r in results}; n_cands = {r[0]: r[4] for r in results}
    with open(HERE / "features.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    X = np.array([[r[k] for k in FEATURES] for r in rows]); y = np.array([r["y"] for r in rows])
    # серии кадров с одним исходным именем не должны попадать и в обучение, и в проверку
    groups = [re.sub(r"_[0-9a-f]{10}$", "", r["name"].rsplit(".", 1)[0]) for r in rows]
    print(f"кандидатов: {len(rows)}, положительных: {y.sum()}, целей без кандидата (промах SAM3): {sum(n_missed.values())}, "
          f"обрезанных среди кандидатов: {sum(r['truncated_target'] for r in rows)}")

    oof = {"lr": np.zeros(len(y)), "gb": np.zeros(len(y))}
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        oof["lr"][te] = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(X[tr]), y[tr]).predict_proba(scaler.transform(X[te]))[:, 1]
        oof["gb"][te] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]

    conf = X[:, FEATURES.index("conf")]
    edge = X[:, FEATURES.index("edge_touch")]
    methods = {
        "порог по conf SAM3": conf,
        "conf + отказ при касании края": conf * (1 - edge),
        "логистическая регрессия": oof["lr"],
        "градиентный бустинг (справочно)": oof["gb"],
    }
    hard = lambda n: n_cands[n] >= 2
    report = {}
    print(f"\n{'метод':34s} {'порог':>5s} | {'все: P':>6s} {'R':>5s} {'F1':>5s} {'точн.карт':>9s} {'FP пуст':>7s} | {'трудн: F1':>9s} {'точн.карт':>9s}")
    for m, probs in methods.items():
        thr = best_threshold(names, rows, probs, n_good, n_missed, args.beta)
        a = evaluate(names, rows, probs, thr, n_good, n_missed)
        hd = evaluate(names, rows, probs, thr, n_good, n_missed, subset=hard)
        report[m] = {"threshold": thr, "all": a, "hard": hd}
        print(f"{m:34s} {thr:5.2f} | {a['precision']:6.3f} {a['recall']:5.3f} {a['f1']:5.3f} {a['exact_images']:9.3f} {a['fp_on_empty']:7.3f} |"
              f" {hd['f1']:9.3f} {hd['exact_images']:9.3f}   tp={a['tp']} fp={a['fp']} fn={a['fn']}")

    print("\nлогистическая регрессия, порог под разные beta:")
    for b in (1.0, 1.25, 1.5, 2.0):
        t = best_threshold(names, rows, oof["lr"], n_good, n_missed, b)
        m = evaluate(names, rows, oof["lr"], t, n_good, n_missed)
        print(f"  beta={b:<4} порог={t:.2f}  P={m['precision']:.3f} R={m['recall']:.3f}  "
              f"трудных верно={evaluate(names, rows, oof['lr'], t, n_good, n_missed, subset=hard)['exact_images']:.3f}")

    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(X), y)
    thr = report["логистическая регрессия"]["threshold"]
    print("\nвеса признаков (на стандартизованных значениях):")
    for k, wgt in sorted(zip(FEATURES, lr.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {k:16s} {wgt:+.2f}")
    calib = {"features": FEATURES, "mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist(), "coef": lr.coef_[0].tolist(),
             "intercept": float(lr.intercept_[0]), "threshold": thr, "beta": args.beta, "feat_side": FEAT_SIDE,
             "segmentation": segmenter.tag, "n_images": len(names), "n_candidates": len(rows), "cv_report": report}
    (HERE / "calibration.json").write_text(json.dumps(calib, ensure_ascii=False, indent=1))
    print("\nсохранено: calibration.json, features.jsonl")


if __name__ == "__main__":
    main()
