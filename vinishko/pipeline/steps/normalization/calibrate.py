import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import rich_click as click
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from vinishko.pipeline.steps.normalization.features import (
    FEAT_SIDE,
    FEATURES,
    candidate_features,
    gray_small,
)
from vinishko.pipeline.steps.normalization.normalize import (
    Normalizer,
    bottle_axis,
    load_config,
)
from vinishko.pipeline.steps.normalization.seg import open_image
from vinishko.pipeline.structs import Rejection

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
    return inter / (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9
    )


def image_rows(
    name: str, label: dict, seg: dict, gray: np.ndarray
) -> tuple[str, list[Row], int, int, int]:
    """Строки признаков кандидатов одной картинки с целевой меткой: (имя, строки, целей, промахов SAM3, кандидатов).

    Кандидат положительный, если совпал по IoU с бутылкой, которую можно опознать по этикетке: целой или обрезанной
    краем кадра (truncated), с частично закрытой, но узнаваемой этикеткой. Бутылка с флагом label_hidden — этикетка
    скрыта или не читается из-за размытия — отрицательная, как и всё остальное на картинке.
    """
    feats = candidate_features(gray, seg)
    good = [t["box"] for t in label["targets"] if not t.get("label_hidden")]
    rows: list[Row] = []
    for c, f in zip(seg["cands"], feats, strict=True):
        best_good = max((iou(c["box"], b) for b in good), default=0)
        rows.append(
            {
                "name": name,
                "cand": c["id"],
                "box": c["box"],
                "y": int(best_good >= IOU_MATCH),
                **f,
            }
        )
    # цели, которых нет среди кандидатов (SAM3 пропустил): ловить их нечем, но в метриках они считаются промахом
    missed = sum(
        1
        for b in good
        if max((iou(c["box"], b) for c in seg["cands"]), default=0) < IOU_MATCH
    )
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


def label_verdicts(norm: Normalizer, label: dict, seg: dict) -> list[tuple[bool, str]]:
    """Проверка этикетки на каждой отмеченной бутылке картинки: (флаг label_hidden из разметки, вердикт проверки).

    Вердикт — "ok" или причина отказа NormalizationReason. Цели без кандидата SAM3 пропускаются.
    """
    out = []
    for t in label["targets"]:
        cand = max(seg["cands"], key=lambda c: iou(c["box"], t["box"]), default=None)
        if cand is None or iou(cand["box"], t["box"]) < IOU_MATCH:
            continue
        res = norm.check_label(
            cand, bottle_axis(cand["polys"], norm.cfg["orientation"]), 0.0
        )
        out.append(
            (
                bool(t.get("label_hidden")),
                res.reason.value if isinstance(res, Rejection) else "ok",
            )
        )
    return out


def label_report(verdicts: list[tuple[bool, str]]) -> None:
    """Отдельная оценка проверки этикетки по флагу label_hidden: сколько скрытых она ловит и сколько видимых бракует."""
    hidden = [v for h, v in verdicts if h]
    visible = [v for h, v in verdicts if not h]
    click.echo("\nпроверка этикетки против разметки (флаг «этикетка скрыта»):")
    caught = sum(v != "ok" for v in hidden)
    click.echo(
        f"  скрытых этикеток {len(hidden)}: проверка отказала {caught}, пропустила {len(hidden) - caught}"
    )
    kept = sum(v == "ok" for v in visible)
    click.echo(
        f"  видимых этикеток {len(visible)}: проверка пропустила {kept}, забраковала {len(visible) - kept}"
    )
    for name, group in (("отказы на скрытых", hidden), ("отказы на видимых", visible)):
        reasons = sorted(
            ((v, group.count(v)) for v in set(group) if v != "ok"), key=lambda t: -t[1]
        )
        if reasons:
            click.echo(f"  {name}: " + ", ".join(f"{v} {k}" for v, k in reasons))


def best_threshold(
    names: list[str],
    rows: list[Row],
    probs: np.ndarray,
    n_good: ImageStats,
    n_missed: ImageStats,
    beta: float = 1.0,
) -> float:
    """Порог, максимизирующий F-бета: beta > 1 ценит полноту выше точности."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [
        fbeta(evaluate(names, rows, probs, t, n_good, n_missed), beta) for t in grid
    ]
    return float(grid[int(np.argmax(scores))])


def collect(
    norm: Normalizer, cache: Path, pairs: list, folder: Path, max_cands: int = 0
) -> tuple[list, list, int]:
    """Признаки и метки всех картинок из pairs [(имя, метка)]: (результаты image_rows, вердикты этикетки, сколько тяжёлых пропущено)."""
    res, verd, skipped = [], [], 0
    for i, (n, v) in enumerate(pairs, 1):
        img = open_image(folder / n)
        f = cache / (n + ".json")
        if f.exists() and f.stat().st_mtime >= (folder / n).stat().st_mtime:
            seg = json.loads(f.read_text())
        else:
            seg = norm.seg(img)
            f.write_text(json.dumps(seg))
        row = image_rows(n, v, seg, gray_small(img))
        if max_cands and row[4] > max_cands:
            skipped += 1
        else:
            res.append(row)
            verd += label_verdicts(norm, v, seg)
        if i % 50 == 0 or i == len(pairs):
            click.echo(f"  сегментация и признаки: {i}/{len(pairs)}")
    return res, verd, skipped


def holdout_report(
    hres: list, model: LogisticRegression, scaler: StandardScaler, thr: float
) -> None:
    """Метрики обученной модели на отложенной выборке: сама выборка модель и порог не выбирала.

    Печатает точность и полноту при пороге thr, долю картинок с целевой бутылкой на первом месте и долю картинок,
    где хотя бы одна цель проходит порог.
    """
    hrows = [row for r in hres for row in r[1]]
    hp = model.predict_proba(
        scaler.transform(np.array([[r[k] for k in FEATURES] for r in hrows]))
    )[:, 1]
    hm = evaluate(
        [r[0] for r in hres],
        hrows,
        hp,
        thr,
        {r[0]: r[2] for r in hres},
        {r[0]: r[3] for r in hres},
    )
    by: dict[str, list[tuple[float, int]]] = {}
    for r, q in zip(hrows, hp, strict=True):
        by.setdefault(str(r["name"]), []).append((float(q), int(r["y"])))
    top1 = float(np.mean([max(v)[1] for v in by.values()]))
    anyhit = float(np.mean([any(y and q >= thr for q, y in v) for v in by.values()]))
    click.echo(
        f"\nотложенная выборка, {len(hres)} картинок, порог {thr:.2f}: P={hm['precision']:.3f} R={hm['recall']:.3f} F1={hm['f1']:.3f}"
        f" точно на картинке {hm['exact_images']:.3f}, лучший кандидат целевой {top1:.3f}, хотя бы одна цель прошла {anyhit:.3f}"
        f" tp={hm['tp']} fp={hm['fp']} fn={hm['fn']}"
    )


@click.command()
@click.option(
    "--images",
    type=click.Path(file_okay=False, path_type=Path),
    default=HERE / "images",
    show_default=True,
)
@click.option(
    "--labels",
    type=click.Path(dir_okay=False, path_type=Path),
    default=HERE / "labels.json",
    show_default=True,
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=HERE / "normalize.toml",
    show_default=True,
)
@click.option(
    "--beta",
    type=float,
    default=1.5,
    show_default=True,
    help="Вес полноты при выборе порога: 1 = F1, 2 = полнота вдвое важнее",
)
@click.option(
    "--holdout-images",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Папка картинок отложенной выборки: на ней ничего не подбирается, только проверка",
)
@click.option(
    "--holdout-labels",
    type=click.Path(dir_okay=False, path_type=Path),
    default=HERE / "labels_test.json",
    show_default=True,
)
@click.option(
    "--max-cands",
    type=int,
    default=0,
    show_default=True,
    help="Не брать в обучение картинки, где кандидатов SAM3 больше стольких; 0 = брать все",
)
@click.option(
    "--exclude",
    default=r"generated[_ ]image|nano_banana|edited_image|vibehype",
    show_default=True,
    help="Regex по имени файла: картинки не идут в обучение и метрики (нейросгенерированные сцены путают отбор)",
)
def main(
    images: Path,
    labels: Path,
    config: Path,
    beta: float,
    exclude: str,
    holdout_images: Path | None,
    holdout_labels: Path,
    max_cands: int,
) -> None:
    """Калибровка отбора целевых бутылок по разметке из labels.json.

    Сегментатор берётся из Normalizer с конфигом нормализации, поэтому калибровка и нормализация видят одних и тех же
    кандидатов. Признаки пишутся в features.jsonl, веса логрега и порог — в calibration.json. Отдельно, без подбора
    порогов, печатается отчёт проверки этикетки против флага label_hidden в разметке.
    """
    marked = {
        n: v
        for n, v in json.loads(labels.read_text()).items()
        if v["decision"] != "ambiguous"
    }
    excl = re.compile(exclude, re.IGNORECASE) if exclude else None
    synthetic = {
        n for n, v in marked.items() if v.get("synthetic")
    }  # ИИ-сцены, инфографика, баннеры: не то, что фотографирует пользователь
    todo = [
        (n, v)
        for n, v in marked.items()
        if (images / n).is_file()
        and n not in synthetic
        and not (excl and excl.search(n))
    ]
    by_regex = sum(1 for n in marked if excl and excl.search(n) and n not in synthetic)  # ty: ignore[redundant-condition]
    click.echo(
        f"размеченных картинок: {len(marked)}, найдено в {images}: {len(todo)}, исключено как синтетические: {len(synthetic)}"
        + (f", по --exclude: {by_regex}" if excl else "")
    )
    norm = Normalizer(load_config(config, []), config.resolve().parent)
    segmenter = norm.seg
    cache = (
        HERE / "cache" / f"calib_{segmenter.tag}"
    )  # сегментация с этикетками: признаки перебираются без повторного прогона SAM3
    cache.mkdir(parents=True, exist_ok=True)

    results, verdicts, skipped_hard = collect(norm, cache, todo, images, max_cands)
    if max_cands:
        click.echo(
            f"исключено сцен с более чем {max_cands} кандидатами: {skipped_hard}"
        )
    names = [r[0] for r in results]
    rows = [row for r in results for row in r[1]]
    n_good = {r[0]: r[2] for r in results}
    n_missed = {r[0]: r[3] for r in results}
    X = np.array([[r[k] for k in FEATURES] for r in rows])
    y = np.array([r["y"] for r in rows])
    # серии кадров с одним исходным именем не должны попадать и в обучение, и в проверку
    groups = [
        re.sub(r"_[0-9a-f]{10}$", "", str(r["name"]).rsplit(".", 1)[0]) for r in rows
    ]
    click.echo(
        f"кандидатов: {len(rows)}, положительных: {y.sum()}, целей без кандидата (промах SAM3): {sum(n_missed.values())}"
    )

    oof = {"lr": np.zeros(len(y)), "gb": np.zeros(len(y))}
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        lr_fold = LogisticRegression(C=1.0, max_iter=2000).fit(
            scaler.transform(X[tr]), y[tr]
        )
        oof["lr"][te] = lr_fold.predict_proba(scaler.transform(X[te]))[:, 1]
        gb_fold = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05).fit(
            X[tr], y[tr]
        )
        oof["gb"][te] = gb_fold.predict_proba(X[te])[:, 1]
    with (HERE / "features.jsonl").open(
        "w"
    ) as f:  # p — предсказание логрега на кандидате, обученного без его картинки
        f.writelines(
            json.dumps({**row, "p": round(float(p), 4)}) + "\n"
            for row, p in zip(rows, oof["lr"], strict=True)
        )

    conf = X[:, FEATURES.index("conf")]
    edge = X[:, FEATURES.index("edge_touch")]
    methods = {
        "порог по conf SAM3": conf,
        "conf + отказ при касании края": conf * (1 - edge),
        "логистическая регрессия": oof["lr"],
        "градиентный бустинг (справочно)": oof["gb"],
    }

    report = {}
    click.echo(
        f"\n{'метод':34s} {'порог':>5s} | {'P':>6s} {'R':>5s} {'F1':>5s} {'точн.карт':>9s} {'FP пуст':>7s}"
    )
    for m, probs in methods.items():
        thr = best_threshold(names, rows, probs, n_good, n_missed, beta)
        a = evaluate(names, rows, probs, thr, n_good, n_missed)
        report[m] = {"threshold": thr, "all": a}
        click.echo(
            f"{m:34s} {thr:5.2f} | {a['precision']:6.3f} {a['recall']:5.3f} {a['f1']:5.3f} {a['exact_images']:9.3f} {a['fp_on_empty']:7.3f}"
            f"   tp={a['tp']} fp={a['fp']} fn={a['fn']}"
        )

    click.echo("\nлогистическая регрессия, порог под разные beta:")
    for b in (1.0, 1.25, 1.5, 2.0):
        t = best_threshold(names, rows, oof["lr"], n_good, n_missed, b)
        m = evaluate(names, rows, oof["lr"], t, n_good, n_missed)
        click.echo(
            f"  beta={b:<4} порог={t:.2f}  P={m['precision']:.3f} R={m['recall']:.3f}  точн.карт={m['exact_images']:.3f}"
        )

    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(X), y)
    thr = report["логистическая регрессия"]["threshold"]
    click.echo("\nвеса признаков (на стандартизованных значениях):")
    for k, wgt in sorted(
        zip(FEATURES, lr.coef_[0], strict=True), key=lambda t: -abs(t[1])
    ):
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
    (HERE / "calibration.json").write_text(
        json.dumps(calib, ensure_ascii=False, indent=1)
    )
    click.echo("\nсохранено: calibration.json, features.jsonl")
    if (
        holdout_images and holdout_labels.exists()
    ):  # отложенная выборка: модель и порог уже выбраны без неё
        hl = [
            (n, v)
            for n, v in json.loads(holdout_labels.read_text()).items()
            if (holdout_images / n).is_file()
        ]
        holdout_report(collect(norm, cache, hl, holdout_images)[0], lr, scaler, thr)
    label_report(verdicts)


if __name__ == "__main__":
    main()
