import base64
import io
from dataclasses import dataclass
from html import escape

import numpy as np
from PIL import Image
from rich.progress import Progress

from scripts.bench_common.data import Split
from scripts.bench_common.data import load_view
from scripts.bench_common.metrics import FPRS
from scripts.bench_common.metrics import KS
from scripts.bench_common.metrics import SIGNALS
from scripts.bench_common.metrics import ModeResult
from scripts.bench_common.metrics import Retrieval
from vinishko.pred.pipeline.steps.normalization.seg import open_image


MODE_TITLES = {"raw": "без нормализации", "norm": "с нормализацией"}
PROTOCOL_TITLES = {"loo": "вся val в галерее", "oneshot": "одно фото класса в галерее"}
ROLE_TITLES = {
    "distractors": "вина не из галереи",
    "negatives": "не вино, Products-10K",
}
THUMB_HEIGHT = 220
THUMB_QUALITY = 82
HIST_BINS = 40


@dataclass(frozen=True)
class ReportSpec:
    """Чем отчёты разных моделей отличаются друг от друга."""

    model: str
    score_name: str
    """Как называется скор выдачи в таблицах: «косинус», «MaxSim на токен»."""
    score_short: str
    """Он же в подписях карточек примеров."""


MetricRow = tuple[str, list[str]]
"""Название метрики и её значения по режимам, уже строками."""


# ---------- таблица метрик: общая для консоли и html ----------


def column_title(result: ModeResult) -> str:
    """Заголовок столбца: режим входа и протокол."""
    return f"{MODE_TITLES[result.mode]}, {PROTOCOL_TITLES[result.protocol]}"


def percent(value: float) -> str:
    """Доля в процентах."""
    return f"{100 * value:.2f}%"


def dropped_cell(result: ModeResult, role: str) -> str:
    """Сколько картинок роли нормализация не допустила до энкодера."""
    if role not in result.dropped:
        return "—"
    dropped, total = result.dropped[role]
    return f"{dropped} из {total}, {percent(dropped / total)}"


def metric_cell(result: ModeResult, key: str) -> str:
    """Метрика долей; прочерк, если в этом режиме она не считалась: после нормализации не осталось запросов роли."""
    return percent(result.metrics[key]) if key in result.metrics else "—"


def rejection_rows(
    results: list[ModeResult], role: str, spec: ReportSpec
) -> list[MetricRow]:
    """Строки метрик отказа одной роли запросов без ответа."""
    rows: list[MetricRow] = [
        (
            "запросов",
            [
                str(len(r.rejects[role].rows)) if role in r.rejects else "0"
                for r in results
            ],
        ),
        ("отсеяно нормализацией", [dropped_cell(r, role) for r in results]),
    ]
    for signal, template in SIGNALS.items():
        title = template.format(score=spec.score_name)
        rows.append(
            (
                f"ROC AUC, {title}",
                [metric_cell(r, f"{role}/auc_{signal}") for r in results],
            )
        )
        rows.extend(
            (
                f"принято val при FPR {fpr:.0%}, {title}",
                [metric_cell(r, f"{role}/tpr_{signal}@{fpr}") for r in results],
            )
            for fpr in FPRS
        )
    return rows


def metric_sections(
    results: list[ModeResult], spec: ReportSpec
) -> list[tuple[str, list[MetricRow]]]:
    """Все метрики замера по разделам; столбцы значений идут в порядке results."""
    roles = [
        role
        for role in ROLE_TITLES
        if any(role in r.rejects or role in r.dropped for r in results)
    ]
    return [
        (
            "Данные",
            [
                ("галерея val, картинок", [str(r.gallery_size) for r in results]),
                ("запросов val", [str(len(r.val.rows)) for r in results]),
                (
                    "val отсеяно нормализацией",
                    [dropped_cell(r, "val") for r in results],
                ),
            ],
        ),
        (
            "Ретрив val",
            [
                *(
                    (f"recall@{k}", [metric_cell(r, f"recall@{k}") for r in results])
                    for k in KS
                ),
                *(
                    (
                        f"сквозной recall@{k}: отсеянный запрос — промах",
                        [metric_cell(r, f"e2e_recall@{k}") for r in results],
                    )
                    for k in KS
                ),
            ],
        ),
        *(
            (f"Отказ: {ROLE_TITLES[role]}", rejection_rows(results, role, spec))
            for role in roles
        ),
        (
            "Скорость эмбеддингов",
            [
                (
                    "картинок в секунду, с чтением и предобработкой",
                    [f"{r.speed:.1f}" for r in results],
                ),
                ("картинок", [str(r.images) for r in results]),
                ("минут", [f"{r.seconds / 60:.1f}" for r in results]),
            ],
        ),
    ]


# ---------- примеры ----------


@dataclass(frozen=True)
class ExampleSet:
    """Раздел примеров: какие запросы одной выдачи показывать."""

    title: str
    note: str
    retrieval: Retrieval
    positions: np.ndarray
    """Номера строк retrieval, не номера картинок."""


def pick_examples(
    result: ModeResult, count: int, rng: np.random.Generator
) -> list[ExampleSet]:
    """Случайные верные и ошибочные запросы val и самые уверенные ложные выдачи по запросам без ответа."""
    correct = np.flatnonzero(result.hits[:, 0])
    wrong = np.flatnonzero(~result.hits[:, 0])
    sets = [
        ExampleSet(
            "Верный top-1",
            "случайные запросы val",
            result.val,
            rng.choice(correct, min(count, len(correct)), replace=False),
        ),
        ExampleSet(
            "Ошибка в top-1",
            "случайные запросы val, где первым найден другой класс",
            result.val,
            rng.choice(wrong, min(count, len(wrong)), replace=False),
        ),
    ]
    sets.extend(
        ExampleSet(
            f"Запросы без ответа: {ROLE_TITLES[role]}",
            "с наибольшим скором top-1: их порог отказа отсеивает последними",
            rejects,
            np.argsort(-rejects.top1)[:count],
        )
        for role, rejects in result.rejects.items()
    )
    return sets


def thumb(img: Image.Image) -> str:
    """Картинка как data-URI: html должен остаться одним файлом."""
    img = img.convert("RGB")
    img.thumbnail((THUMB_HEIGHT * 3, THUMB_HEIGHT), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=THUMB_QUALITY)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def card(src: str, caption: str, badge: str = "", state: str = "") -> str:
    """Карточка с картинкой; state красит рамку, badge дублирует его словом."""
    mark = f'<span class="badge {state}">{escape(badge)}</span>' if badge else ""
    return f'<figure class="card {state}"><img loading="lazy" src="{src}" alt=""><figcaption>{mark}{caption}</figcaption></figure>'


def example_row(
    retrieval: Retrieval,
    position: int,
    gallery: Split,
    render_cfg: dict,
    spec: ReportSpec,
) -> str:
    """Запрос и его выдача. В режиме нормализации у запроса показаны и оригинал, и кроп, который видела модель."""
    split, row = retrieval.split, int(retrieval.rows[position])
    label = split.labels[row]
    cards = []
    if split.spans is not None:
        cards.append(
            card(
                thumb(open_image(split.path(row))),
                f"оригинал<br>{escape(split.names[row])}",
            )
        )
    cards.append(
        card(
            thumb(load_view(split, row, render_cfg)),
            f"запрос · класс {escape(label)}",
            "запрос",
            "query",
        )
    )
    for score, found in zip(
        retrieval.scores[position], retrieval.ids[position], strict=True
    ):
        hit = split.role == gallery.role and gallery.labels[found] == label
        caption = f"{escape(spec.score_short)} {score:.3f} · класс {escape(gallery.labels[found])}"
        cards.append(
            card(
                thumb(load_view(gallery, int(found), render_cfg)),
                caption,
                "верно" if hit else "мимо",
                "hit" if hit else "miss",
            )
        )
    return f'<div class="row">{"".join(cards)}</div>'


def examples_html(
    result: ModeResult,
    sets: list[ExampleSet],
    render_cfg: dict,
    spec: ReportSpec,
    progress: Progress,
) -> str:
    """Разделы примеров одного режима."""
    task = progress.add_task(
        result.key,
        name=f"примеры {result.key}",
        total=sum(len(s.positions) for s in sets),
    )
    parts = []
    for s in sets:
        rows = []
        for position in s.positions:
            rows.append(
                example_row(
                    s.retrieval, int(position), result.gallery, render_cfg, spec
                )
            )
            progress.update(task, advance=1)
        parts.append(
            f'<details open><summary>{escape(s.title)} <span class="muted">— {escape(s.note)}, {len(rows)} шт.</span></summary>{"".join(rows)}</details>'
        )
    return "".join(parts)


# ---------- гистограммы ----------


def score_groups(result: ModeResult) -> list[tuple[str, np.ndarray]]:
    """Скор top-1 по группам запросов: у хорошей модели верные val справа, всё остальное слева."""
    groups = [
        ("val, top-1 верный", result.val.top1[result.hits[:, 0]]),
        ("val, top-1 ошибочный", result.val.top1[~result.hits[:, 0]]),
    ]
    groups.extend(
        (ROLE_TITLES[role], rejects.top1) for role, rejects in result.rejects.items()
    )
    return [(name, values) for name, values in groups if len(values)]


def histogram_svg(groups: list[tuple[str, np.ndarray]], lo: float, hi: float) -> str:
    """Гистограммы групп одна под другой на общей оси скора; высота столбцов нормирована внутри группы."""
    width, row_h, plot_h, pad = 720, 96, 60, 8
    edges = np.linspace(lo, hi, HIST_BINS + 1)
    step = (width - 2 * pad) / HIST_BINS
    parts = []
    for n, (name, values) in enumerate(groups):
        top = n * row_h
        counts, _ = np.histogram(values, edges)
        title = f"{name} · {len(values)} запросов · медиана {np.median(values):.3f}"
        parts.append(
            f'<text class="label" x="{pad}" y="{top + 16}">{escape(title)}</text>'
        )
        base = top + 24 + plot_h
        parts.append(
            f'<line class="axis" x1="{pad}" x2="{width - pad}" y1="{base}" y2="{base}"/>'
        )
        for b, c in enumerate(counts):
            if not c:
                continue
            h = max(1.0, plot_h * c / counts.max())
            tip = f"{edges[b]:.3f}–{edges[b + 1]:.3f}: {c} запросов, {100 * c / len(values):.1f}%"
            parts.append(
                f'<rect class="s{n + 1}" x="{pad + b * step + 1:.1f}" y="{base - h:.1f}" width="{step - 2:.1f}" height="{h:.1f}" rx="2"><title>{escape(tip)}</title></rect>'
            )
    axis_y = len(groups) * row_h + 14
    ticks = np.linspace(lo, hi, 6)
    anchors = [
        "start",
        *["middle"] * (len(ticks) - 2),
        "end",
    ]  # крайние подписи иначе обрезаются краем svg
    for tick, anchor in zip(ticks, anchors, strict=True):
        x = pad + (tick - lo) / (hi - lo) * (width - 2 * pad)
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{axis_y}" text-anchor="{anchor}">{tick:.2f}</text>'
        )
    return f'<svg class="hist" viewBox="0 0 {width} {axis_y + 8}" role="img" aria-label="Распределение скора top-1 по группам запросов">{"".join(parts)}</svg>'


# ---------- страница ----------

CSS = """
:root{--bg:#f4f3f0;--surface:#fcfcfb;--text:#0b0b0b;--muted:#52514e;--line:#dddbd5;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--good:#0ca30c;--bad:#d03b3b}
@media (prefers-color-scheme:dark){:root{--bg:#111110;--surface:#1a1a19;--text:#fff;--muted:#c3c2b7;--line:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}}
*{box-sizing:border-box}
body{margin:0;padding:24px 16px 64px;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 12px}h3{font-size:16px;margin:24px 0 8px}
.muted{color:var(--muted);font-weight:400}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th{white-space:normal;vertical-align:bottom;min-width:120px}
th:first-child,td:first-child{text-align:left;white-space:normal}
tr.section td{font-weight:600;padding-top:16px;border-bottom:2px solid var(--line)}
dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;margin:0}dt{color:var(--muted)}dd{margin:0;overflow-wrap:anywhere}
nav a{color:var(--s1);margin-right:16px;display:inline-block}
details{margin:12px 0}summary{cursor:pointer;font-weight:600;padding:6px 0}
.row{display:flex;gap:8px;overflow-x:auto;padding:8px;margin:8px 0;background:var(--surface);border:1px solid var(--line);border-radius:10px}
.card{flex:0 0 156px;margin:0;padding:4px;border:3px solid transparent;border-radius:8px}
.card img{display:block;width:100%;height:190px;object-fit:contain;border-radius:4px;background:var(--bg)}
.card figcaption{font-size:12px;color:var(--muted);margin-top:4px;overflow-wrap:anywhere}
.card.query{border-color:var(--s1)}.card.hit{border-color:var(--good)}.card.miss{border-color:var(--bad)}
.badge{display:inline-block;margin-right:6px;padding:0 6px;border-radius:4px;font-weight:600;color:#fff;background:var(--muted)}
.badge.query{background:var(--s1)}.badge.hit{background:var(--good)}.badge.miss{background:var(--bad)}
.hist{width:100%;height:auto;display:block}
.hist .label{font-size:13px;fill:var(--text)}.hist .tick{font-size:11px;fill:var(--muted)}.hist .axis{stroke:var(--line)}
.hist .s1{fill:var(--s1)}.hist .s2{fill:var(--s2)}.hist .s3{fill:var(--s3)}.hist .s4{fill:var(--s4)}
"""


def metrics_table_html(results: list[ModeResult], spec: ReportSpec) -> str:
    """Таблица метрик: строка на метрику, столбец на режим."""
    head = "".join(f"<th>{escape(column_title(r))}</th>" for r in results)
    body = []
    for section, rows in metric_sections(results, spec):
        body.append(
            f'<tr class="section"><td colspan="{len(results) + 1}">{escape(section)}</td></tr>'
        )
        body.extend(
            f"<tr><td>{escape(name)}</td>{''.join(f'<td>{escape(v)}</td>' for v in values)}</tr>"
            for name, values in rows
        )
    return f'<div class="panel"><table><thead><tr><th>метрика</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def build_html(
    results: list[ModeResult],
    spec: ReportSpec,
    run_info: dict[str, str],
    render_cfg: dict,
    examples: int,
    seed: int,
    progress: Progress,
) -> str:
    """Отчёт одним самодостаточным файлом: параметры прогона, метрики, распределения скоров и примеры выдачи по каждому режиму."""
    all_scores = np.concatenate(
        [values for r in results for _, values in score_groups(r)]
    )
    lo, hi = (
        float(np.floor(all_scores.min() * 20) / 20),
        float(np.ceil(all_scores.max() * 20) / 20),
    )
    info = "".join(
        f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in run_info.items()
    )
    nav = "".join(f'<a href="#{r.key}">{escape(column_title(r))}</a>' for r in results)
    sections = []
    for r in results:
        sets = pick_examples(r, examples, np.random.default_rng(seed))
        sections.append(
            f'<h2 id="{r.key}">{escape(column_title(r)[0].upper() + column_title(r)[1:])}</h2>'
            f'<h3>{escape(spec.score_name[0].upper() + spec.score_name[1:])} top-1 по группам запросов <span class="muted">— общая ось, высота нормирована внутри группы; точные числа по наведению</span></h3>'
            f'<div class="panel">{histogram_svg(score_groups(r), lo, hi)}</div>'
            f"<h3>Примеры выдачи</h3>{examples_html(r, sets, render_cfg, spec, progress)}"
        )
    return (
        '<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Замер {escape(spec.model)}</title><style>{CSS}</style></head><body><main>"
        f'<h1>Замер {escape(spec.model)} на WineSensed</h1><p class="muted">{escape(run_info["модель"])} · {escape(run_info["конец прогона"])}</p>'
        f"<nav>{nav}</nav><h2>Метрики</h2>{metrics_table_html(results, spec)}"
        f'<h2>Параметры прогона</h2><div class="panel"><dl>{info}</dl></div>'
        f"{''.join(sections)}</main></body></html>"
    )
