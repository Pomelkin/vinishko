#!/usr/bin/env python3
"""Object-level audit of duplicate and near-duplicate wines.

The older ``near_duplicates.py`` measures similarity between rendered files.  This
audit answers a different question: are two catalogue positions the same wine
object, or different wines whose physical packages are easy to confuse?

The image descriptor intentionally discards most photometric information.  It is
based on edges after alpha-cropping and fixed-size normalization, so a new render,
different exposure, or slightly different crop does not by itself create a new
object.  Catalogue attributes then separate the two cases:

* same semantic product + same package design -> duplicate, not near-duplicate;
* different vintage/grape/category/sugar + same package design -> near-duplicate;
* one reference file assigned to different products -> data collision, not visual
  evidence that the physical products are identical.

The script is deliberately conservative.  It reports candidates and evidence;
manual front-label review remains authoritative because the catalogue omits some
vintages and contains incorrect image links.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import near_duplicates as legacy  # noqa: E402


IMG_DIR = REPO / "data" / "technical" / "strapi" / "img"
EDGE_SIZE = (64, 128)
EDGE_THRESHOLD = 0.84
EDGE_CACHE = REPO / "dev" / "semantic_edge_descriptors.npz"


# Vintages which are legible on the front labels but absent from catalogue fields.
# These values are evidence gathered from the generated contact sheets, not guesses
# from ABV suffixes.
LABEL_VINTAGE = {
    "abrau-dyurso-abrau-dyurso-pino-nuar-krasnoe-suhoe-13": "2023",
    "abrau-dyurso-pino-nuar-krasnoe-suhoe-12": "2023",
    "abrau-dyurso-pino-nuar-krasnoe-suhoe-125": "2023",
    "abrau-dyurso-risling-beloe-suhoe-12": "2022",
    "abrau-dyurso-risling-beloe-suhoe-13": "2023",
    "abrau-dyurso-shardone-beloe-suhoe-12": "2023",
    "abrau-dyurso-shardone-beloe-suhoe-13": "2023",
    "alma-valley-merlo-rezerv-krasnoe-suhoe-14": "2021",
    "alma-valley-merlo-rezerv-krasnoe-suhoe-15": "2021",
    "alma-valley-shardone-rezerv-beloe-suhoe-135": "2021",
    "alma-valley-shardone-rezerv-beloe-suhoe-14": "2020",
    "esse-brut-zero-dosage-shardone-beloe-ekstra-bryut-12": "2013",
    "esse-pino-gri-beloe-suhoe-12": "2021",
    "esse-pino-gri-beloe-suhoe-135": "2024",
}


# Reviewed aliases: both catalogue positions show the same front-label object.
# The photographs/renders are not necessarily pixel-identical.
SAME_OBJECT_PAIRS = {
    frozenset(pair)
    for pair in [
        ("abrau-dyurso-abrau-dyurso-pino-nuar-krasnoe-suhoe-13",
         "abrau-dyurso-pino-nuar-krasnoe-suhoe-125"),
        ("abrau-dyurso-pino-nuar-krasnoe-suhoe-12",
         "abrau-dyurso-pino-nuar-krasnoe-suhoe-125"),
        ("abrau-dyurso-brut-dor-blanc-de-blancs-shardone-beloe-bryut-12",
         "abrau-dyurso-brut-dor-blanc-de-blancs-shardone-beloe-bryut-125"),
        ("abrau-dyurso-brut-dor-blanc-de-noirs-pino-nuar-beloe-bryut-115",
         "abrau-dyurso-brut-dor-blanc-de-noirs-pino-nuar-beloe-bryut-125"),
        ("abrau-dyurso-shardone-beloe-suhoe-12",
         "abrau-dyurso-shardone-beloe-suhoe-13"),
        ("alma-valley-merlo-rezerv-krasnoe-suhoe-14",
         "alma-valley-merlo-rezerv-krasnoe-suhoe-15"),
        ("belbek-belbek-kaberne-fran-krasnoe-suhoe-13",
         "belbek-kaberne-fran-krasnoe-suhoe-134"),
        ("belbek-belbek-muskat-beloe-suhoe-128",
         "belbek-muskat-muskat-belyy-beloe-suhoe-127"),
        ("belbek-belbek-pino-nuar-krasnoe-suhoe-13",
         "belbek-pino-nuar-krasnoe-suhoe-133"),
        ("belbek-belbek-risling-beloe-suhoe-128",
         "belbek-risling-risling-reynskiy-beloe-suhoe-12"),
        ("belbek-belbek-rkatsiteli-barrik-beloe-suhoe-13",
         "belbek-rkatsiteli-barrik-beloe-suhoe-13"),
        ("domaine-lipko-penpalo-cabernet-sauvignon-domaine-lipko-kaberne-sovinon-krasnoe-suhoe-14",
         "domaine-lipko-penpalo-kaberne-sovinon-krasnoe-suhoe-14"),
        ("domaine-lipko-white-blend-lipko-muskat-belyy-beloe-polusuhoe-115",
         "domaine-lipko-white-blend-muskat-beloe-polusuhoe-12"),
        ("esse-rose-cuvee-prestige-pino-nuar-rozovoe-bryut-115",
         "esse-rose-cuvee-prestige-pino-nuar-rozovoe-bryut-13"),
        ("belmas-winery-sauvignon-blanc-katya-belmas-sovinon-blan-beloe-suhoe-115",
         "belmas-winery-sauvignon-blanc-katya-sovinon-blan-beloe-suhoe-115"),
        ("aligote-avtorskoe", "aligote-avtorskoe-vino"),
        ("kokur-2025", "kokur-suhoe-2025"),
        # Both cards in each pair share the same binary and the same front label.
        # The Aristov label explicitly says 2020; the second slug omits the year.
        ("kuban-vino-aristov-kaberne-sovinon-roze-2020-rozovoe-suhoe-12",
         "kuban-vino-aristov-kaberne-sovinon-roze-rozovoe-suhoe-12"),
        # Both Villa Urkusta cards show Cabernet Sauvignon Classic 2023.
        ("villa-urkusta-kaberne-sovinon-klassik-krasnoe-suhoe-12",
         "villa-urkusta-kaberne-sovinon-klassik-krasnoe-suhoe-139"),
    ]
}


# Same catalogue wine, but the references show a redesign rather than a confusing
# neighbour.  These are duplicate/versioned cards, not near-duplicates.
SAME_WINE_REDESIGN_PAIRS = {
    frozenset((
        "novyj-svet-shardone-kyuve-de-prestizh",
        "novyj-svet-shardone-kyuve-de-prestizh-1",
    )),
    frozenset((
        "derbent-vino-di-kaspiko-shardone-beloe-bryut-105-125",
        "igristoe-vino-bryut-beloe-di-caspico",
    )),
}


# Exhaustive review of the broad candidates for which structured catalogue
# attributes alone did not settle the object relation.  These overrides prevent
# similarity of files from silently becoming a product-level verdict.
MANUAL_PAIR_REVIEW = {
    frozenset((slug_a, slug_b)): relation
    for slug_a, slug_b, relation in [
        # Different products with a genuinely close package system.
        ("fanagoriya-primum-alveus-brut-2016-shardone-igristoe-bryut-beloe-12",
         "fanagoriya-primum-alveus-extra-brut-2016-shardone-beloe-bryut-12",
         "different_wines"),
        ("fanagoriya-brule-muscat-ottonel-brut-muskat-ottonel-beloe-polusladkoe-12",
         "fanagoriya-brule-muscat-ottonel-polusladkoe-muskat-ottonel-beloe-115",
         "different_wines"),
        ("marquetry-syrah", "marquetry-syrah-merlot-cabernet-franc", "different_wines"),
        ("belbek-belbek-sira-krasnoe-suhoe-132",
         "belbek-sira-rezerv-krasnoe-suhoe-132", "different_wines"),
        ("abrau-dyurso-victor-dravigny-brut-shardone-beloe-bryut-12",
         "abrau-dyurso-victor-dravigny-koshernoe-bryut-shardone-beloe-125",
         "different_wines"),
        ("cantiani-rkatsiteli-1", "cantiani-semisweet", "different_wines"),
        ("belbek-pino-nuar-krasnoe-suhoe-133",
         "belbek-pino-nuar-rezerv-krasnoe-suhoe-135", "different_wines"),
        ("vibes-silvaner-2022", "vibes-silvaner-barrel-fermented-2022",
         "different_wines"),
        ("fanagoriya-100-ottenkov-krasnogo-kaberne-kaberne-sovinon-krasnoe-suhoe-135",
         "fanagoriya-101-ottenok-krasnogo-kaberne-kaberne-sovinon-krasnoe-suhoe-14",
         "different_wines"),
        ("fanagoriya-100-ottenkov-krasnogo-saperavi-krasnoe-suhoe-14",
         "fanagoriya-101-ottenok-krasnogo-saperavi-krasnoe-suhoe-14",
         "different_wines"),
        ("fanagoriya-100-ottenkov-krasnogo-pino-nuar-krasnoe-suhoe-135",
         "fanagoriya-101-ottenok-krasnogo-pino-nuar-krasnoe-suhoe-135",
         "different_wines"),
        ("chateau-de-talu-yuzhnaya-vertikal-kaberne-fran-krasnoe-suhoe-142",
         "chateau-de-talu-yuzhnaya-vertikal-kaberne-fran-rezerv-krasnoe-suhoe-146",
         "different_wines"),
        ("chateau-de-talu-yuzhnaya-vertikal-kaberne-fran-premium-krasnoe-suhoe-143",
         "chateau-de-talu-yuzhnaya-vertikal-kaberne-fran-rezerv-krasnoe-suhoe-146",
         "different_wines"),
        ("chateau-de-talu-yuzhnaya-vertikal-kaberne-fran-krasnoe-suhoe-142",
         "chateau-de-talu-yuzhnaya-vertikal-kaberne-fran-premium-krasnoe-suhoe-143",
         "different_wines"),

        # Same front-label object in a different render.
        ("belmas-winery-sauvignon-blanc-katya-belmas-sovinon-blan-beloe-suhoe-115",
         "belmas-winery-sauvignon-blanc-katya-sovinon-blan-beloe-suhoe-115",
         "same_object"),

        # Wrong/duplicated reference, not evidence about the physical products.
        ("esse-demi-sec-muscat-nectar-muskat-belyy-beloe-ekstra-bryut-115",
         "esse-muskat-muskat-belyy-beloe-ekstra-bryut-115", "data_collision"),
        ("zhemchuzhnaya-9-czitron-shardone", "zhemchuzhnaya-9-czitron-shardone-1",
         "data_collision"),
        ("magnatum-blanc-de-blancs", "mantra-blanc-de-blancs", "data_collision"),
        ("magnatum-blanc-de-blancs", "magnatum-cuvee-m-blanc-de-blancs",
         "data_collision"),

        # Similar names/attributes, but the physical packages are not confusing.
        ("belmas-winery-riesling-belmas-risling-beloe-suhoe-135",
         "belmas-winery-riesling-katya-belmas-risling-beloe-suhoe-125",
         "visually_distinct"),
        ("belmas-winery-sauvignon-blanc-belmas-sovinon-blan-beloe-suhoe-13",
         "belmas-winery-sauvignon-blanc-katya-belmas-sovinon-blan-beloe-suhoe-115",
         "visually_distinct"),
        ("esse-sovinon-blan-beloe-suhoe-12", "esse-sovinon-blan-fyume-beloe-suhoe-128",
         "visually_distinct"),
        ("bogovich-wine-vineyard-pino-nuar-2-krasnoe-suhoe-12",
         "bogovich-wine-vineyard-pino-nuar-krasnoe-suhoe-125", "visually_distinct"),
        ("cantiani-rkatsiteli", "cantiani-rkatsiteli-1", "visually_distinct"),
        ("cantiani-riesling", "cantiani-riesling-1", "visually_distinct"),
        ("belmas-winery-viognier-belmas-vione-beloe-suhoe-122",
         "belmas-winery-viognier-katya-belmas-vione-beloe-suhoe-135",
         "visually_distinct"),
    ]
}

# Manual review of series that had paired references in the first volume.
# Newly image-backed series in the full dump default to ``needs_review``.
# Keys are exactly ``series_groups`` keys. ``true_near`` means different wine
# objects with a confusingly similar package system.
SERIES_REVIEW = {
    ("Коммуналка", "алиготе баррель"): "data_collision",
    ("WINEMAFIA", "amelia"): "true_near",
    ("Абрау-Дюрсо", "brut d or blanc de blancs"): "same_object",
    ("Абрау-Дюрсо", "brut d or blanc de noirs"): "same_object",
    ("ESSE", "brut zero dosage"): "needs_review",
    ("Vibes", "cabernet franc pinot noir"): "data_collision",
    ("Шато АЛВИСА", "cantiani riesling"): "visually_distinct",
    ("Шато АЛВИСА", "cantiani rkatsiteli"): "visually_distinct",
    ("WINEMAFIA", "daniel"): "true_near",
    ("WINEMAFIA", "david"): "true_near",
    ("AYA Organic Wine & Vineyards", "evolution pinot noir"): "visually_distinct",
    ("Château Le Grand Vostock", "fagotine"): "true_near",
    ("Агролайн", "heritage dg skin contact rkatsiteli"): "true_near",
    ("Фанагория", "par amour"): "true_near",
    ("Фанагория", "primum alveus brut"): "true_near",
    ("Фанагория", "primum alveus extra brut"): "true_near",
    ("AYA Organic Wine & Vineyards", "purity in syrah"): "true_near",
    ("ESSE", "rose cuvee prestige"): "same_object",
    ("Фанагория", "velvet season"): "true_near",
    ("Domaine Lipko", "white blend"): "true_near",
    ("Alma Valley", "гравити"): "true_near",
    ("Дербент Вино", "ди каспико"): "true_near",
    ("АРАТТИ", "жемчужная 9 цитрон шардоне"): "data_collision",
    ("Дербент Вино", "кавказиан"): "data_collision",
    ("Alma Valley", "мерло резерв"): "same_object",
    ("Новый Свет. Дом шампанских вин", "новый свет шардоне кюве де престиж"): "same_wine_redesign",
    ("ESSE", "пино гри"): "true_near",
    ("Абрау-Дюрсо", "пино нуар"): "same_object",
    ("Alma Valley", "пино нуар"): "visually_distinct",
    ("Абрау-Дюрсо", "рислинг"): "true_near",
    ("Alma Valley", "солнце воздух виноград"): "true_near",
    ("Абрау-Дюрсо", "шардоне"): "same_object",
    ("Alma Valley", "шардоне резерв"): "true_near",
    ("Дербент Вино", "эндемы каберне совиньон"): "visually_distinct",
    ("Дербент Вино", "эндемы саперави"): "data_collision",
    ("Дербент Вино", "эндемы шардоне"): "data_collision",
}


def preferred_references() -> tuple[dict[str, dict], dict[str, str]]:
    """Return the Catalog's single collision-reviewed reference per slug."""
    catalog, slug_files = legacy.load_catalog_with_images()
    return catalog, {slug: files[0] for slug, files in slug_files.items()}


def normalized_image(path: Path) -> Image.Image:
    """Alpha-crop an image and composite it on white."""
    with Image.open(path) as source:
        source.load()
        if source.mode in {"RGBA", "LA"} or (source.mode == "P" and "transparency" in source.info):
            rgba = source.convert("RGBA")
            box = rgba.getchannel("A").getbbox()
            if box:
                rgba = rgba.crop(box)
            image = Image.new("RGB", rgba.size, "white")
            image.paste(rgba, mask=rgba.getchannel("A"))
            return image
        return source.convert("RGB")


def edge_descriptor(filename: str) -> np.ndarray:
    """Photometry-resistant full-object descriptor used only for candidate search."""
    image = normalized_image(IMG_DIR / filename).resize(EDGE_SIZE, Image.Resampling.LANCZOS)
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    gradient = np.maximum(np.hypot(gx, gy) - 0.03, 0.0)
    vector = gradient.ravel()
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def edge_descriptors(slugs: list[str], references: dict[str, str]) -> np.ndarray:
    """Load or build the deterministic edge descriptor matrix."""
    filenames = [references[slug] for slug in slugs]
    signatures = []
    for filename in filenames:
        stat = (IMG_DIR / filename).stat()
        signatures.append(f"{filename}|{stat.st_size}|{stat.st_mtime_ns}")
    reusable = {}
    if EDGE_CACHE.exists():
        with np.load(EDGE_CACHE) as cached:
            if "signatures" in cached.files and cached["signatures"].tolist() == signatures:
                return cached["descriptors"]
            if "signatures" in cached.files and "descriptors" in cached.files:
                reusable = dict(zip(cached["signatures"].tolist(), cached["descriptors"], strict=True))
    descriptors = np.stack([
        reusable[signature] if signature in reusable else edge_descriptor(filename)
        for signature, filename in zip(signatures, filenames, strict=True)
    ])
    EDGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EDGE_CACHE,
        filenames=np.asarray(filenames),
        signatures=np.asarray(signatures),
        descriptors=descriptors,
    )
    return descriptors


def label_year(slug: str, catalog: dict[str, dict]) -> str | None:
    """Best available vintage: manual label read, then catalogue name/slug."""
    if slug in LABEL_VINTAGE:
        return LABEL_VINTAGE[slug]
    attrs = legacy.parse_attrs(slug, catalog[slug])
    return attrs["year"]


def semantic_difference(slug_a: str, slug_b: str, catalog: dict[str, dict]) -> list[str]:
    """Meaningful product attributes which make two positions different wines."""
    attrs_a = legacy.parse_attrs(slug_a, catalog[slug_a])
    attrs_b = legacy.parse_attrs(slug_b, catalog[slug_b])
    values = [
        ("винтаж", label_year(slug_a, catalog), label_year(slug_b, catalog)),
        ("категория", attrs_a["category"], attrs_b["category"]),
        ("сорт", attrs_a["grape"], attrs_b["grape"]),
        ("сахар", attrs_a["sugar"], attrs_b["sugar"]),
    ]
    return [label for label, value_a, value_b in values if value_a != value_b]


def name_affinity(slug_a: str, slug_b: str, catalog: dict[str, dict]) -> float:
    """Similarity of product names, used to retain text candidates missed by vision."""
    name_a = legacy.norm_name(catalog[slug_a]["Название вина"])
    name_b = legacy.norm_name(catalog[slug_b]["Название вина"])
    return SequenceMatcher(None, name_a, name_b).ratio()


def reviewed_relation(slug_a: str, slug_b: str, catalog: dict[str, dict]) -> str:
    """Classify a reviewed/candidate pair without treating file pixels as identity."""
    pair = frozenset((slug_a, slug_b))
    if pair in SAME_OBJECT_PAIRS:
        return "same_object"
    if pair in SAME_WINE_REDESIGN_PAIRS:
        return "same_wine_redesign"
    if pair in MANUAL_PAIR_REVIEW:
        return MANUAL_PAIR_REVIEW[pair]
    if semantic_difference(slug_a, slug_b, catalog):
        return "different_wines"
    return "needs_review"


def connected_components(pairs: list[tuple[str, str]]) -> list[list[str]]:
    """Connected components of an undirected pair relation."""
    neighbours: dict[str, set[str]] = collections.defaultdict(set)
    for slug_a, slug_b in pairs:
        neighbours[slug_a].add(slug_b)
        neighbours[slug_b].add(slug_a)
    seen: set[str] = set()
    components = []
    for start in neighbours:
        if start in seen:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            slug = stack.pop()
            component.append(slug)
            for neighbour in neighbours[slug]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    return components


def reviewed_series(catalog: dict[str, dict], references: dict[str, str]) -> dict:
    """Summarize the exhaustive manual review of image-backed text series."""
    rows = []
    for key, slugs in legacy.series_groups(catalog).items():
        with_reference = [slug for slug in slugs if slug in references]
        if len(with_reference) < 2:
            continue
        rows.append({
            "winery": key[0],
            "series": key[1],
            # A fuller media dump can reveal series that had no paired references
            # when the manual table was written.  Keep those pairs unreviewed.
            "relation": SERIES_REVIEW.get(key, "needs_review"),
            "slugs": slugs,
            "slugs_with_reference": with_reference,
        })
    counts = collections.Counter(row["relation"] for row in rows)
    positions = {
        relation: len({
            slug
            for row in rows if row["relation"] == relation
            for slug in row["slugs_with_reference"]
        })
        for relation in counts
    }
    return {"groups": len(rows), "relations": dict(counts), "positions": positions, "rows": rows}


def legacy_visual_audit(catalog: dict[str, dict]) -> dict:
    """Reclassify the complete visual-candidate table generated by the legacy scan."""
    index_path = REPO / "data" / "near_duplicates" / "index.md"
    pairs = []
    if index_path.exists():
        quote = "`"
        pattern = re.compile(
            re.escape(quote) + "([^" + quote + "]+)" + re.escape(quote)
            + r" (?:<br>|↔) "
            + re.escape(quote) + "([^" + quote + "]+)" + re.escape(quote)
        )
        for line in index_path.read_text(encoding="utf-8").splitlines():
            match = pattern.search(line)
            if match and ("C-visually-close" in line or "↔" in line):
                pairs.append(match.groups())
    else:
        # Fresh checkout: rebuild the same candidate set.  This is slower, but keeps
        # the object-level report reproducible without committing the large gallery.
        legacy_state = legacy.build_groups(with_visual=True, max_visual_pairs=600)
        pairs = [
            tuple(group["slugs"])
            for group in legacy_state["groups"]
            if group["type"] == "C_visually_close"
        ]
    by_relation: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for pair in pairs:
        by_relation[reviewed_relation(*pair, catalog)].append(pair)
    same_object = by_relation["same_object"]
    true_near = by_relation["different_wines"]
    redesign = by_relation["same_wine_redesign"]
    data_collision = by_relation["data_collision"]
    visually_distinct = by_relation["visually_distinct"]
    unresolved = by_relation["needs_review"]
    components = connected_components(true_near)
    return {
        "screened_pairs": len(pairs),
        "same_object_pairs_removed": len(same_object),
        "same_wine_redesign_pairs_removed": len(redesign),
        "data_collision_pairs_removed": len(data_collision),
        "visually_distinct_pairs_removed": len(visually_distinct),
        "unresolved_pairs_removed": len(unresolved),
        "true_near_pairs": len(true_near),
        "true_near_positions": len({slug for pair in true_near for slug in pair}),
        "true_near_components": len(components),
        "components": components,
    }


def confirmed_registry(catalog: dict[str, dict], references: dict[str, str]) -> dict:
    """Keep reviewed pairs separate from pairs proposed by the image search."""
    path = REPO / "data" / "near_duplicates" / "all_candidates.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    pairs = []
    for row in rows:
        slug_a, slug_b = row["slug_1"], row["slug_2"]
        if row["review_verdict"] != "confirmed_near_duplicate" or row["is_confirmed_near_duplicate"] != "true":
            raise ValueError(f"Unreviewed pair in confirmed registry: {slug_a}, {slug_b}")
        if slug_a not in catalog or slug_b not in catalog or slug_a not in references or slug_b not in references:
            raise ValueError(f"Confirmed pair missing Catalog/reference: {slug_a}, {slug_b}")
        pairs.append(tuple(sorted((slug_a, slug_b))))
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate pair in confirmed registry")
    components = connected_components(pairs)
    return {
        "pairs": len(pairs),
        "families": len(components),
        "catalog_positions": len({slug for pair in pairs for slug in pair}),
        "components": components,
    }


def build_audit(edge_threshold: float = EDGE_THRESHOLD, include_legacy_visual: bool = False) -> dict:
    """Build object-level candidates and summary."""
    catalog, references = preferred_references()
    slugs = sorted(references)
    descriptors = edge_descriptors(slugs, references)
    similarity = descriptors @ descriptors.T

    candidates = []
    for index_a, index_b in itertools.combinations(range(len(slugs)), 2):
        slug_a, slug_b = slugs[index_a], slugs[index_b]
        same_winery = catalog[slug_a]["Винодельня"].strip() == catalog[slug_b]["Винодельня"].strip()
        pair = frozenset((slug_a, slug_b))
        if not same_winery and pair not in SAME_OBJECT_PAIRS:
            continue
        edge_score = float(similarity[index_a, index_b])
        text_score = name_affinity(slug_a, slug_b, catalog)
        if edge_score < edge_threshold and text_score < 0.82 and pair not in (
            SAME_OBJECT_PAIRS | SAME_WINE_REDESIGN_PAIRS
        ):
            continue
        candidates.append({
            "slugs": [slug_a, slug_b],
            "winery": catalog[slug_a]["Винодельня"].strip(),
            "names": [catalog[slug_a]["Название вина"].strip(),
                      catalog[slug_b]["Название вина"].strip()],
            "files": [references[slug_a], references[slug_b]],
            "edge_similarity": round(edge_score, 4),
            "name_similarity": round(text_score, 4),
            "semantic_difference": semantic_difference(slug_a, slug_b, catalog),
            "relation": reviewed_relation(slug_a, slug_b, catalog),
        })

    # A single physical file assigned to semantically different products is a
    # linking/data collision.  It is excluded from visual near-duplicate evidence.
    file_slugs: dict[str, list[str]] = collections.defaultdict(list)
    for slug, filename in references.items():
        digest = hashlib.sha256((IMG_DIR / filename).read_bytes()).hexdigest()
        file_slugs[digest].append(slug)
    shared_file_groups = []
    for digest, grouped_slugs in file_slugs.items():
        if len(grouped_slugs) < 2:
            continue
        relations = [
            reviewed_relation(a, b, catalog)
            for a, b in itertools.combinations(sorted(grouped_slugs), 2)
        ]
        shared_file_groups.append({
            "sha256": digest,
            "files": sorted({references[slug] for slug in grouped_slugs}),
            "slugs": sorted(grouped_slugs),
            "object_relation": (
                "same_object" if relations and all(r == "same_object" for r in relations)
                else "data_collision" if "data_collision" in relations
                else "needs_review"
            ),
            "relations": relations,
        })

    counts = collections.Counter(candidate["relation"] for candidate in candidates)
    series_audit = reviewed_series(catalog, references)
    visual_audit = legacy_visual_audit(catalog) if include_legacy_visual else {"status": "not_run"}
    shared_counts = collections.Counter(group["object_relation"] for group in shared_file_groups)
    same_object_components = connected_components([
        tuple(sorted(pair)) for pair in SAME_OBJECT_PAIRS
    ])
    same_object_summary = {
        "pair_relations": len(SAME_OBJECT_PAIRS),
        "groups": len(same_object_components),
        "positions": len({slug for component in same_object_components for slug in component}),
    }
    redesign_components = connected_components([
        tuple(sorted(pair)) for pair in SAME_WINE_REDESIGN_PAIRS
    ])
    redesign_summary = {
        "pair_relations": len(SAME_WINE_REDESIGN_PAIRS),
        "groups": len(redesign_components),
        "positions": len({slug for component in redesign_components for slug in component}),
    }
    image_backed_unique_objects = len(references) - sum(
        max(0, len(set(component) & references.keys()) - 1)
        for component in same_object_components
    )
    registry = confirmed_registry(catalog, references)
    return {
        "summary": {
            "catalog_positions": len(catalog),
            "positions_with_reference": len(references),
            "image_backed_unique_objects": image_backed_unique_objects,
            "broad_candidate_pairs": len(candidates),
            "broad_candidate_relations": dict(counts),
            "shared_reference_groups": len(shared_file_groups),
            "shared_reference_relations": dict(shared_counts),
            "confirmed_same_object": same_object_summary,
            "confirmed_same_wine_redesign": redesign_summary,
            "edge_threshold": edge_threshold,
            "reviewed_series": {key: value for key, value in series_audit.items() if key != "rows"},
            "manual_pair_review": dict(collections.Counter(MANUAL_PAIR_REVIEW.values())),
            "legacy_visual_reclassified": {
                key: value for key, value in visual_audit.items() if key != "components"
            },
            "confirmed_true_near_union": {key: value for key, value in registry.items() if key != "components"},
        },
        "candidates": sorted(
            candidates,
            key=lambda row: (row["relation"], -row["edge_similarity"], row["slugs"]),
        ),
        "shared_reference_groups": shared_file_groups,
        "confirmed_same_object": {
            **same_object_summary,
            "components": same_object_components,
        },
        "confirmed_same_wine_redesign": {
            **redesign_summary,
            "components": redesign_components,
        },
        "reviewed_series": series_audit,
        "legacy_visual_reclassified": visual_audit,
        "confirmed_true_near_union": registry,
    }


def print_report(audit: dict) -> None:
    """Print a concise, review-oriented report."""
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))
    for relation in (
        "same_object", "same_wine_redesign", "different_wines", "data_collision",
        "visually_distinct", "needs_review",
    ):
        rows = [row for row in audit["candidates"] if row["relation"] == relation]
        print(f"\n{relation}: {len(rows)}")
        for row in rows[:30]:
            differences = ", ".join(row["semantic_difference"]) or "нет"
            print(
                f"  {row['edge_similarity']:.3f} | {differences:24s} | "
                f"{row['slugs'][0]} <> {row['slugs'][1]}"
            )
        if len(rows) > 30:
            print(f"  ... ещё {len(rows) - 30}; полный список: --json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable audit")
    parser.add_argument("--output", type=Path, help="write the report to this UTF-8 file")
    parser.add_argument("--edge-threshold", type=float, default=EDGE_THRESHOLD)
    parser.add_argument("--legacy-visual", action="store_true",
                        help="also run the slower pixel-level candidate scan")
    args = parser.parse_args()
    audit = build_audit(args.edge_threshold, include_legacy_visual=args.legacy_visual)
    if args.output:
        payload = json.dumps(audit, ensure_ascii=False, indent=2)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    elif args.json:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    else:
        print_report(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
