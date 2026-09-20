"""Build the manually reviewed OCR feature dataset for ``data/test``.

The catalog tells us which product a directory represents, but it is not used to
invent label text.  ``FEATURES_BY_FOLDER`` contains only text that was confirmed
on the query image(s).  Image-specific additions capture vintages that differ
between photographs of the same catalog product.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "test"
CATALOG_PATH = ROOT / "data" / "strapi" / "catalog_dataset.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "dataset.jsonl"


def feature(kind: str, value: str, role: str, *aliases: str) -> dict[str, Any]:
    """Create one gold feature using the label spelling as the canonical value."""
    item: dict[str, Any] = {"kind": kind, "value": value, "role": role}
    if aliases:
        item["aliases"] = list(aliases)
    return item


ID = "identity"
D = "disambiguation"
S = "secondary"
f = feature


FEATURES_BY_FOLDER: dict[str, list[dict[str, Any]]] = {
    "abrau-dyurso-abrau-durso-brut-rose-reserve-pino-nuar-beloe-bryut-12": [
        f("brand", "ABRAU-DURSO", ID, "АБРАУ-ДЮРСО"),
        f("line", "RESERVE", ID, "РЕЗЕРВ"),
        f("wine_type", "SPARKLING WINE", D, "ИГРИСТОЕ ВИНО"),
        f("color", "РОЗОВОЕ", D, "ROSÉ", "ROSE"),
    ],
    "abrau-dyurso-imperial-brut-rose-pino-nuar-rozovoe-bryut-12": [
        f("brand", "АБРАУ-ДЮРСО", ID, "ABRAU-DURSO"),
        f("line", "ИМПЕРИАЛ", ID, "IMPERIAL"),
        f("sugar", "BRUT", D, "БРЮТ"),
        f("color", "ROSE", D, "ROSÉ", "РОЗОВОЕ"),
        f("vintage", "2018", D),
    ],
    "abrau-dyurso-russkoe-igristoe-polusladkoe-shardone-beloe-12": [
        f("brand", "АБРАУ-ДЮРСО", ID, "ABRAU-DURSO"),
        f("wine_type", "РУССКОЕ ИГРИСТОЕ", ID),
        f("sugar", "ПОЛУСЛАДКОЕ", D),
        f("color", "БЕЛОЕ", D),
    ],
    "agora-rosa-viva-cabernet-sauvignon-shiraz": [
        f("producer", "AGORA WINERY", ID),
        f("product", "ROSA VIVA", ID),
        f("grape", "CABERNET SAUVIGNON", D, "КАБЕРНЕ СОВИНЬОН"),
        f("grape", "SHIRAZ", D, "ШИРАЗ", "СИРА"),
        f("wine_type", "RED DRY", D, "КРАСНОЕ СУХОЕ"),
    ],
    "agora-yachting-chardonnay": [
        f("brand", "AGORA", ID),
        f("line", "YACHTING", ID),
        f("grape", "CHARDONNAY", D, "ШАРДОНЕ"),
        f("wine_type", "WHITE DRY WINE", D, "БЕЛОЕ СУХОЕ ВИНО"),
        f("line", "RESERVE", D, "РЕЗЕРВ"),
    ],
    "alma-valley-pino-nuar-beloe-ekstra-bryut-115": [
        f("brand", "ALMA VALLEY", ID),
        f("product", "BLANC DE NOIR", ID, "BLANC DE NOIRS"),
        f("sugar", "BRUT NATURE", D, "ЭКСТРА БРЮТ"),
    ],
    "alma-valley-semilon-beloe-suhoe-135": [
        f("brand", "ALMA VALLEY", ID),
        f("grape", "SÉMILLON", D, "SEMILLON", "СЕМИЛЬОН"),
        f("vintage", "2023", D),
        f("wine_type", "БЕЛОЕ СУХОЕ ВИНО", D, "WHITE DRY WINE"),
        f("region", "КРЫМ", S, "CRIMEA"),
    ],
    "alma-valley-shardone-beloe-suhoe-135": [
        f("brand", "ALMA VALLEY", ID),
        f("grape", "CHARDONNAY", D, "ШАРДОНЕ"),
        f("wine_type", "БЕЛОЕ СУХОЕ ВИНО", D, "WHITE DRY WINE"),
        f("region", "КРЫМ", S, "CRIMEA"),
    ],
    "alma-valley-shiraz-sira-rezerv-krasnoe-suhoe-14": [
        f("brand", "ALMA VALLEY", ID),
        f("line", "RESERVE", D, "РЕЗЕРВ"),
        f("grape", "SHIRAZ", D, "ШИРАЗ", "СИРА"),
        f("vintage", "2021", D),
        f("wine_type", "КРАСНОЕ СУХОЕ ВИНО", D, "RED DRY WINE"),
    ],
    "alma-valley-traminer-beloe-suhoe-125": [
        f("brand", "ALMA VALLEY", ID),
        f("grape", "TRAMINER", D, "ТРАМИНЕР"),
        f("vintage", "2021", D),
        f("wine_type", "БЕЛОЕ СУХОЕ ВИНО", D, "WHITE DRY WINE"),
    ],
    "aya-organic-wine-vineyards-purity-in-syrah-sira-krasnoe-suhoe-146": [
        f("brand", "AYA", ID, "АЯ"),
        f("product", "PURITY IN SYRAH", ID),
        f("grape", "SYRAH", D, "SHIRAZ", "СИРА", "ШИРАЗ"),
        f("vintage", "2023", D),
    ],
    "aya-organic-wine-vineyards-purity-in-trinity-pino-nuar-rozovoe-suhoe-13": [
        f("brand", "AYA", ID, "АЯ"),
        f("product", "PURITY IN TRINITY", ID),
        f("vintage", "2023", D),
    ],
    "belbek-sira-rezerv-krasnoe-suhoe-132": [
        f("brand", "БЕЛЬБЕК", ID, "BELBEK"),
        f("grape", "СИРА", D, "SYRAH", "SHIRAZ", "ШИРАЗ"),
        f("line", "РЕЗЕРВ", D, "RESERVE"),
        f("vintage", "2021", D),
    ],
    "bogovich-wine-vineyard-klassika-pino-nuar-krasnoe-suhoe-125": [
        f("producer", "IRINA BOGOVICH", ID, "ИРИНА БОГОВИЧ"),
        f("grape", "PINOT NOIR", D, "ПИНО НУАР"),
    ],
    "bogovich-wine-vineyard-rkatsiteli-oranzh-beloe-suhoe-125": [
        f("producer", "BOGOVICH", ID, "БОГОВИЧ"),
        f("grape", "RKATSITELI", D, "РКАЦИТЕЛИ"),
        f("wine_type", "ORANGE", D, "ОРАНЖ"),
    ],
    "cantiani-riesling-1": [
        f("brand", "CANTIANI", ID, "КАНТИАНИ"),
        f("wine_type", "SPARKLING WINE", D, "ИГРИСТОЕ ВИНО"),
        f("grape", "RIESLING", D, "РИСЛИНГ"),
        f("sugar", "BRUT", D, "БРЮТ"),
    ],
    "cantiani-riesling-rkatsiteli": [
        f("brand", "CANTIANI", ID, "КАНТИАНИ"),
        f("grape", "RIESLING", D, "РИСЛИНГ"),
        f("grape", "RKATSITELI", D, "РКАЦИТЕЛИ"),
    ],
    "cantiani-semisweet": [
        f("brand", "CANTIANI", ID, "КАНТИАНИ"),
        f("color", "WHITE", D, "БЕЛОЕ"),
        f("sugar", "SEMISWEET", D, "ПОЛУСЛАДКОЕ"),
    ],
    "chateau-andre-gryuner-beloe-suhoe-13": [
        f("producer", "CHÂTEAU ANDRÉ", ID, "CHATEAU ANDRE", "ШАТО АНДРЕ"),
        f("grape", "GRÜNER VELTLINER", D, "GRUNER VELTLINER", "ГРЮНЕР ВЕЛЬТЛИНЕР"),
        f("vintage", "2022", D),
    ],
    "chateau-cachalot-muskat-blan-muskat-belyy-beloe-suhoe-117": [
        f("producer", "CHÂTEAU CACHALOT", ID, "CHATEAU CACHALOT", "ШАТО КАШАЛОТ"),
        f("grape", "МУСКАТ БЛАН", D, "MUSCAT BLANC"),
        f("wine_type", "СУХОЕ", D, "DRY"),
    ],
    "chateau-de-talu-uroki-frantsuzskogo-shardone-beloe-suhoe-124": [
        f("producer", "CHÂTEAU DE TALU", ID, "CHATEAU DE TALU", "ШАТО ДЕ ТАЛЮ"),
        f("line", "УРОКИ ФРАНЦУЗСКОГО", ID),
        f("grape", "ШАРДОНЕ", D, "CHARDONNAY"),
        f("appellation", "ГЕЛЕНДЖИК", S),
    ],
    "chateau-de-talu-uroki-frantsuzskogo-sovinon-blan-beloe-suhoe-115": [
        f("producer", "CHÂTEAU DE TALU", ID, "CHATEAU DE TALU", "ШАТО ДЕ ТАЛЮ"),
        f("line", "УРОКИ ФРАНЦУЗСКОГО", ID),
        f("grape", "СОВИНЬОН", D, "SAUVIGNON", "СОВИНЬОН БЛАН", "SAUVIGNON BLANC"),
        f("appellation", "ГЕЛЕНДЖИК", S),
    ],
    "chateau-de-talu-yuzhnaya-vertikal-kaberne-fran-krasnoe-suhoe-142": [
        f("producer", "CHÂTEAU DE TALU", ID, "CHATEAU DE TALU", "ШАТО ДЕ ТАЛЮ"),
        f("line", "ЮЖНАЯ ВЕРТИКАЛЬ", ID),
        f("grape", "КАБЕРНЕ ФРАН", D, "CABERNET FRANC"),
    ],
    "chateau-le-grand-vostock-kyuve-karsov-rezerv-shardone-beloe-suhoe-14": [
        f("producer", "CHÂTEAU LE GRAND VOSTOCK", ID, "CHATEAU LE GRAND VOSTOCK"),
        f("product", "CUVÉE KARSOV", ID, "CUVEE KARSOV", "КЮВЕ КАРСОВ"),
        f("line", "RESERVE", D, "РЕЗЕРВ"),
    ],
    "chateau-le-grand-vostock-pino-nuar-rezerv-krasnoe-suhoe-14": [
        f("producer", "CHÂTEAU LE GRAND VOSTOCK", ID, "CHATEAU LE GRAND VOSTOCK"),
        f("grape", "PINOT NOIR", D, "ПИНО НУАР"),
        f("line", "RESERVE", D, "РЕЗЕРВ"),
    ],
    "chateau-tamagne-eno-traminer-i-shardone": [
        f("brand", "CHÂTEAU TAMAGNE", ID, "CHATEAU TAMAGNE", "ШАТО ТАМАНЬ"),
        f("grape", "ТРАМИНЕР", D, "TRAMINER"),
        f("grape", "ШАРДОНЕ", D, "CHARDONNAY"),
    ],
    "chteau-le-grand-vostock-cabernet-franc-reserve-kaberne-fran-krasnoe-suhoe-14": [
        f("producer", "CHÂTEAU LE GRAND VOSTOCK", ID, "CHATEAU LE GRAND VOSTOCK"),
        f("grape", "CABERNET FRANC", D, "КАБЕРНЕ ФРАН"),
        f("line", "RESERVE", D, "РЕЗЕРВ"),
    ],
    "chteau-le-grand-vostock-cadet-karsov-shardone-beloe-suhoe-135": [
        f("producer", "CHÂTEAU LE GRAND VOSTOCK", ID, "CHATEAU LE GRAND VOSTOCK"),
        f("product", "CADET KARSOV", ID, "КАДЕТ КАРСОВ"),
        f("color", "BLANC", D, "БЕЛОЕ"),
    ],
    "chteau-le-grand-vostock-krasnostop-reserve-krasnostop-krasnoe-suhoe-14": [
        f("producer", "CHÂTEAU LE GRAND VOSTOCK", ID, "CHATEAU LE GRAND VOSTOCK"),
        f("grape", "KRASNOSTOP", D, "КРАСНОСТОП"),
        f("line", "RESERVA", D, "RESERVE", "РЕЗЕРВ"),
    ],
    "chteau-le-grand-vostock-le-chene-royal-reserve-merlo-krasnoe-suhoe-145": [
        f("producer", "CHÂTEAU LE GRAND VOSTOCK", ID, "CHATEAU LE GRAND VOSTOCK"),
        f("product", "LE CHÊNE ROYAL", ID, "LE CHENE ROYAL"),
        f("line", "RESERVE", D, "РЕЗЕРВ"),
    ],
    "chteau-le-grand-vostock-pinot-gris-rose-reserve-pino-gri-rozovoe-suhoe-14": [
        f("producer", "CHÂTEAU LE GRAND VOSTOCK", ID, "CHATEAU LE GRAND VOSTOCK"),
        f("grape", "PINOT GRIS", D, "ПИНО ГРИ"),
        f("line", "RESERVE", D, "РЕЗЕРВ"),
        f("color", "ROSÉ", D, "ROSE", "РОЗОВОЕ"),
    ],
    "chteau-le-grand-vostock-vostock-blanc-sovinon-blan-beloe-suhoe-135": [
        f("producer", "CHÂTEAU LE GRAND VOSTOCK", ID, "CHATEAU LE GRAND VOSTOCK"),
        f("product", "VOSTOCK", ID, "ВОСТОК"),
        f("line", "SÉLECTION BLANC", D, "SELECTION BLANC", "БЕЛОЕ"),
    ],
    "cloudy-winery-krash-cabernet-sauvignon-kaberne-sovinon-krasnoe-suhoe-14": [
        f("product", "KRASH", ID, "КРАШ"),
        f("grape", "CABERNET SAUVIGNON", D, "КАБЕРНЕ СОВИНЬОН"),
        f("vintage", "2020", D),
    ],
    "cock-test-belle-kyuve-2-blan-de-nuar-pino-mene-beloe-ekstra-bryut-12": [
        f("product", "Nº 2 CUVÉE", ID, "NO 2 CUVEE", "№ 2 КЮВЕ"),
        f("wine_type", "BLANC DE NOIRS", D, "BLANC DE NOIR", "БЛАН ДЕ НУАР"),
        f(
            "method",
            "MÉTHODE CHAMPENOISE",
            S,
            "METHODE CHAMPENOISE",
            "ШАМПАНСКИЙ МЕТОД",
        ),
    ],
    "denisov-winery-tsitron-oranzh-tsitronnyy-magaracha-oranzhevoe-suhoe-105": [
        f("producer", "DENISOV", ID, "ДЕНИСОВ"),
        f("product", "ОРАНЖ", ID, "ORANGE"),
        f("grape", "ЦИТРОН", D, "TSITRON"),
        f("vintage", "2022", D),
    ],
    "denisov-winery-tsitron-tsitronnyy-magaracha-beloe-suhoe-112": [
        f("producer", "DENISOV", ID, "ДЕНИСОВ"),
        f("grape", "ЦИТРОН", D, "TSITRON"),
        f("wine_type", "СУХОЕ БЕЛОЕ", D, "WHITE DRY"),
        f("abv", "12.9%", S, "12,9%"),
    ],
    "derbent-vino-desono-merlo-krasnoe-suhoe-125": [
        f("brand", "DESONO", ID, "ДЭСОНО"),
        f("grape", "MERLOT", D, "МЕРЛО"),
        f("region", "СДЕЛАНО В ДАГЕСТАНЕ", S, "ДАГЕСТАН"),
    ],
    "derbent-vino-desono-risling-beloe-suhoe-125": [
        f("brand", "DESONO", ID, "ДЭСОНО"),
        f("grape", "RIESLING", D, "РИСЛИНГ"),
        f("region", "СДЕЛАНО В ДАГЕСТАНЕ", S, "ДАГЕСТАН"),
    ],
    "derbent-vino-di-kaspiko-kaberne-sovinon-krasnoe-suhoe-135": [
        f("brand", "DI CASPICO", ID, "ДИ КАСПИКО"),
        f("grape", "КАБЕРНЕ СОВИНЬОН", D, "CABERNET SAUVIGNON"),
    ],
    "derbent-vino-di-kaspiko-risling-beloe-suhoe-12": [
        f("brand", "DI CASPICO", ID, "ДИ КАСПИКО"),
        f("grape", "РИСЛИНГ", D, "RIESLING"),
    ],
    "derbent-vino-di-kaspiko-roze-shardone-rozovoe-suhoe-12": [
        f("brand", "DI CASPICO", ID, "ДИ КАСПИКО"),
        f("color", "РОЗЕ", D, "ROSÉ", "ROSE"),
        f("wine_type", "РОЗОВОЕ СУХОЕ ВИНО", D, "ROSÉ DRY WINE"),
    ],
    "derbent-vino-di-kaspiko-shardone-beloe-polusladkoe-105-125": [
        f("brand", "DI CASPICO", ID, "ДИ КАСПИКО"),
        f("wine_type", "ИГРИСТОЕ ВИНО", D, "SPARKLING WINE"),
        f("color", "БЕЛОЕ", D, "WHITE"),
        f("sugar", "ПОЛУСЛАДКОЕ", D, "SEMISWEET"),
    ],
    "domaine-lipko-penpalo-chardonnay-domaine-lipko-shardone-beloe-suhoe-122": [
        f("brand", "PENPALO", ID, "ПЕНПАЛО"),
        f("producer", "DOMAINE LIPKO", ID, "ДОМЕН ЛИПКО"),
        f("grape", "CHARDONNAY", D, "ШАРДОНЕ"),
        f("vintage", "2020", D),
    ],
    "dubinin-winery-risling-beloe-suhoe-12": [
        f("producer", "DUBININ WINERY", ID, "ДУБИНИН"),
        f("grape", "РИСЛИНГ", D, "RIESLING"),
        f("wine_type", "ВИНО СУХОЕ БЕЛОЕ", D, "WHITE DRY WINE"),
    ],
    "dubinin-winery-roze-kaberne-fran-rozovoe-suhoe-13": [
        f("producer", "DUBININ WINERY", ID, "ДУБИНИН"),
        f("product", "РОЗЕ", ID, "ROSÉ", "ROSE"),
        f("wine_type", "ВИНО СУХОЕ РОЗОВОЕ", D, "ROSÉ DRY WINE"),
    ],
    "dva-brata-pet-nat-kokur-2024": [
        f("brand", "ДВА БРАТА", ID, "DVA BRATA"),
        f("wine_type", "PETNAT", ID, "PET NAT", "ПЕТНАТ", "ПЕТ НАТ"),
        f("grape", "KOKUR", D, "КОКУР"),
        f("vintage", "2024", D),
    ],
    "esse-prirodno-polusladkoe-krasnoe-merlo-13": [
        f("brand", "ESSE", ID, "ЭССЕ"),
        f("sugar", "ПРИРОДНО ПОЛУСЛАДКОЕ", D),
        f("color", "КРАСНОЕ", D, "RED"),
        f("region", "КРЫМ", S, "CRIMEA"),
    ],
    "fanagoriya-alta-qortis-saperavi-krasnoe-suhoe-145": [
        f("producer", "FANAGORIA", ID, "ФАНАГОРИЯ"),
        f("product", "ALTA QORTIS", ID, "АЛЬТА КОРТИС"),
        f("vintage", "2020", D),
        f("line", "QUINTESSENCE", S, "КВИНТЭССЕНЦИЯ"),
    ],
    "fanagoriya-alveus-ultra-cuvee-ekstra-bryut-beloe-risling-reynskiy-12": [
        f("product", "ALVEUS", ID, "АЛЬВЕУС"),
        f("line", "ULTRA CUVEE", ID, "ULTRA CUVÉE", "УЛЬТРА КЮВЕ"),
        f("sugar", "EXTRA BRUT", D, "ЭКСТРА БРЮТ"),
    ],
    "fanagoriya-brule-frizzante-brut-beloe-pino-nuar-bryut-115": [
        f("producer", "FANAGORIA", ID, "ФАНАГОРИЯ"),
        f("product", "BRÛLÉ", ID, "BRULE", "БРЮЛЕ"),
        f("wine_type", "FRIZZANTE", D, "ФРИЗЗАНТЕ"),
        f("color", "BIANCO", D, "BLANC", "БЕЛОЕ"),
        f("sugar", "BRUT", D, "БРЮТ"),
    ],
    "fanagoriya-brule-frizzante-polusladkoe-beloe-pino-nuar-11": [
        f("producer", "FANAGORIA", ID, "ФАНАГОРИЯ"),
        f("product", "BRÛLÉ", ID, "BRULE", "БРЮЛЕ"),
        f("wine_type", "FRIZZANTE", D, "ФРИЗЗАНТЕ"),
        f("color", "BIANCO", D, "БЕЛОЕ"),
        f("sugar", "DEMI-DOLCE", D, "DEMI DOLCE", "ПОЛУСЛАДКОЕ"),
    ],
    "fanagoriya-cru-lermont-saperavi-saperavi-krasnoe-suhoe-135": [
        f("producer", "FANAGORIA", ID, "ФАНАГОРИЯ"),
        f("product", "CRU LERMONT", ID, "КРЮ ЛЕРМОНТ"),
        f("grape", "SAPERAVI", D, "САПЕРАВИ"),
        f("vintage", "2019", D),
        f("wine_type", "ВИНО КРАСНОЕ СУХОЕ", D, "RED DRY WINE"),
    ],
    "fanagoriya-dekanter-formula-q-kaberne-sovinon-krasnoe-suhoe-135": [
        f("producer", "FANAGORIA", ID, "ФАНАГОРИЯ"),
        f("line", "ДЕКАНТЕР", ID, "DECANTER"),
        f("product", "ФОРМУЛА Q", ID, "FORMULA Q"),
        f("line", "КОЛЛЕКЦИОННОЕ", S, "COLLECTION"),
    ],
    "fanagoriya-dekanter-saperavi-2017-krasnoe-suhoe-135": [
        f("producer", "FANAGORIA", ID, "ФАНАГОРИЯ"),
        f("line", "ДЕКАНТЕР", ID, "DECANTER"),
        f("grape", "САПЕРАВИ", D, "SAPERAVI"),
        f("vintage", "2017", D),
        f("line", "КОЛЛЕКЦИОННОЕ", S, "COLLECTION"),
    ],
    "fanagoriya-ona-skazala-da-beloe-bryut": [
        f("producer", "FANAGORIA", ID, "ФАНАГОРИЯ"),
        f("product", "ОНА СКАЗАЛА ДА", ID),
        f("sugar", "БРЮТ", D, "BRUT"),
    ],
    "fanagoriya-rose-kaberne-sovinon-rozovoe-polusuhoe-13": [
        f("producer", "FANAGORIA", ID, "ФАНАГОРИЯ"),
        f("product", "ROSE", ID, "ROSÉ", "РОЗЕ"),
        f("grape", "CABERNET SAUVIGNON", D, "КАБЕРНЕ СОВИНЬОН"),
    ],
    "gavras-levokumskiy-krasnoe-suhoe-12": [
        f("brand", "GAVRAS", ID, "ГАВРАС"),
        f("grape", "ЛЕВОКУМСКИЙ", D, "LEVOKUMSKIY"),
        f("vintage", "2022", D),
        f("region", "КРЫМ", S, "CRIMEA"),
    ],
    "millstream-av-igristoe-molodoe-polusuhoe-beloe": [
        f("brand", "AV", ID),
        f("producer", "MILLSTREAM", ID, "МИЛЬСТРИМ"),
        f("wine_type", "РОССИЙСКОЕ ИГРИСТОЕ ВИНО", D, "RUSSIAN SPARKLING WINE"),
        f("line", "МОЛОДОЕ", D, "YOUNG"),
        f("sugar", "ПОЛУСУХОЕ", D, "SEMI-DRY"),
        f("color", "БЕЛОЕ", D, "WHITE"),
    ],
    "millstream-av-igristoe-molodoe-suhoe-beloe": [
        f("brand", "AV", ID),
        f("producer", "MILLSTREAM", ID, "МИЛЬСТРИМ"),
        f("wine_type", "РОССИЙСКОЕ ИГРИСТОЕ ВИНО", D, "RUSSIAN SPARKLING WINE"),
        f("line", "МОЛОДОЕ", D, "YOUNG"),
        f("sugar", "СУХОЕ", D, "DRY"),
        f("color", "БЕЛОЕ", D, "WHITE"),
    ],
    "not_found_19-crimes-red-blend": [
        f("brand", "19 CRIMES", ID),
        f("vintage", "2017", D),
        f("wine_type", "RED WINE", D, "КРАСНОЕ ВИНО"),
    ],
    "not_found_fanagoriya-primum-alveus-blanc-de-blancs-bryut-2019": [
        f("producer", "FANAGORIA", ID, "ФАНАГОРИЯ"),
        f("product", "PRIMUM ALVEUS", ID, "ПРИМУМ АЛЬВЕУС"),
        f("wine_type", "BLANC DE BLANCS", D, "БЛАН ДЕ БЛАН"),
        f("sugar", "BRUT", D, "БРЮТ"),
        f("vintage", "2019", D),
    ],
    "not_found_jacobs-creek-reserve-shiraz-barossa-2013": [
        f("brand", "JACOB'S CREEK", ID, "JACOBS CREEK", "ДЖЕЙКОБС КРИК"),
        f("line", "RESERVE", D, "РЕЗЕРВ"),
        f("grape", "SHIRAZ", D, "ШИРАЗ", "SYRAH", "СИРА"),
        f("vintage", "2013", D),
        f("region", "BAROSSA", S, "БАРОССА"),
    ],
    "not_found_martini-asti": [
        f("brand", "MARTINI", ID, "МАРТИНИ"),
        f("product", "ASTI", ID, "АСТИ"),
        f("appellation", "D.O.C.G.", S, "DOCG"),
        f("abv", "7.5%", S, "7,5%"),
    ],
    "novyj-svet-aligote": [
        f("brand", "НОВЫЙ СВЕТ", ID, "NOVY SVET"),
        f("grape", "АЛИГОТЕ", D, "ALIGOTE"),
        f("sugar", "ЭКСТРА БРЮТ", D, "EXTRA BRUT"),
        f("vintage", "2021", D),
        f("method", "КЛАССИЧЕСКИЙ МЕТОД", S, "TRADITIONAL METHOD"),
    ],
    "shato-taman-kaberne-sovinon": [
        f("brand", "CHÂTEAU TAMAGNE", ID, "CHATEAU TAMAGNE", "ШАТО ТАМАНЬ"),
        f("grape", "CABERNET SAUVIGNON", D, "КАБЕРНЕ СОВИНЬОН"),
        f("wine_type", "СУХОЕ РОЗОВОЕ", D, "DRY ROSÉ", "DRY ROSE"),
    ],
}


IMAGE_FEATURES: dict[str, list[dict[str, Any]]] = {
    "data/test/derbent-vino-desono-merlo-krasnoe-suhoe-125/images.jpg": [
        f("vintage", "2022", D)
    ],
    "data/test/derbent-vino-desono-merlo-krasnoe-suhoe-125/images_2.jpg": [
        f("vintage", "2021", D)
    ],
    "data/test/derbent-vino-desono-merlo-krasnoe-suhoe-125/images_3.jpg": [
        f("vintage", "2022", D)
    ],
    "data/test/derbent-vino-desono-merlo-krasnoe-suhoe-125/images_4.jpg": [
        f("vintage", "2023", D)
    ],
    "data/test/derbent-vino-desono-merlo-krasnoe-suhoe-125/S.webp": [
        f("vintage", "2022", D)
    ],
    "data/test/derbent-vino-desono-risling-beloe-suhoe-125/images.jpg": [
        f("vintage", "2023", D)
    ],
}


def catalog_slugs() -> set[str]:
    """Read allowed slugs with a real CSV parser."""
    with CATALOG_PATH.open(encoding="utf-8-sig", newline="") as source:
        return {row["Slug"] for row in csv.DictReader(source) if row.get("Slug")}


def build_record(
    index: int, image_path: Path, allowed_slugs: set[str]
) -> dict[str, Any]:
    """Build one JSON-serializable annotation record."""
    relative_path = image_path.relative_to(ROOT).as_posix()
    folder = image_path.parent.name
    is_not_found = folder.startswith("not_found_")
    expected_slug = None if is_not_found else folder
    if expected_slug is not None and expected_slug not in allowed_slugs:
        raise ValueError(f"Test directory is not a Catalog slug: {folder}")
    if folder not in FEATURES_BY_FOLDER:
        raise ValueError(f"Missing manual feature annotation for: {folder}")

    with Image.open(image_path) as image:
        width, height = image.size
        image_format = image.format

    gold_features = [
        *FEATURES_BY_FOLDER[folder],
        *IMAGE_FEATURES.get(relative_path, []),
    ]
    return {
        "query_id": f"ocr-{index:06d}",
        "image_path": relative_path,
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "image_width": width,
        "image_height": height,
        "image_format": image_format,
        "expected_slug": expected_slug,
        "catalog_status": "not_found" if is_not_found else "found",
        "not_found_id": folder if is_not_found else None,
        "annotation_source": "manual_front_label_review",
        "gold_features": gold_features,
    }


def main() -> None:
    """Rebuild ``dataset.jsonl`` deterministically from the reviewed annotations."""
    images = sorted(
        (path for path in TEST_DIR.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    if not images:
        raise ValueError(f"No test images found under {TEST_DIR}")

    allowed_slugs = catalog_slugs()
    records = [
        build_record(index, path, allowed_slugs)
        for index, path in enumerate(images, start=1)
    ]
    annotated_folders = set(FEATURES_BY_FOLDER)
    actual_folders = {path.parent.name for path in images}
    if annotated_folders != actual_folders:
        missing = sorted(actual_folders - annotated_folders)
        stale = sorted(annotated_folders - actual_folders)
        raise ValueError(
            f"Annotation folders differ from data/test; missing={missing}, stale={stale}"
        )

    text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    OUTPUT_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {len(records)} records to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
