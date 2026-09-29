import base64
import json
import shutil
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup


root = Path(__file__).resolve().parents[1]
assets = root / "references/assets"
dest = root / "public/assets"
dest.mkdir(parents=True, exist_ok=True)
names = {
    "bd89f77550e689ed.svg": "logo.svg",
    "fb510207085404e9.svg": "scanner.svg",
    "d3a063920cf26067.svg": "rating.svg",
    "c4d4e302fa241433.webp": "symbiosis.webp",
    "a2ffdc9c67c4d989.webp": "ai-red.webp",
    "de211d7103d82e30.webp": "ai-white.webp",
    "f1722489c06ec270.webp": "grenache.webp",
    "35c1f0fbf7c21d3d.webp": "saperavi.webp",
    "a6f099c00dbfe4a4.webp": "krasnostop.webp",
    "c3e0bed3249d5c22.webp": "region.webp",
    "bb25164b23f58d28.webp": "grapes.webp",
    "95a33c84765d5f5c.webp": "bbq.webp",
    "db809d50a54fef4e.webp": "vegetables.webp",
    "06d9ec5e4909dc52.webp": "steak.webp",
    "37d7c94e17c72b32.webp": "cheese.webp",
}
for source, target in names.items():
    shutil.copyfile(assets / source, dest / target)
soup = BeautifulSoup(
    (root / "references/Download.html").read_text(encoding="utf8"), "html.parser"
)
svg = soup.select_one("svg.wine-bottle-loader")
svg["viewBox"] = svg.attrs.pop("viewbox")
(dest / "loader.svg").write_text(str(svg), encoding="utf8")
font = dest / "playfair.woff2"
if not font.exists():
    urllib.request.urlretrieve(
        "https://vino-svoe.ru/fonts/Playfair/PlayfairDisplay-VariableFont_wght.woff2",
        font,
    )
# Code-native demo composition of the archived studio photographs. Silhouettes are
# manually traced in the original 540 x 540 coordinate system, not CV detections.
p1 = [
    (254, 65),
    (286, 65),
    (290, 71),
    (290, 173),
    (293, 184),
    (307, 199),
    (315, 218),
    (317, 233),
    (317, 447),
    (313, 461),
    (302, 469),
    (281, 474),
    (251, 474),
    (233, 469),
    (224, 462),
    (220, 450),
    (220, 233),
    (225, 210),
    (234, 195),
    (247, 181),
    (250, 171),
    (250, 75),
]
p2 = [
    (251, 65),
    (281, 65),
    (287, 70),
    (287, 165),
    (294, 199),
    (307, 237),
    (319, 269),
    (324, 290),
    (324, 453),
    (318, 466),
    (300, 474),
    (247, 474),
    (226, 468),
    (220, 455),
    (220, 289),
    (227, 260),
    (240, 213),
    (248, 176),
    (248, 78),
]


def generate(name, placements):
    shapes = []
    polys = []
    for i, (file, poly, x) in enumerate(placements):
        points = " ".join(f"{a},{b}" for a, b in poly)
        encoded = base64.b64encode((dest / file).read_bytes()).decode()
        shapes.append(
            f'<g transform="translate({x} 80) scale(1.5)"><defs><clipPath id="p{i}"><polygon points="{points}"/></clipPath></defs><image width="540" height="540" href="data:image/webp;base64,{encoded}" clip-path="url(#p{i})"/></g>'
        )
        polys.append(
            [
                [round((x + a * 1.5) / 900, 6), round((80 + b * 1.5) / 1000, 6)]
                for a, b in poly
            ]
        )
    text = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="1000" viewBox="0 0 900 1000"><rect width="900" height="1000" fill="#e9dfcf"/><path d="M0 710H900V1000H0Z" fill="#d9c8ad"/><path d="M0 710H900" stroke="#c5b59d" stroke-width="2"/>'
        + "".join(shapes)
        + "</svg>"
    )
    (dest / f"{name}.svg").write_text(text, encoding="utf8")
    return polys


polys = {
    "single": generate("demo-single", [("symbiosis.webp", p1, 45)]),
    "multiple": generate(
        "demo-multiple", [("symbiosis.webp", p1, -135), ("ai-white.webp", p2, 215)]
    ),
}
out = root / "src/mocks"
out.mkdir(parents=True, exist_ok=True)
(out / "polygons.json").write_text(json.dumps(polys), encoding="utf8")
print("Prepared local brand assets, font and traced demo compositions")
