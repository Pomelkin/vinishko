"""Extract inert, offline visual references. Never executes archived scripts."""

import hashlib
import json
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup


root = Path(__file__).resolve().parents[2]
out = root / "frontend" / "references"
assets = out / "assets"
assets.mkdir(parents=True, exist_ok=True)
manifest = []
for source in root.glob("*.mhtml"):
    msg = BytesParser(policy=policy.default).parsebytes(source.read_bytes())
    mapping = {}
    parts = list(msg.walk())
    for part in parts:
        if part.is_multipart():
            continue
        loc = part.get("Content-Location", "")
        ct = part.get_content_type()
        data = part.get_payload(decode=True)
        if not data:
            continue
        ext = {
            "text/html": ".html",
            "text/css": ".css",
            "image/svg+xml": ".svg",
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "font/woff2": ".woff2",
            "application/font-woff": ".woff",
        }.get(ct, Path(loc.split("?")[0]).suffix or ".bin")
        name = hashlib.sha256(data).hexdigest()[:16] + ext
        mapping[loc] = "assets/" + name
        (assets / name).write_bytes(data)
        manifest.append(
            {"source": source.name, "url": loc, "type": ct, "path": "assets/" + name}
        )
    for part in parts:
        if part.get_content_type() not in ("text/html", "text/css"):
            continue
        data = part.get_payload(decode=True)
        if not data:
            continue
        content = data.decode(part.get_content_charset() or "utf-8", errors="replace")
        for url, local in sorted(mapping.items(), key=lambda x: -len(x[0])):
            if url:
                content = content.replace(
                    url,
                    local
                    if part.get_content_type() == "text/html"
                    else Path(local).name,
                )
        if part.get_content_type() == "text/html":
            soup = BeautifulSoup(content, "html.parser")
            for tag in soup.select("script, iframe, noscript, base"):
                tag.decompose()
            for tag in soup.find_all(True):
                for key in list(tag.attrs):
                    if key.startswith("on"):
                        del tag[key]
                if tag.name == "img":
                    tag.attrs.pop("srcset", None)
                    tag.attrs["loading"] = "eager"
            (out / (source.stem + ".html")).write_text(str(soup), encoding="utf-8")
        else:
            (out / mapping[part.get("Content-Location", "")]).write_text(
                content, encoding="utf-8"
            )
(out / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
pdf = pymupdf.open(next(root.glob("*.pdf")))
(out / "brief.txt").write_text(
    "\n".join(page.get_text() for page in pdf), encoding="utf-8"
)
pdf[0].get_pixmap(matrix=pymupdf.Matrix(1, 1)).save(str(out / "brief-page-1.png"))
print(f"Extracted {len(manifest)} resources and {len(pdf)} PDF pages")
