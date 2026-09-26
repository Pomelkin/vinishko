from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
RESULT_CSV = HERE / "sorted_photos_with_retrieval_siglip2_dense.csv"
CATALOG_IMAGES_DIR = HERE.parent / "data" / "images_without_garbarage"
HOST = "0.0.0.0"
PORT = 1338
IMAGE_COLUMN = "image"
BEFORE_VLM_COLUMN = "retrieval_results_before_vlm"
RESULT_COLUMN = "retrieval_results"

app = FastAPI(title="SigLIP2 dense retrieval viewer")

HTML = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SigLIP2 dense retrieval viewer</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif}body{margin:0;background:#111318;color:#eee}
header{position:sticky;top:0;z-index:2;display:flex;gap:12px;align-items:center;padding:14px 20px;background:#191c23}
button{padding:8px 14px;border:0;border-radius:7px;cursor:pointer}main{max-width:1400px;margin:auto;padding:20px}
.grid{display:grid;grid-template-columns:minmax(240px,1fr) 3fr;gap:24px}.candidates{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}
.card{padding:10px;border:3px solid #343944;border-radius:12px;background:#1b1f27}.selected{border-color:#35d07f}.truth{outline:3px solid #4da3ff;outline-offset:2px}
img{width:100%;height:55vh;object-fit:contain;background:#0b0c0f;border-radius:7px}.meta{margin-top:8px;overflow-wrap:anywhere;font-size:13px;color:#b9c0cc}
.badge{color:#35d07f;font-weight:700}.truth-badge{color:#4da3ff;font-weight:700}@media(max-width:800px){.grid{grid-template-columns:1fr}img{height:42vh}}
</style></head><body><header><button id="prev">←</button><button id="next">→</button><strong id="position"></strong><span id="slug"></span></header>
<main><div class="grid"><section><h2>Query</h2><div class="card"><img id="query" alt="Query photo"></div></section><section><h2>Dense candidates</h2><div id="candidates" class="candidates"></div></section></div></main>
<script>let rows=[],index=0;const show=()=>{const row=rows[index];position.textContent=`${index+1} / ${rows.length}`;slug.textContent=row.slug||'';query.src=row.query_url;
candidates.replaceChildren(...row.candidates.map((candidate,i)=>{const card=document.createElement('article');card.className=`card ${candidate.selected?'selected':''} ${candidate.ground_truth?'truth':''}`;
card.innerHTML=`<img alt="Candidate ${i+1}" src="${candidate.image_url}"><div class="meta"><b>#${i+1}</b> ${candidate.selected?'<span class="badge">selected ✓</span>':''} ${candidate.ground_truth?'<span class="truth-badge">GT ✓</span>':''}<br>${candidate.slug||''}<br>score: ${candidate.score?.toFixed(4)??'—'}</div>`;return card}))};
prev.onclick=()=>{index=(index-1+rows.length)%rows.length;show()};next.onclick=()=>{index=(index+1)%rows.length;show()};addEventListener('keydown',e=>{if(e.key==='ArrowLeft')prev.click();if(e.key==='ArrowRight')next.click()});
fetch('/api/results').then(r=>r.ok?r.json():Promise.reject(r)).then(data=>{rows=data;if(rows.length)show()}).catch(()=>{position.textContent='Не удалось загрузить результаты'});</script></body></html>"""


def load_results() -> list[dict[str, Any]]:
    with RESULT_CSV.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["candidates"] = json.loads(row[BEFORE_VLM_COLUMN])
        row["selected"] = json.loads(row[RESULT_COLUMN])
    return rows


def result_image(row_index: int, candidate_index: int | None = None) -> Path:
    try:
        row = load_results()[row_index]
        path = (
            PROJECT_DIR / row[IMAGE_COLUMN]
            if candidate_index is None
            else CATALOG_IMAGES_DIR / row["candidates"][candidate_index]["metadata"]["photo"]
        ).resolve()
    except (IndexError, KeyError, TypeError):
        raise HTTPException(status_code=404, detail="Image not found") from None
    allowed_root = PROJECT_DIR if candidate_index is None else CATALOG_IMAGES_DIR.resolve()
    if not path.is_relative_to(allowed_root) or not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return path


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/api/results")
def results() -> list[dict[str, Any]]:
    output = []
    for row_index, row in enumerate(load_results()):
        selected_ids = {str(item.get("point_id")) for item in row["selected"]}
        output.append({
            "slug": row.get("slug"),
            "query_url": f"/image/{row_index}/query",
            "candidates": [
                {
                    "image_url": f"/image/{row_index}/candidate/{candidate_index}",
                    "slug": candidate.get("metadata", {}).get("slug"),
                    "score": candidate.get("normalized_score"),
                    "selected": str(candidate.get("point_id")) in selected_ids,
                    "ground_truth": candidate.get("metadata", {}).get("slug") == row.get("slug"),
                }
                for candidate_index, candidate in enumerate(row["candidates"])
            ],
        })
    return output


@app.get("/image/{row_index}/query", response_class=FileResponse)
def query_image(row_index: int) -> Path:
    return result_image(row_index)


@app.get("/image/{row_index}/candidate/{candidate_index}", response_class=FileResponse)
def candidate_image(row_index: int, candidate_index: int) -> Path:
    return result_image(row_index, candidate_index)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
