"""Публичный контракт React-интерфейса поверх распознавания и каталога."""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from vinishko.pred.pipeline.pipeline import Pipeline
from vinishko.pred.pipeline.steps.vis_searcher.search import VisSearcher
from vinishko.pred.router import catalog_image, health, recognize
from vinishko.pred.ui import recognition_for_ui


router = APIRouter(prefix="/api", tags=["frontend"])
router.add_api_route("/health", health, methods=["GET"])
router.add_api_route("/catalog/images/{name}", catalog_image, methods=["GET"])


def catalog_images(pipe: Pipeline) -> dict[str, str]:
    """Имя картинки берётся из payload, расширение не угадывается по slug."""
    if not isinstance(pipe.searcher, VisSearcher):
        return {}
    images = {}
    offset = None
    while True:
        points, offset = pipe.searcher.client.scroll(
            pipe.searcher.collection,
            limit=1024,
            offset=offset,
            with_payload=["slug", "image"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            images[payload["slug"]] = payload["image"]
        if offset is None:
            return images


@router.post("/recognize")
async def recognize_ui(request: Request, image: Annotated[UploadFile, File()]) -> dict:
    result = await recognize(request, image)
    return recognition_for_ui(result, request.app.state.ui_catalog)


@router.get("/wines")
async def wines(
    request: Request, q: Annotated[str, Query(max_length=300)] = ""
) -> list[dict]:
    return request.app.state.ui_catalog.search(q)


@router.get("/wines/{slug}")
async def wine(request: Request, slug: str) -> dict:
    catalog = request.app.state.ui_catalog
    if slug not in catalog.rows:
        raise HTTPException(status_code=404, detail="Вино не найдено в каталоге")
    return catalog.details(slug)
