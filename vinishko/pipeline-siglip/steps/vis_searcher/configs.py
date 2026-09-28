"""Параметры поиска; редактируются здесь, без CLI."""

from pathlib import Path

from vinishko.pipeline.steps.vis_searcher.configs import (
    LocalImagesConfig, QdrantConfig, TopNSearch, VisSearcherConfig,
)

ROOT = Path(__file__).resolve().parents[4]
MODEL = "pomelk1n/siglipus-twinturbo-gguf-umer"
REVISION = "f0ddd86548aa9af55bd55eb39f15fa97b06015be"
DEVICE = "auto"
BATCH_SIZE = 2
CPU_BATCH_SIZE = 1
TOP_K = 10
COSINE_THRESHOLD = -1.0
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
COLLECTION = "catalog_siglip2_dense_new"
TIMEOUT = 60.0
CATALOG_IMAGES = ROOT / "datasets/local/catalog/images"
CACHE_DIR = ROOT / "weights/siglip-cache"
DEBUG_PATH = None


def load_config() -> VisSearcherConfig:
    """Тот же контракт настроек поиска, с отдельными значениями для SigLIP."""
    return VisSearcherConfig(
        model=MODEL, revision=REVISION, device=DEVICE,
        batch_size=BATCH_SIZE, cpu_batch_size=CPU_BATCH_SIZE, top_k=TOP_K,
        qdrant=QdrantConfig(host=QDRANT_HOST, port=QDRANT_PORT, collection=COLLECTION, timeout=TIMEOUT),
        search=TopNSearch(mode="top_n", cosine_threshold=COSINE_THRESHOLD),
        reference_images=LocalImagesConfig(kind="local", dir=CATALOG_IMAGES),
        cache_dir=CACHE_DIR, debug_path=DEBUG_PATH,
    )
