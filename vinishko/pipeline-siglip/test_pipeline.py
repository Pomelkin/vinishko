"""python -m vinishko.pipeline-siglip.test_pipeline — без скачивания модели и сервера."""

import json
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

from vinishko.pipeline.structs import BottleCandidates, UnmatchedBottle

from .debug import box_input, select_images, summary
from .pipeline import Pipeline, raw_crop
from .steps.vis_searcher import search
from .steps.vis_searcher.configs import load_config
from .steps.vis_searcher.model import Preprocess


def main() -> None:
    with tempfile.TemporaryDirectory() as temp, closing(QdrantClient(":memory:")) as client:
        root = Path(temp)
        pixels = np.random.default_rng(17).integers(0, 256, (53, 27, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        image.save(root / "photo.png")
        box_file = root / "runs" / "photo" / "normalization" / "photo_b1_box.jpg"
        box_file.parent.mkdir(parents=True)
        image.save(box_file)
        assert box_input(root / "photo.png", root / "runs") == box_file
        assert box_input(root / "missing.png", root / "runs") is None
        raw = raw_crop(root / "photo.png")
        assert np.array_equal(raw.crop, pixels)
        assert raw.crop is raw.box_crop and raw.crop_info["mode"] == "raw"
        processed = Preprocess((12, 18), (0, 0, 0), "squash")(pixels)
        expected = np.asarray(image.resize((18, 12), Image.Resampling.BILINEAR)).transpose(2, 0, 1)
        assert processed.shape == (3, 12, 18) and processed.dtype == np.float32
        assert np.array_equal(processed, expected)
        try:
            Preprocess((12, 18), (0, 0, 0))(pixels.astype(float))
        except ValueError:
            pass
        else:
            raise AssertionError("невалидный вход принят")

        cfg = load_config()
        cfg.reference_images.dir = root
        cfg.top_k = 2
        (root / "config.json").write_text(json.dumps({"model_type": "siglip2_vision", "architecture": "google/siglip2-so400m-patch14-384"}))
        files = SimpleNamespace(repo=cfg.model, revision=cfg.revision, root=root, embed_dim=2, input_size=(384, 384))
        client.create_collection(cfg.qdrant.collection, vectors_config=VectorParams(size=2, distance=Distance.COSINE))
        client.upsert(cfg.qdrant.collection, points=[
            PointStruct(id=0, vector=[1.0, 0.0], payload={"slug": "first", "photo": "photo.png"}),
            PointStruct(id=1, vector=[0.0, 1.0], payload={"slug": "second", "photo": "photo.png"}),
        ])

        class EncoderStub:
            description = "offline"

            def __call__(self, images):
                assert all(np.array_equal(item, pixels) for item in images)
                return np.array([[1.0, 0.0] for _ in images], dtype=np.float32)

        with patch.object(search, "fetch_model", return_value=files), patch.object(search, "Encoder", return_value=EncoderStub()):
            searcher = search.VisSearcher(cfg, device=SimpleNamespace(precision="fp32"), client=client)
        pipeline = Pipeline(searcher)
        emitted = []
        results = pipeline.run_many([image, root / "photo.png"], on_result=lambda i, r: emitted.append((i, r)))
        assert len(results) == 2 and [i for i, _ in emitted] == [0, 1]
        assert pipeline.run_many([]) == [] and searcher([]) == []
        for result in results:
            answer = result.search[0]
            assert isinstance(answer, BottleCandidates)
            assert [c.slug for c in answer.candidates] == ["first", "second"]
            assert answer.candidates[0].score > 0.999
            assert all(c.retrieved and c.group == c.slug for c in answer.candidates)
            assert result.crops == [answer.crop]
            assert set(result.timings) == {"raw_input", "search"}
        searcher.dump(results[0].search, root / "dump")
        assert json.loads((root / "dump/results.json").read_text())["queries"][0]["candidates"][0]["slug"] == "first"
        assert summary(root / "photo.png", results[0])["bottles"][0]["status"] == "candidates"
        assert select_images(root / "photo.png", 1) == [root / "photo.png"]
        rejected = searcher._search(raw, [])
        assert isinstance(rejected, UnmatchedBottle) and rejected.rejected.uuid == raw.uuid
        cfg.search.cosine_threshold = 0.5
        below = searcher._search(raw, [ScoredPoint(id=0, version=0, score=0.1,
                                                  payload={"slug": "first", "photo": "photo.png"})])
        assert isinstance(below, UnmatchedBottle)
        cfg.search.cosine_threshold = -1.0
        no_match = Pipeline(lambda crops: [searcher._rejected(c, 0.1, 0.5, "позиции") for c in crops])(image)
        assert no_match.crops == [] and no_match.items[0].uuid == no_match.search[0].crop.uuid
        for payload in ({"slug": "x", "photo": "../escape.png"}, {"photo": "photo.png"}):
            try:
                searcher._validate_payload(payload)
            except ValueError:
                pass
            else:
                raise AssertionError("некорректный payload принят")
        for search_fn in (lambda crops: [], lambda crops: list(reversed(searcher(crops)))):
            try:
                Pipeline(search_fn).run_many([image, image])
            except ValueError:
                pass
            else:
                raise AssertionError("нарушенный контракт батча принят")
        try:
            searcher._check_model_metadata({"model": "different"})
        except ValueError:
            pass
        else:
            raise AssertionError("другая модель принята")
        assert not any(name.startswith(("vinishko.pipeline.steps.normalization", "vinishko.pipeline.steps.near_duplicates", "vinishko.pipeline.steps.vanilla_vlm_rerank")) for name in sys.modules)
    print("SigLIP pipeline checks passed")


if __name__ == "__main__":
    main()
