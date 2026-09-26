from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel
from transformers import AutoProcessor

MODEL_ID = "google/siglip2-so400m-patch14-384"
DEVICE = "cuda"
MODEL_DTYPE = torch.float16


class Siglip2Encoder:
    """Encode images into normalized dense SigLIP2 vectors on CUDA."""

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SigLIP2 требует доступную CUDA-видеокарту")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID, dtype=MODEL_DTYPE).to(DEVICE).eval()

    @torch.inference_mode()
    def encode(self, image_paths: list[Path]) -> list[list[float]]:
        if not image_paths:
            return []

        images = []
        for image_path in image_paths:
            if not image_path.is_file():
                raise FileNotFoundError(f"Не найдена фотография: {image_path}")
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {
            name: tensor.to(DEVICE, dtype=MODEL_DTYPE) if tensor.is_floating_point() else tensor.to(DEVICE)
            for name, tensor in inputs.items()
        }
        output = self.model.get_image_features(**inputs)
        features = output.pooler_output if hasattr(output, "pooler_output") else output
        return F.normalize(features, dim=-1).float().cpu().tolist()


def _self_check() -> None:
    vectors = F.normalize(torch.tensor([[3.0, 4.0]]), dim=-1)
    assert torch.allclose(vectors.norm(dim=-1), torch.ones(1))


if __name__ == "__main__":
    _self_check()
