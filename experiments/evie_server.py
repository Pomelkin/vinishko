from io import BytesIO
from typing import Annotated

import torch
import uvicorn
from colpali_engine.models import ColQwen3_5
from colpali_engine.models import ColQwen3_5Processor
from colpali_engine.models.qwen3_5.colqwen3_5.modeling_colqwen3_5 import set_active_head
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from PIL import Image
from PIL import UnidentifiedImageError


MODEL_NAME = "tencent/EVIE-4.5B"
HEAD_DIM = 128
HOST = "0.0.0.0"
PORT = 8000

model = ColQwen3_5.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
).eval()
model.enable_bidirectional_attention()
set_active_head(model, HEAD_DIM)
processor = ColQwen3_5Processor.from_pretrained(MODEL_NAME)
app = FastAPI(openapi_url=None)


@app.post("/encode")
def encode(files: Annotated[list[UploadFile], File()]) -> dict[str, list[list[list[float]]]]:
    images = []

    try:
        for file in files:
            try:
                images.append(Image.open(BytesIO(file.file.read())).convert("RGB"))
            except (UnidentifiedImageError, OSError) as error:
                raise HTTPException(
                    status_code=400,
                    detail=f"{file.filename or 'Файл'} не является изображением",
                ) from error

        image_batch = processor.process_images(images).to(model.device)
        with torch.inference_mode():
            embeddings = model(**image_batch)
        return {"embeddings": embeddings.float().cpu().tolist()}
    finally:
        for image in images:
            image.close()


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
