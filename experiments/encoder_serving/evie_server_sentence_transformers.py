from io import BytesIO
from typing import Annotated

import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from PIL import Image
from PIL import UnidentifiedImageError
from sentence_transformers import MultiVectorEncoder
from torch.nn.functional import normalize


# MODEL_NAME = "tencent/EVIE-4.5B"
MODEL_NAME = "tencent/EVIE-4.5B"
HEAD_DIM = 128
HOST = "0.0.0.0"
PORT = 8000

model = MultiVectorEncoder(MODEL_NAME, device="cuda")
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

        embeddings = model.encode_document(images)
        embeddings = [
            normalize(embedding[..., :HEAD_DIM], dim=-1)
            for embedding in embeddings
        ]
        return {"embeddings": [embedding.tolist() for embedding in embeddings]}
    finally:
        for image in images:
            image.close()


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
