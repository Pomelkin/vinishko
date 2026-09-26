from io import BytesIO
from typing import Annotated

import torch
import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from PIL import Image
from PIL import UnidentifiedImageError
from transformers import NeoMMEForRetrieval
from transformers import NeoMMEProcessor

MODEL_NAME = "Hcompany/NeoMME-800M-Retriever"
HOST = "0.0.0.0"
PORT = 8001

processor = NeoMMEProcessor.from_pretrained(MODEL_NAME)
model = NeoMMEForRetrieval.from_pretrained(MODEL_NAME, device_map="cuda").eval()
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

        messages = [
            [{"role": "user", "content": [{"type": "image", "image": image}]}]
            for image in images
        ]
        inputs = processor.apply_chat_template(
            messages,
            task="document",
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": "longest"},
        ).to(model.device)

        with torch.inference_mode():
            embeddings = model(**inputs).embeddings.float().cpu()

        masks = inputs["attention_mask"].bool().cpu()
        return {
            "embeddings": [
                embedding[mask].tolist()
                for embedding, mask in zip(embeddings, masks, strict=True)
            ],
        }
    finally:
        for image in images:
            image.close()


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
