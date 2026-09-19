from sentence_transformers import MultiVectorEncoder

MODEL_ID = "tencent/EVIE-4.5B"

model = MultiVectorEncoder(
    MODEL_ID,
)