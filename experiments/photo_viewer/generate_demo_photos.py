import random
from pathlib import Path

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

OUTPUT_DIR = Path(__file__).with_name("demo_photos")
IMAGE_COUNT = 24
SEED = 54


def generate() -> None:
    randomizer = random.Random(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    font = ImageFont.load_default(size=48)

    for index in range(IMAGE_COUNT):
        width, height = ((900, 600), (600, 900), (800, 800))[index % 3]
        image = Image.frombytes("RGB", (width, height), randomizer.randbytes(width * height * 3))
        draw = ImageDraw.Draw(image, "RGBA")
        color = tuple(randomizer.randrange(256) for _ in range(3))
        draw.rounded_rectangle((35, 35, width - 35, 140), radius=24, fill=(*color, 210))
        draw.text((65, 62), f"SYNTHETIC NOISE {index + 1:02}", font=font, fill="white", stroke_width=2,
                  stroke_fill="black")
        image.save(OUTPUT_DIR / f"noise_{index + 1:02}_{width}x{height}.jpg", quality=80)

    assert len(list(OUTPUT_DIR.glob("*.jpg"))) == IMAGE_COUNT


if __name__ == "__main__":
    generate()
