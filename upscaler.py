import sys
import os
from super_image import EdsrModel, ImageLoader
from PIL import Image


def main():
    if len(sys.argv) != 2:
        print("Usage: python upscaler.py <image>")
        sys.exit(1)

    img_path = sys.argv[1]

    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        sys.exit(1)

    name, ext = os.path.splitext(img_path)
    out_path = f"{name}_2x{ext}"
    model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=2)
    img = Image.open(img_path)
    inputs = ImageLoader.load_image(img)
    preds = model(inputs)
    ImageLoader.save_image(preds, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()