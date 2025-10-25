from PIL import Image
from pathlib import Path


figures_dir = Path("reports/figures")

for img_path in figures_dir.glob("curves_baseline_cat.png"):
    with Image.open(img_path) as img:
        width, height = img.size
        crop_box = (0, 0, width // 2, height)
        left_half = img.crop(crop_box)
        left_half.save(img_path)
