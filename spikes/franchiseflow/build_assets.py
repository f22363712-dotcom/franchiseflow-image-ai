"""Throwaway signage assets for the tracking probe; no model-generated images."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent

def main():
    out = ROOT / "assets"
    out.mkdir(parents=True, exist_ok=True)
    variants = [
        ("riverside", "RIVERSIDE", "#e8e4d7", "#163e35"),
        ("downtown", "DOWNTOWN", "#dce8f1", "#173f60"),
        ("northpark", "NORTH PARK", "#f0dfd2", "#603727"),
    ]
    font_path = "C:/Windows/Fonts/arialbd.ttf"
    brand_font = ImageFont.truetype(font_path, 98)
    branch_font = ImageFont.truetype(font_path, 47)
    for name, branch, bg, fg in variants:
        im = Image.new("RGB", (1200, 300), bg)
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((20, 20, 1180, 280), radius=12, outline=fg, width=4)
        d.text((600, 105), "NOVA COFFEE", font=brand_font, fill=fg, anchor="mm")
        d.text((600, 221), branch, font=branch_font, fill=fg, anchor="mm")
        im.save(out / f"{name}.png")
    print(str(out))

if __name__ == "__main__":
    main()
