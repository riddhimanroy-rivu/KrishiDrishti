"""
Generates branding assets for KrishiDrishti including logo.png and app iconography.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def generate_logo(output_path: Path):
    size = (512, 512)
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Background circle with subtle gradient or soft green fill
    draw.ellipse([20, 20, 492, 492], fill=(232, 245, 233, 255), outline=(46, 125, 50, 255), width=10)

    # Outer decorative AI ring / circuit nodes
    nodes = [
        (256, 45), (380, 80), (450, 180), (470, 290), (410, 400),
        (256, 470), (102, 400), (42, 290), (62, 180), (132, 80)
    ]
    for x, y in nodes:
        draw.ellipse([x - 12, y - 12, x + 12, y + 12], fill=(67, 160, 71, 255), outline=(27, 94, 32, 255), width=3)
        draw.line([(256, 256), (x, y)], fill=(165, 214, 167, 180), width=3)

    # Center leaf geometry
    leaf_poly = [
        (256, 90),
        (370, 180),
        (390, 290),
        (330, 390),
        (256, 430),
        (182, 390),
        (122, 290),
        (142, 180)
    ]
    draw.polygon(leaf_poly, fill=(46, 125, 50, 255))

    # Inner leaf half highlight
    half_leaf = [
        (256, 90),
        (370, 180),
        (390, 290),
        (330, 390),
        (256, 430)
    ]
    draw.polygon(half_leaf, fill=(56, 142, 60, 255))

    # Main leaf vein
    draw.line([(256, 100), (256, 420)], fill=(255, 255, 255, 230), width=6)

    # Lateral veins
    draw.line([(256, 170), (320, 140)], fill=(255, 255, 255, 200), width=4)
    draw.line([(256, 170), (192, 140)], fill=(255, 255, 255, 200), width=4)
    draw.line([(256, 240), (340, 210)], fill=(255, 255, 255, 200), width=4)
    draw.line([(256, 240), (172, 210)], fill=(255, 255, 255, 200), width=4)
    draw.line([(256, 310), (330, 300)], fill=(255, 255, 255, 200), width=4)
    draw.line([(256, 310), (182, 300)], fill=(255, 255, 255, 200), width=4)

    # Center AI pulse node
    draw.ellipse([240, 240, 272, 272], fill=(255, 235, 59, 255), outline=(245, 124, 0, 255), width=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    print(f"Generated logo at {output_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    generate_logo(base_dir / "logo.png")
