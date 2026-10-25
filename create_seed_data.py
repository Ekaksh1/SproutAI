"""
SPROUT AI - Seed Dataset Generator
Generates initial botanical training and evaluation samples with distinct
visual signatures (healthy uniform chlorophyll vs stressed chlorosis/necrosis/spotting).
Allows the user to immediately train and test the model out of the box.
"""

import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def generate_leaf_image(
    is_healthy: bool,
    seed: int,
    size: int = 384
) -> Image.Image:
    """
    Renders an organic leaf with realistic veins, coloring, and stress markers.
    """
    rng = np.random.default_rng(seed)

    # Base background (neutral indoor backdrop: soft warm grey/cream)
    bg_color = (
        rng.integers(220, 240),
        rng.integers(220, 235),
        rng.integers(215, 230)
    )
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2

    # Leaf shape points (oval / lanceolate with stem)
    length = rng.integers(int(size * 0.38), int(size * 0.45))
    width = rng.integers(int(size * 0.18), int(size * 0.26))
    angle = rng.uniform(-15, 15)  # slight tilt

    # Draw organic leaf contour
    num_pts = 60
    leaf_pts = []
    for i in range(num_pts):
        t = (i / num_pts) * 2 * math.pi
        # Parametric leaf curve
        x = width * math.sin(t) * (1 - 0.2 * math.cos(t))
        y = -length * math.cos(t) + (0.15 * length if math.cos(t) > 0 else 0)

        # Rotate
        rad = math.radians(angle)
        rx = x * math.cos(rad) - y * math.sin(rad) + cx
        ry = x * math.sin(rad) + y * math.cos(rad) + cy
        leaf_pts.append((rx, ry))

    # Base leaf coloration
    if is_healthy:
        # Lush vibrant forest / emerald greens
        leaf_color = (
            rng.integers(34, 55),
            rng.integers(130, 175),
            rng.integers(40, 75)
        )
    else:
        # Stressed: pale yellowish green or chlorotic olive
        leaf_color = (
            rng.integers(160, 205),
            rng.integers(160, 195),
            rng.integers(40, 70)
        )

    draw.polygon(leaf_pts, fill=leaf_color)

    # Draw central stem & primary midrib
    stem_color = (
        min(255, leaf_color[0] + 20),
        min(255, leaf_color[1] + 25),
        max(0, leaf_color[2] - 10)
    )
    rad = math.radians(angle)
    top_x = cx - math.sin(rad) * length * 0.95
    top_y = cy - math.cos(rad) * length * 0.95
    bot_x = cx + math.sin(rad) * length * 0.95
    bot_y = cy + math.cos(rad) * length * 0.95

    draw.line([(bot_x, bot_y), (top_x, top_y)], fill=stem_color, width=4)

    # Secondary lateral veins
    for step in range(1, 9):
        frac = step / 10.0
        vx = bot_x + (top_x - bot_x) * frac
        vy = bot_y + (top_y - bot_y) * frac
        span = width * 0.7 * (1.0 - abs(frac - 0.5) * 1.2)

        # Left vein
        lx = vx - math.cos(rad) * span - math.sin(rad) * 15
        ly = vy + math.sin(rad) * span - math.cos(rad) * 15
        draw.line([(vx, vy), (lx, ly)], fill=stem_color, width=2)

        # Right vein
        rx = vx + math.cos(rad) * span - math.sin(rad) * 15
        ry = vy - math.sin(rad) * span - math.cos(rad) * 15
        draw.line([(vx, vy), (rx, ry)], fill=stem_color, width=2)

    # If stressed, add realistic necrosis (brown patches), chlorosis, or dark spots
    if not is_healthy:
        stress_draw = ImageDraw.Draw(img)
        # Necrotic brown leaf margins
        for _ in range(rng.integers(3, 7)):
            spot_x = cx + rng.integers(-int(width * 0.9), int(width * 0.9))
            spot_y = cy + rng.integers(-int(length * 0.7), int(length * 0.7))
            spot_r = rng.integers(12, 28)
            brown = (
                rng.integers(110, 150),
                rng.integers(60, 90),
                rng.integers(20, 45)
            )
            stress_draw.ellipse(
                [spot_x - spot_r, spot_y - spot_r, spot_x + spot_r, spot_y + spot_r],
                fill=brown
            )

        # Fungal/pest leaf spotting
        for _ in range(rng.integers(6, 14)):
            px = cx + rng.integers(-int(width * 0.7), int(width * 0.7))
            py = cy + rng.integers(-int(length * 0.8), int(length * 0.8))
            pr = rng.integers(3, 7)
            dark_spot = (rng.integers(50, 75), rng.integers(30, 45), rng.integers(15, 25))
            stress_draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=dark_spot)

    # Slight blur to blend organic texture
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    return img


def create_all_seed_data():
    """Generates training and testing images for immediate demonstration."""
    dirs = [
        os.path.join("dataset", "healthy"),
        os.path.join("dataset", "stressed"),
        os.path.join("sample_test", "healthy"),
        os.path.join("sample_test", "stressed")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 1. Training set: 12 healthy, 12 stressed
    print("[*] Generating seed training dataset (12 healthy, 12 stressed)...")
    for i in range(12):
        img_h = generate_leaf_image(is_healthy=True, seed=100 + i)
        img_h.save(os.path.join("dataset", "healthy", f"healthy_leaf_{i+1:02d}.jpg"), quality=95)

        img_s = generate_leaf_image(is_healthy=False, seed=200 + i)
        img_s.save(os.path.join("dataset", "stressed", f"stressed_leaf_{i+1:02d}.jpg"), quality=95)

    # 2. Unseen Test set: 5 healthy, 5 stressed
    print("[*] Generating unseen evaluation test set (5 healthy, 5 stressed)...")
    for i in range(5):
        img_h = generate_leaf_image(is_healthy=True, seed=300 + i)
        img_h.save(os.path.join("sample_test", "healthy", f"test_healthy_{i+1:02d}.jpg"), quality=95)

        img_s = generate_leaf_image(is_healthy=False, seed=400 + i)
        img_s.save(os.path.join("sample_test", "stressed", f"test_stressed_{i+1:02d}.jpg"), quality=95)

    print("[*] Seed dataset generation complete!")

if __name__ == "__main__":
    create_all_seed_data()
