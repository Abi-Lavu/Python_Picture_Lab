"""
Loading and saving images as plain numpy arrays.

A "picture" in this codebase is just a numpy array of shape (height, width, 3),
dtype uint8, holding red, green, blue values 0-255 for every pixel. There is no
Picture class wrapping it -- the array *is* the picture. That's deliberate: it's
the same representation you'll use later when images become model inputs.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file into an (H, W, 3) uint8 numpy array."""
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def save_image(pixels: np.ndarray, path: str | Path) -> None:
    """Save an (H, W, 3) uint8 numpy array to an image file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)


def blank_canvas(height: int, width: int, fill=(255, 255, 255)) -> np.ndarray:
    """Create a new blank (H, W, 3) uint8 array filled with one color."""
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:, :] = fill
    return canvas
