"""
Pixel-level transforms, ported from the AP CSA Picture Lab.

These are written with explicit row/col loops on purpose, even though numpy
could do most of this in one vectorized line (see fast_transforms.py for the
vectorized versions). Looping over pixels here keeps the actual encoding
visible: every operation is "read three numbers, do some math, write three
numbers back."
"""

import numpy as np

def get_pixel(pixels: np.ndarray, row: int, col: int) -> tuple[int, int, int]:
    """Return the (r, g, b) values at a single pixel."""
    r, g, b = pixels[row, col]
    return int(r), int(g), int(b)


def set_pixel(pixels: np.ndarray, row: int, col: int, rgb: tuple[int, int, int]) -> None:
    """Set the (r, g, b) values at a single pixel, in place."""
    pixels[row, col] = rgb


def zero_blue(pixels: np.ndarray) -> np.ndarray:
    """Return a copy with the blue channel set to 0 everywhere."""
    result = pixels.copy()
    height, width, _ = result.shape
    for row in range(height):
        for col in range(width):
            r, g, _ = get_pixel(result, row, col)
            set_pixel(result, row, col, (r, g, 0))
    return result


def mirror_vertical(pixels: np.ndarray) -> np.ndarray:
    """Return a copy mirrored left-right around the vertical center line."""
    result = pixels.copy()
    height, width, _ = result.shape
    for row in range(height):
        for col in range(width // 2):
            left = get_pixel(result, row, col)
            set_pixel(result, row, width - 1 - col, left)
    return result


def mirror_region(
    pixels: np.ndarray,
    row_start: int,
    row_end: int,
    col_start: int,
    mirror_point: int,
) -> np.ndarray:
    """
    Mirror just a rectangular region of the image around a vertical line at
    mirror_point (a column index). Everything from col_start up to
    mirror_point gets copied onto its reflection to the right of mirror_point.
    """
    result = pixels.copy()
    for row in range(row_start, row_end):
        for col in range(col_start, mirror_point):
            left = get_pixel(result, row, col)
            target_col = mirror_point - col + mirror_point
            if target_col < result.shape[1]:
                set_pixel(result, row, target_col, left)
    return result


def copy_into(
    dest: np.ndarray,
    src: np.ndarray,
    start_row: int,
    start_col: int,
) -> np.ndarray:
    """Return a copy of dest with src pasted in at (start_row, start_col)."""
    result = dest.copy()
    dest_h, dest_w, _ = result.shape
    src_h, src_w, _ = src.shape
    for src_row, dest_row in zip(range(src_h), range(start_row, dest_h)):
        for src_col, dest_col in zip(range(src_w), range(start_col, dest_w)):
            set_pixel(result, dest_row, dest_col, get_pixel(src, src_row, src_col))
    return result


def color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """Euclidean distance between two (r, g, b) colors."""
    r1, g1, b1 = c1
    r2, g2, b2 = c2
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5


def edge_detection(pixels: np.ndarray, edge_dist: float) -> np.ndarray:
    """Return a black-and-white copy highlighting large color jumps left-to-right."""
    result = pixels.copy()
    height, width, _ = result.shape
    for row in range(height):
        for col in range(width - 1):
            left = get_pixel(result, row, col)
            right = get_pixel(result, row, col + 1)
            if color_distance(left, right) > edge_dist:
                set_pixel(result, row, col, (0, 0, 0))
            else:
                set_pixel(result, row, col, (255, 255, 255))
    return result
