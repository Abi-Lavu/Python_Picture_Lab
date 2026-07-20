import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from piclab import transforms, fast_transforms, encoding


def make_gradient(h=4, w=4):
    pixels = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            pixels[row, col] = (col * 10, row * 10, 100)
    return pixels


def test_zero_blue():
    pixels = make_gradient()
    result = transforms.zero_blue(pixels)
    assert (result[:, :, 2] == 0).all()
    assert (result[:, :, 0] == pixels[:, :, 0]).all()


def test_zero_blue_matches_fast():
    pixels = make_gradient(6, 6)
    assert (transforms.zero_blue(pixels) == fast_transforms.zero_blue(pixels)).all()


def test_mirror_vertical():
    pixels = make_gradient(2, 4)
    result = transforms.mirror_vertical(pixels)
    # left half is untouched; right half becomes a mirror of the left half
    assert tuple(result[0, 0]) == tuple(pixels[0, 0])
    assert tuple(result[0, 3]) == tuple(pixels[0, 0])
    assert tuple(result[0, 2]) == tuple(pixels[0, 1])


def test_mirror_vertical_matches_fast():
    pixels = make_gradient(5, 6)
    assert (transforms.mirror_vertical(pixels) == fast_transforms.mirror_vertical(pixels)).all()


def test_copy_into():
    dest = np.zeros((6, 6, 3), dtype=np.uint8)
    src = np.full((2, 2, 3), 200, dtype=np.uint8)
    result = transforms.copy_into(dest, src, 1, 1)
    assert tuple(result[1, 1]) == (200, 200, 200)
    assert tuple(result[0, 0]) == (0, 0, 0)


def test_color_distance_zero_for_same_color():
    assert transforms.color_distance((10, 20, 30), (10, 20, 30)) == 0


def test_encoding_roundtrip():
    assert encoding.to_hex((255, 0, 128)) == "#FF0080"
    assert encoding.to_binary((255, 0, 128)) == "11111111 00000000 10000000"
    assert encoding.to_packed_int((255, 0, 128)) == 0xFF0080
