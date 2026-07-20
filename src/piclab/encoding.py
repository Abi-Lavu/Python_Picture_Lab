"""Helpers for showing how a pixel's color is actually encoded as numbers."""


def to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02X}{g:02X}{b:02X}"


def to_binary(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"{r:08b} {g:08b} {b:08b}"


