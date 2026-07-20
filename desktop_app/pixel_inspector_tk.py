"""
Standalone pixel inspector, Tkinter version.

Run with: python desktop_app/pixel_inspector_tk.py [path/to/image.jpg]

Left panel: the full image. Drag a box on it to pick a region.
Right panel: that region zoomed way in, one square per pixel, with the
R/G/B values drawn as text on top of each square -- the same picture,
but showing you the actual numbers it's made of.
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageTk
import numpy as np

from piclab import io_utils, transforms, encoding

ZOOM_CELL = 40  # pixel size of each inspected cell, in screen pixels
MAX_REGION = 12  # cap the inspected region so the grid stays readable


class PixelInspectorApp:
    def __init__(self, root: tk.Tk, image_path: str | None):
        self.root = root
        self.root.title("Pixel Inspector")

        self.pixels: np.ndarray = (
            io_utils.load_image(image_path) if image_path else io_utils.blank_canvas(200, 300)
        )
        self.display_scale = 1.0
        self.drag_start = None
        self.region = None  # (row0, col0, row1, col1) in image coordinates

        self._build_layout()
        self._render_full_image()

    # ---------- layout ----------

    def _build_layout(self):
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)
        tk.Button(top, text="Open image...", command=self._open_image).pack(side="left")
        tk.Label(top, text="  Drag a box on the image to inspect that region").pack(side="left")

        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main)
        left.pack(side="left", padx=8, pady=8)
        self.canvas = tk.Canvas(left, bg="#222", cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        right = tk.Frame(main)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        tk.Label(right, text="Pixel grid (region zoomed in)", font=("Helvetica", 12, "bold")).pack(
            anchor="w"
        )
        self.grid_canvas = tk.Canvas(right, width=MAX_REGION * ZOOM_CELL, height=MAX_REGION * ZOOM_CELL, bg="white")
        self.grid_canvas.pack(anchor="w", pady=4)
        self.grid_canvas.bind("<Button-1>", self._on_grid_click)

        self.detail = tk.Label(right, text="Click a cell for its exact encoding", justify="left", font=("Courier", 11))
        self.detail.pack(anchor="w", pady=6)

    # ---------- image loading / display ----------

    def _open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self.pixels = io_utils.load_image(path)
        self.region = None
        self._render_full_image()
        self.grid_canvas.delete("all")

    def _render_full_image(self):
        h, w, _ = self.pixels.shape
        max_dim = 480
        self.display_scale = min(1.0, max_dim / max(h, w))
        disp_w, disp_h = int(w * self.display_scale), int(h * self.display_scale)
        img = Image.fromarray(self.pixels).resize((disp_w, disp_h), Image.NEAREST)
        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

    # ---------- region selection ----------

    def _on_press(self, event):
        self.drag_start = (event.x, event.y)

    def _on_drag(self, event):
        if not self.drag_start:
            return
        self.canvas.delete("selectbox")
        x0, y0 = self.drag_start
        self.canvas.create_rectangle(x0, y0, event.x, event.y, outline="yellow", width=2, tags="selectbox")

    def _on_release(self, event):
        if not self.drag_start:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        self.drag_start = None

        col0, col1 = sorted((int(x0 / self.display_scale), int(x1 / self.display_scale)))
        row0, row1 = sorted((int(y0 / self.display_scale), int(y1 / self.display_scale)))
        col1 = min(col1, col0 + MAX_REGION)
        row1 = min(row1, row0 + MAX_REGION)
        if col1 <= col0 or row1 <= row0:
            return
        self.region = (row0, col0, row1, col1)
        self._render_grid()

    # ---------- pixel grid ----------

    def _render_grid(self):
        self.grid_canvas.delete("all")
        row0, col0, row1, col1 = self.region
        for r in range(row0, row1):
            for c in range(col0, col1):
                rgb = transforms.get_pixel(self.pixels, r, c)
                x0 = (c - col0) * ZOOM_CELL
                y0 = (r - row0) * ZOOM_CELL
                fill = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                self.grid_canvas.create_rectangle(
                    x0, y0, x0 + ZOOM_CELL, y0 + ZOOM_CELL, fill=fill, outline="#888", tags=f"cell_{r}_{c}"
                )
                text_color = "white" if sum(rgb) < 380 else "black"
                self.grid_canvas.create_text(
                    x0 + ZOOM_CELL / 2,
                    y0 + ZOOM_CELL / 2,
                    text=f"{rgb[0]}\n{rgb[1]}\n{rgb[2]}",
                    fill=text_color,
                    font=("Courier", 7),
                )

    def _on_grid_click(self, event):
        if not self.region:
            return
        row0, col0, _, _ = self.region
        col = col0 + event.x // ZOOM_CELL
        row = row0 + event.y // ZOOM_CELL
        try:
            rgb = transforms.get_pixel(self.pixels, row, col)
        except IndexError:
            return
        self.detail.config(
            text=(
                f"row {row}, col {col}\n"
                f"RGB       = {rgb}\n"
                f"hex       = {encoding.to_hex(rgb)}\n"
                f"binary    = {encoding.to_binary(rgb)}\n"
                f"packed int = {encoding.to_packed_int(rgb)}"
            )
        )


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parents[1] / "images" / "beach.jpg"
    )
    root = tk.Tk()
    PixelInspectorApp(root, image_path)
    root.mainloop()


if __name__ == "__main__":
    main()
