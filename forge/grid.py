"""Bảng tư thế: vẽ cả action trong MỘT ảnh SD, nhân vật mẫu nằm sẵn ở ô đầu.

Vì sao: vẽ từng frame riêng thì mỗi lần SD "nghĩ lại" chi tiết (hoa văn áo, màu dây
lưng, độ dài kiếm) -> frame nhấp nháy dù cùng seed + IP-Adapter. Trong một ảnh, các ô
nhìn thấy nhau qua self-attention nên SD vẽ chúng giống nhau. Chính vì thế checkpoint
pixel art hay tự vẽ "character sheet" 3 nhân vật giống hệt: ở đây biến xu hướng đó
thành lợi thế, mỗi ô một tư thế theo khung xương.

Ô 0 là nhân vật mẫu, được GIỮ NGUYÊN suốt quá trình khử nhiễu (SetLatentNoiseMask):
ở mỗi bước SD thấy nhân vật mẫu ngay bên cạnh và vẽ các ô còn lại theo đúng nó. Ô thừa
(lưới không lấp đầy) cũng giữ nguyên nền trắng, nên SD không có chỗ vẽ thêm người.
"""
from __future__ import annotations

import math

from PIL import Image

MAX_COLS = 4  # ảnh quá dẹt (5:1) SD1.5 vẽ kém; tối đa 4 ô một hàng


def shape(cells: int) -> tuple[int, int]:
    """(cột, hàng): ảnh ngang hoặc vuông, ít ô thừa nhất, rồi gần vuông nhất.
    VD 3 ô -> 3x1, 4 -> 2x2, 7 -> 4x2, 9 -> 3x3."""
    cells = max(1, cells)
    best = None
    for cols in range(1, min(MAX_COLS, cells) + 1):
        rows = math.ceil(cells / cols)
        if rows > cols and cols < min(MAX_COLS, cells):
            continue
        score = (cols * rows - cells, abs(cols - rows))
        if best is None or score < best[0]:
            best = (score, cols, rows)
    return best[1], best[2]


def chunks(n_frames: int, max_cells: int) -> list[list[int]]:
    """Chia frame thành các bảng, mỗi bảng <= max_cells ô (kể cả ô nhân vật mẫu), chia đều:
    10 frame / 9 ô -> 2 bảng x 5 frame (không phải 8 + 2)."""
    per = max(1, max_cells - 1)
    boards = max(1, math.ceil(n_frames / per))
    size = math.ceil(n_frames / boards)
    return [list(range(i, min(i + size, n_frames))) for i in range(0, n_frames, size)]


def box(index: int, cols: int, cell: int) -> tuple[int, int, int, int]:
    x, y = (index % cols) * cell, (index // cols) * cell
    return x, y, x + cell, y + cell


def compose(tiles: list[Image.Image | None], cols: int, rows: int, cell: int,
            fill: tuple[int, int, int]) -> Image.Image:
    """Ghép các ô (None = để trống màu `fill`) thành một ảnh RGB."""
    board = Image.new("RGB", (cols * cell, rows * cell), fill)
    for i, tile in enumerate(tiles):
        if tile is not None:
            board.paste(tile.convert("RGB").resize((cell, cell)), box(i, cols, cell)[:2])
    return board


def mask(active: list[int], cols: int, rows: int, cell: int) -> Image.Image:
    """Mặt nạ cho LoadImageMask (kênh red): trắng = ô SD được vẽ, đen = giữ nguyên."""
    m = Image.new("RGB", (cols * cell, rows * cell), (0, 0, 0))
    for i in active:
        m.paste((255, 255, 255), box(i, cols, cell))
    return m


def split(board: Image.Image, indices: list[int], cols: int, cell: int) -> list[Image.Image]:
    return [board.crop(box(i, cols, cell)) for i in indices]
