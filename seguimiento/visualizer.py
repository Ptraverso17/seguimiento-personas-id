"""Dibujo de cajas, IDs, trayectorias, línea de conteo y marcador."""

from __future__ import annotations

import colorsys
from collections import defaultdict, deque

import cv2
import numpy as np

from .tracker import TrackedPerson


def color_for_id(person_id: int) -> tuple[int, int, int]:
    """Color BGR estable y bien separado para cada ID (proporción áurea)."""
    hue = (person_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


class Visualizer:
    def __init__(self, trail_length: int = 30, show_trails: bool = True):
        self.show_trails = show_trails
        self.trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_length))

    def draw(
        self,
        frame: np.ndarray,
        people: list[TrackedPerson],
        line: tuple[tuple[float, float], tuple[float, float]] | None,
        stats: dict[str, int],
        flash_ids: set[int] | None = None,
    ) -> np.ndarray:
        out = frame.copy()
        scale = max(out.shape[1] / 1280, 0.5)
        thickness = max(1, int(2 * scale))
        flash_ids = flash_ids or set()

        if line is not None:
            self._draw_line(out, line, scale, thickness)

        for p in people:
            color = color_for_id(p.person_id)
            x1, y1, x2, y2 = map(int, p.box)
            box_thickness = thickness * 3 if p.person_id in flash_ids else thickness
            cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thickness)
            self._label(out, f"ID {p.person_id}", (x1, y1), color, scale)

            if self.show_trails:
                trail = self.trails[p.person_id]
                trail.append(tuple(map(int, p.feet)))
                if len(trail) > 1:
                    pts = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(out, [pts], False, color, thickness)

        self._panel(out, stats, scale)
        return out

    @staticmethod
    def _label(img, text, origin, color, scale):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.5 * scale
        (tw, th), base = cv2.getTextSize(text, font, fs, 1)
        x, y = origin
        y = max(y, th + base + 2)
        cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 6, y), color, -1)
        cv2.putText(img, text, (x + 3, y - base - 1), font, fs, (0, 0, 0), max(1, int(scale)), cv2.LINE_AA)

    @staticmethod
    def _draw_line(img, line, scale, thickness):
        (x1, y1), (x2, y2) = [(int(x), int(y)) for x, y in line]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), thickness + 1, cv2.LINE_AA)
        # Flecha en el centro que indica el sentido "entrada".
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        nx, ny = -dy / norm, dx / norm  # normal hacia el lado "positivo"
        length = 40 * scale
        tip = (int(mx + nx * length), int(my + ny * length))
        cv2.arrowedLine(img, (int(mx), int(my)), tip, (0, 255, 255), thickness, cv2.LINE_AA, tipLength=0.35)
        org = (tip[0] + 5, tip[1])
        fs = 0.5 * scale
        cv2.putText(img, "entrada", org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), max(1, int(scale)) + 2, cv2.LINE_AA)
        cv2.putText(img, "entrada", org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), max(1, int(scale)), cv2.LINE_AA)

    @staticmethod
    def _panel(img, stats, scale):
        lines = [f"{k}: {v}" for k, v in stats.items()]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.6 * scale
        lh = int(26 * scale)
        width = int(max(cv2.getTextSize(t, font, fs, 2)[0][0] for t in lines) + 20 * scale)
        height = lh * len(lines) + int(12 * scale)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
        for i, t in enumerate(lines):
            cv2.putText(img, t, (int(10 * scale), lh * (i + 1)), font, fs, (255, 255, 255),
                        max(1, int(2 * scale)), cv2.LINE_AA)
