import numpy as np
from dataclasses import dataclass
from matplotlib.path import Path

@dataclass
class Vessel:
    kind: str = "Прямоугольник"
    rect: tuple = (-100.0, -100.0, 100.0, 100.0)
    circle: tuple = (0.0, 0.0, 100.0)
    poly: np.ndarray | None = None

    def contains(self, p: np.ndarray) -> bool:
        if self.kind in ("rect", "Прямоугольник"):
            xmin, ymin, xmax, ymax = self.rect
            return (xmin <= p[0] <= xmax) and (ymin <= p[1] <= ymax)
        if self.kind in ("circle", "Круг"):
            cx, cy, r = self.circle
            return (p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= r * r + 1e-12
        if self.kind in ("poly", "Многоугольник") and self.poly is not None and len(self.poly) >= 3:
            return bool(Path(self.poly, closed=True).contains_point((p[0], p[1])))
        return False

    def bounds(self, margin: float = 0.0):
        if self.kind in ("rect", "Прямоугольник"):
            xmin, ymin, xmax, ymax = self.rect
            return (xmin + margin, ymin + margin, xmax - margin, ymax - margin)
        elif self.kind in ("circle", "Круг"):
            cx, cy, R = self.circle
            return (cx - R + margin, cy - R + margin, cx + R - margin, cy + R - margin)
        elif self.kind in ("poly", "Многоугольник") and self.poly is not None and len(self.poly) > 0:
            xmin, ymin = self.poly.min(axis=0)
            xmax, ymax = self.poly.max(axis=0)
            return (xmin + margin, ymin + margin, xmax - margin, ymax - margin)
        else:
            return (-100 + margin, -100 + margin, 100 - margin, 100 - margin)

    @staticmethod
    def polygon_area(poly):
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
