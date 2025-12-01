import numpy as np
from dataclasses import dataclass
from matplotlib.path import Path
import config.constants

@dataclass
class Vessel:
    kind: str = "Прямоугольник"
    rect: tuple = None
    circle: tuple = None
    poly: np.ndarray | None = None

    def __post_init__(self):
        """Set defaults from constants if not provided."""
        if self.rect is None:
            self.rect = config.constants.VESSEL_RECT_BOUNDS
        if self.circle is None:
            self.circle = config.constants.VESSEL_CIRCLE_CENTER_AND_RADIUS

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
            return (config.constants.VESSEL_RECT_BOUNDS[0] + margin, 
                    config.constants.VESSEL_RECT_BOUNDS[1] + margin, 
                    config.constants.VESSEL_RECT_BOUNDS[2] - margin, 
                    config.constants.VESSEL_RECT_BOUNDS[3] - margin)

    @staticmethod
    def polygon_area(poly):
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
