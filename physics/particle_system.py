import numpy as np
import math
from dataclasses import dataclass, field
from typing import Optional
from matplotlib.path import Path

import config.constants
from .vessels import Vessel
from .potentials import PotentialParams
from utils.numba_functions import (
    compute_forces_numba, handle_particle_collisions,
    reflect_rectangular_numba, reflect_circular_numba, reflect_polygonal_numba
)

@dataclass
class System:
    vessel: Vessel
    N: int = config.constants.DEFAULT_N
    radius: float = config.constants.DEFAULT_RADIUS
    visual_radius: float = config.constants.DEFAULT_RADIUS
    temp: float = config.constants.DEFAULT_TEMP
    dt: float = config.constants.DEFAULT_DT
    params: PotentialParams = field(default_factory=PotentialParams)
    mass: float = config.constants.DEFAULT_MASS
    friction_gamma: float = 0.0
    enable_collisions: bool = True
    interaction_step: int = 1
    step: int = 0
    pos: Optional[np.ndarray] = None
    vel: Optional[np.ndarray] = None
    F: Optional[np.ndarray] = None
    F_old: Optional[np.ndarray] = None

    def n(self):
        return 0 if self.pos is None else self.pos.shape[0]

    def seed(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.pos = np.zeros((self.N, 2), dtype=np.float64)
        self.vel = np.zeros((self.N, 2), dtype=np.float64)
        self.F = np.zeros((self.N, 2), dtype=np.float64)
        self.F_old = np.zeros((self.N, 2), dtype=np.float64)
        if self.vessel.kind in ("poly", "Многоугольник") and self.vessel.poly is not None:
            self._seed_polygon_triangulation(rng)
        else:
            self._seed_standard(rng)
        sigma_v = math.sqrt(self.temp / max(self.mass, 1e-12))
        self.vel = rng.normal(0.0, sigma_v, size=(self.N, 2))

    def _seed_standard(self, rng):
        xmin, ymin, xmax, ymax = self.vessel.bounds(margin=self.radius)
        for i in range(self.N):
            for _ in range(1000):
                if self.vessel.kind in ("rect", "Прямоугольник"):
                    p = np.array([rng.uniform(xmin + self.radius, xmax - self.radius),
                                  rng.uniform(ymin + self.radius, ymax - self.radius)])
                elif self.vessel.kind in ("circle", "Круг"):
                    cx, cy, R = self.vessel.circle
                    ang = rng.uniform(0, 2 * np.pi)
                    rad = math.sqrt(rng.uniform(0, (R - self.radius) ** 2))
                    p = np.array([cx, cy]) + rad * np.array([math.cos(ang), math.sin(ang)])
                else:
                    p = np.array([0.0, 0.0])
                if self.vessel.contains(p):
                    if i == 0 or np.all(np.linalg.norm(self.pos[:i] - p, axis=1) >= 2 * self.radius):
                        self.pos[i] = p
                        break
            else:
                self._place_forced(i, rng)

    def _seed_polygon_triangulation(self, rng):
        poly = self.vessel.poly
        xmin, ymin, xmax, ymax = self.vessel.bounds(margin=self.radius)
        bbox_area = (xmax - xmin) * (ymax - ymin)
        poly_area = self._polygon_area(poly)
        fill_ratio = max(poly_area / max(bbox_area, 1e-9), 1e-3)
        grid_size = int(math.sqrt(self.N * 3 / fill_ratio)) + 1
        x_points = np.linspace(xmin + self.radius, xmax - self.radius, grid_size)
        y_points = np.linspace(ymin + self.radius, ymax - self.radius, grid_size)
        xx, yy = np.meshgrid(x_points, y_points)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        inside_mask = Path(poly).contains_points(grid_points)
        valid_points = grid_points[inside_mask]
        rng.shuffle(valid_points)
        placed = []
        for point in valid_points:
            if len(placed) >= self.N: break
            if len(placed) == 0:
                placed.append(point)
            else:
                if np.all(np.linalg.norm(np.array(placed) - point, axis=1) >= 2 * self.radius):
                    placed.append(point)
        if len(placed) < self.N:
            self._fill_remaining_particles(rng, placed, len(placed))
        else:
            self.pos = np.array(placed)

    def _polygon_area(self, poly):
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _place_forced(self, i, rng):
        """Place particle i when normal placement fails - used as fallback."""
        xmin, ymin, xmax, ymax = self.vessel.bounds(margin=self.radius)
        for _ in range(100):
            if self.vessel.kind in ("rect", "Прямоугольник"):
                p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
            elif self.vessel.kind in ("circle", "Круг"):
                cx, cy, R = self.vessel.circle
                ang = rng.uniform(0, 2 * np.pi)
                rad = math.sqrt(rng.uniform(0, (R - self.radius) ** 2))
                p = np.array([cx, cy]) + rad * np.array([math.cos(ang), math.sin(ang)])
            else:
                p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
            if self.vessel.contains(p):
                self.pos[i] = p
                return
        # Last resort: place at center
        if self.vessel.kind in ("circle", "Круг"):
            cx, cy, R = self.vessel.circle
            self.pos[i] = np.array([cx, cy])
        else:
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            self.pos[i] = np.array([cx, cy])

    def _fill_remaining_particles(self, rng, placed, start_idx):
        """Fill remaining particles when grid placement doesn't have enough."""
        xmin, ymin, xmax, ymax = self.vessel.bounds(margin=self.radius)
        for i in range(start_idx, self.N):
            for _ in range(1000):
                # Random point in bounding box
                p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
                if self.vessel.contains(p):
                    if np.all(np.linalg.norm(np.array(placed) - p, axis=1) >= 2 * self.radius):
                        placed.append(p)
                        break
            else:
                # Fallback: place anyway
                placed.append(np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)]))
        self.pos = np.array(placed[:self.N])

    def compute_forces(self):
        if self.pos is None:
            return np.zeros((0, 2))
        # Обновляем mapping для поддержки китайских названий
        kind_mapping = {
            "Нет": 0, "None": 0, "无": 0,
            "Отталкивание": 1, "Repulsion": 1, "排斥": 1,
            "Притяжение": 2, "Attraction": 2, "吸引": 2,
            "Леннард-Джонс": 3, "Lennard-Jones": 3, "伦纳德-琼斯": 3,
            "Морзе": 4, "Morse": 4, "莫尔斯": 4
        }

        kind_code = kind_mapping.get(self.params.kind, 0)
        return compute_forces_numba(self.pos, kind_code, self.params.k, self.params.epsilon,
                                    self.params.sigma, self.params.De, self.params.a, self.params.r0,
                                    self.params.rcut_lr, self.mass)

    def integrate(self):
        if self.pos is None:
            return

        self.step += 1

        if self.F_old is None or self.F_old.shape[0] != self.pos.shape[0]:
            self.F_old = np.zeros_like(self.pos)

        # ПРАВИЛЬНЫЙ Velocity Verlet алгоритм:
        # 1. Обновляем позиции используя текущие скорости и силы
        if self.F_old is not None:
            self.pos += self.vel * self.dt + 0.5 * self.F_old / self.mass * self.dt * self.dt
        else:
            self.pos += self.vel * self.dt

        # 2. Вычисляем новые силы
        do_interact = (self.step % max(1, self.interaction_step) == 0)
        if do_interact:
            F_new = self.compute_forces()
            # Ограничиваем силу для стабильности
            mags = np.linalg.norm(F_new, axis=1)
            if np.any(mags > 500.0):
                F_new *= (500.0 / np.max(mags))
        else:
            F_new = self.F_old if self.F_old is not None else np.zeros_like(self.vel)

        # 3. Обновляем скорости используя среднее старых и новых сил
        if self.F_old is not None:
            self.vel += 0.5 * (self.F_old + F_new) / self.mass * self.dt
        else:
            self.vel += 0.5 * F_new / self.mass * self.dt

        # Сохраняем силы для следующего шага
        self.F_old = F_new.copy() if F_new is not None else np.zeros_like(self.vel)

        # Обработка столкновений
        if self.enable_collisions and do_interact:
            handle_particle_collisions(self.pos, self.vel, self.radius, self.mass)

        # Граничные условия
        if self.vessel.kind in ("rect", "Прямоугольник"):
            xmin, ymin, xmax, ymax = self.vessel.rect
            reflect_rectangular_numba(self.pos, self.vel, self.radius, xmin, ymin, xmax, ymax)
        elif self.vessel.kind in ("circle", "Круг"):
            cx, cy, R = self.vessel.circle
            reflect_circular_numba(self.pos, self.vel, self.radius, cx, cy, R)
        elif self.vessel.kind in ("poly", "Многоугольник") and self.vessel.poly is not None:
            reflect_polygonal_numba(self.pos, self.vel, self.radius, self.vessel.poly)

        # Ограничение скорости для стабильности
        # max_speed = 50.0
        # speeds = np.linalg.norm(self.vel, axis=1)
        # too_fast = speeds > max_speed
        # if np.any(too_fast):
        #     self.vel[too_fast] *= (max_speed / speeds[too_fast])[:, None]

        # Трение
        if self.friction_gamma > 0:
            damp = math.exp(-self.friction_gamma * self.dt)
            self.vel *= damp

    def pairwise_distances_fast(self, max_pairs=10000):
        from utils.numba_functions import pairwise_distances_fast_numba
        if self.pos is None or self.n() < 2:
            return np.array([])
        return pairwise_distances_fast_numba(self.pos, max_pairs)
