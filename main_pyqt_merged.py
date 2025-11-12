# main_pyqt.py
# GUI с переключателем языка (RU/EN/CN) по кнопке-флагу в левом нижнем углу главного меню.
# Требуются: PyQt6, numpy, matplotlib, numba, Pillow (PIL). Картинки флагов в images/.

import sys
import os
import time
import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from PIL import Image, ImageQt

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib import patches

from numba import njit, prange

import logging
import matplotlib

# ====================== Локализация ======================

class Lang(str, Enum):
    RU = "ru"
    EN = "en"
    CN = "cn"  # ← добавляем китайский


STRINGS = {
    "ru": {
        "app.title": "Распределение расстояний между частицами в сосудах различной формы",
        "menu.title": "Распределение расстояний между частицами",
        "menu.subtitle": "в сосудах различной формы",
        "menu.start": "Начать симуляцию",
        "menu.authors": "Об авторах",
        "menu.exit": "Выход",

        "vessel.rect": "Прямоугольник",
        "vessel.circle": "Круг",
        "vessel.poly": "Многоугольник",

        "sim.settings": "Настройки симуляции",
        "sim.N": "N",
        "sim.N.unit": "шт",
        "sim.N.help": "Число частиц",
        "sim.R": "R",
        "sim.R.unit": "ед",
        "sim.R.help": "Радиус частицы",
        "sim.T": "T",
        "sim.T.unit": "kT",
        "sim.T.help": "Температура",
        "sim.m": "m",
        "sim.m.unit": "ед",
        "sim.m.help": "Масса частицы",
        "sim.collisions": "Столкновения частиц",
        "sim.pot": "Pot",
        "sim.pot.help": "Тип парного потенциала",
        "sim.eps": "ε",
        "sim.eps.unit": "кДж/моль",
        "sim.eps.help": "Энергетический масштаб",
        "sim.sigma": "σ",
        "sim.sigma.unit": "ед",
        "sim.sigma.help": "Характерное расстояние LJ",
        "sim.De": "De",
        "sim.De.unit": "кДж/моль",
        "sim.De.help": "Глубина ямы Morse",
        "sim.a": "a",
        "sim.a.unit": "1/ед",
        "sim.a.help": "Жёсткость Morse",
        "sim.r0": "r₀",
        "sim.r0.unit": "ед",
        "sim.r0.help": "Равновесное расстояние (Morse)",
        "sim.vessel": "Сосуд",
        "sim.vessel.help": "Форма сосуда",
        "sim.interact": "Интеракции",
        "sim.interact.unit": "к-я итерация",
        "sim.bins": "Бины",
        "sim.bins.help": "Количество бинов для гистограмм",

        "sim.btn.draw": "Рисовать полигон",
        "sim.btn.clear": "Очистить полигон",
        "sim.btn.pause": "Пауза",
        "sim.btn.start": "Старт",
        "sim.btn.apply": "Применить",
        "sim.btn.back": "Назад",
        "sim.btn.collapse": "Свернуть",
        "sim.btn.expand": "Развернуть",
        "sim.top.settings": "≡ Настройки",
        "sim.anim.title": "Сосуд — область частиц",

        "sim.hist.x": "Распределение X",
        "sim.hist.y": "Распределение Y",
        "sim.hist.r": "Распределение расстояний",

        "menu.authors": "Об авторах",
        "authors.title": "Команда разработчиков",
        "authors.name1": "Енягин Станислав",
        "authors.name2": "Кожемякова Елизавета",
        "boss": "Научный руководитель : Чичигина Ольга Александровна",
        "authors.back": "Вернуться",
    },
    "en": {
        "app.title": "Distribution of Distances Between Particles in Vessels of Various Shapes",
        "menu.title": "Distribution of Particle Distances",
        "menu.subtitle": "in vessels of various shapes",
        "menu.start": "Start Simulation",
        "menu.authors": "Authors",
        "menu.exit": "Exit",

        "vessel.rect": "Rectangle",
        "vessel.circle": "Circle",
        "vessel.poly": "Polygon",

        "sim.settings": "Simulation settings",
        "sim.N": "N",
        "sim.N.unit": "pcs",
        "sim.N.help": "Number of particles",
        "sim.R": "R",
        "sim.R.unit": "u",
        "sim.R.help": "Particle radius",
        "sim.T": "T",
        "sim.T.unit": "kT",
        "sim.T.help": "Temperature",
        "sim.m": "m",
        "sim.m.unit": "u",
        "sim.m.help": "Particle mass",
        "sim.collisions": "Particle collisions",
        "sim.pot": "Pot",
        "sim.pot.help": "Pair potential type",
        "sim.eps": "ε",
        "sim.eps.unit": "kJ/mol",
        "sim.eps.help": "Energy scale",
        "sim.sigma": "σ",
        "sim.sigma.unit": "u",
        "sim.sigma.help": "Characteristic LJ distance",
        "sim.De": "De",
        "sim.De.unit": "kJ/mol",
        "sim.De.help": "Morse well depth",
        "sim.a": "a",
        "sim.a.unit": "1/u",
        "sim.a.help": "Morse stiffness",
        "sim.r0": "r₀",
        "sim.r0.unit": "u",
        "sim.r0.help": "Equilibrium distance (Morse)",
        "sim.vessel": "Vessel",
        "sim.vessel.help": "Vessel shape",
        "sim.interact": "Interactions",
        "sim.interact.unit": "th iter",
        "sim.bins": "Bins",
        "sim.bins.help": "Number of bins for histograms",

        "sim.btn.draw": "Draw polygon",
        "sim.btn.clear": "Clear polygon",
        "sim.btn.pause": "Pause",
        "sim.btn.start": "Start",
        "sim.btn.apply": "Apply",
        "sim.btn.back": "Back",
        "sim.btn.collapse": "Collapse",
        "sim.btn.expand": "Expand",
        "sim.top.settings": "≡ Settings",
        "sim.anim.title": "Vessel — particle domain",

        "sim.hist.x": "Distribution of X",
        "sim.hist.y": "Distribution of Y",
        "sim.hist.r": "Distribution of pair distances",

        "menu.authors": "Authors",
        "authors.title": "Development Team",
        "authors.name1": "Stanislav Eniagin",
        "authors.name2": "Elizaveta Kozhemyakova",
        "boss": "Scientific supervisor: Olga Aleksandrovna Chichigina",
        "authors.back": "Back",
    },
    "cn": {
        "app.title": "各种形状容器中粒子间距离分布",
        "menu.title": "粒子距离分布",
        "menu.subtitle": "在各种形状容器中",
        "menu.start": "开始模拟",
        "menu.authors": "关于作者",
        "menu.exit": "退出",

        "vessel.rect": "矩形",
        "vessel.circle": "圆形",
        "vessel.poly": "多边形",

        "sim.settings": "模拟设置",
        "sim.N": "N",
        "sim.N.unit": "个",
        "sim.N.help": "粒子数量",
        "sim.R": "R",
        "sim.R.unit": "单位",
        "sim.R.help": "粒子半径",
        "sim.T": "T",
        "sim.T.unit": "kT",
        "sim.T.help": "温度",
        "sim.m": "m",
        "sim.m.unit": "单位",
        "sim.m.help": "粒子质量",
        "sim.collisions": "粒子碰撞",
        "sim.pot": "势能",
        "sim.pot.help": "对势类型",
        "sim.eps": "ε",
        "sim.eps.unit": "千焦/摩尔",
        "sim.eps.help": "能量尺度",
        "sim.sigma": "σ",
        "boss": "奇奇吉娜·奥尔加·亚历山德罗夫娜",
        "sim.sigma.unit": "单位",
        "sim.sigma.help": "LJ特征距离",
        "sim.De": "De",
        "sim.De.unit": "千焦/摩尔",
        "sim.De.help": "Morse势阱深度",
        "sim.a": "a",
        "sim.a.unit": "1/单位",
        "sim.a.help": "Morse刚度",
        "sim.r0": "r₀",
        "sim.r0.unit": "单位",
        "sim.r0.help": "平衡距离 (Morse)",
        "sim.vessel": "容器",
        "sim.vessel.help": "容器形状",
        "sim.interact": "相互作用",
        "sim.interact.unit": "第 次迭代",
        "sim.bins": "分箱",
        "sim.bins.help": "直方图的分箱数量",

        "sim.btn.draw": "绘制多边形",
        "sim.btn.clear": "清除多边形",
        "sim.btn.pause": "暂停",
        "sim.btn.start": "开始",
        "sim.btn.apply": "应用",
        "sim.btn.back": "返回",
        "sim.btn.collapse": "折叠",
        "sim.btn.expand": "展开",
        "sim.top.settings": "≡ 设置",
        "sim.anim.title": "容器 - 粒子区域",

        "sim.hist.x": "X分布",
        "sim.hist.y": "Y分布",
        "sim.hist.r": "距离分布",

        "menu.authors": "关于作者",
        "authors.title": "开发团队",
        "authors.name1": "叶尼亚金·斯坦尼斯拉夫",
        "authors.name2": "科热米亚科娃·伊丽莎白",
        "authors.back": "返回",
    },
}


# ====================== Физика / Численные функции ======================

@dataclass
class PotentialParams:
    kind: str = "Нет"
    k: float = 100.0
    epsilon: float = 10.0
    sigma: float = 15.0 
    De: float = 20.0
    a: float = 0.5 # было 0.02
    r0: float = 2.5  # было 2.5
    rcut_lr: float = 80.0  # было 30.0


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
            from matplotlib.path import Path
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


@njit(fastmath=True, cache=True)
def compute_forces_numba(pos, params_kind, k, epsilon, sigma, De, a, r0, rcut_lr, mass):
    N = pos.shape[0]
    F = np.zeros((N, 2), dtype=np.float64)

    # Радиусы отсечения адаптированы под большие масштабы
    if params_kind == 3:  # Lennard-Jones
        rcut = 3.0 * sigma
    elif params_kind == 4:  # Morse
        rcut = r0 + 5.0 / a
    elif params_kind in (1, 2):
        rcut = rcut_lr
    else:
        rcut = 0.0

    rcut2 = rcut * rcut if rcut > 0 else 1e20
    rmin = max(0.1 * sigma, 0.001)

    for i in prange(N):
        xi, yi = pos[i]
        for j in range(i + 1, N):
            xj, yj = pos[j]
            dx = xj - xi
            dy = yj - yi
            r2 = dx * dx + dy * dy
            if r2 > rcut2:
                continue
            if r2 < 1e-14:
                r2 = 1e-14
            r = math.sqrt(r2)
            if r < rmin:
                r = rmin

            invr = 1.0 / r
            nx = dx * invr
            ny = dy * invr

            mag = 0.0

            if params_kind == 1:  # Отталкивание ~ k / r^2
                mag = k / r /r

            elif params_kind == 2:  # Притяжение ~ -k / r^2
                mag = -k / r / r

            elif params_kind == 3:  # Lennard-Jones (12-6)
                # F = 24ε(2(σ/r)^12 - (σ/r)^6) / r
                sr = sigma / r
                sr6 = sr**6
                sr12 = sr6 * sr6
                mag = 24.0 * epsilon * (2.0 * sr12 - sr6) / r

            elif params_kind == 4:  # Morse
                # F = 2Dea e^{-a(r - r0)}(1 - e^{-a(r - r0)})
                d = r - r0
                ea = math.exp(-a * d)
                mag = 2.0 * De * a * ea * (1.0 - ea)

            fx = mag * nx
            fy = mag * ny

            F[i, 0] -= fx
            F[i, 1] -= fy
            F[j, 0] += fx
            F[j, 1] += fy

    return F


@njit(fastmath=True, cache=True)
def handle_particle_collisions(pos, vel, radius, mass):
    N = pos.shape[0]
    restitution = 0.9
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            r2 = dx * dx + dy * dy
            min_dist = 2.0 * radius
            if r2 < (min_dist * min_dist):
                r = math.sqrt(r2)
                if r < 1e-8:
                    continue
                nx = dx / r
                ny = dy / r

                # позиционная коррекция (устраняем наложение)
                penetration = min_dist - r
                correction = 0.5 * penetration + 1e-3
                pos[i, 0] -= correction * nx
                pos[i, 1] -= correction * ny
                pos[j, 0] += correction * nx
                pos[j, 1] += correction * ny

                # импульсная коррекция скоростей
                dvx = vel[j, 0] - vel[i, 0]
                dvy = vel[j, 1] - vel[i, 1]
                vn = dvx * nx + dvy * ny
                if vn < 0:
                    j_imp = -(1.0 + restitution) * vn / (2.0 / mass)
                    jx = j_imp * nx
                    jy = j_imp * ny
                    vel[i, 0] -= jx / mass
                    vel[i, 1] -= jy / mass
                    vel[j, 0] += jx / mass
                    vel[j, 1] += jy / mass


               
@njit(fastmath=True, cache=True)
def pairwise_distances_fast_numba(pos, max_pairs=10000):
    N = pos.shape[0]
    if N < 2:
        return np.zeros(0, dtype=np.float64)
    total_pairs = N * (N - 1) // 2
    if total_pairs <= max_pairs:
        distances = np.zeros(total_pairs, dtype=np.float64)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                distances[idx] = math.sqrt(dx * dx + dy * dy)
                idx += 1
        return distances
    else:
        distances = np.zeros(max_pairs, dtype=np.float64)
        count = 0
        for _ in range(max_pairs * 2):
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            if i < j:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                distances[count] = math.sqrt(dx * dx + dy * dy)
                count += 1
                if count >= max_pairs:
                    break
        return distances[:count]


@njit(fastmath=True, cache=True)
def reflect_rectangular_numba(pos, vel, radius, xmin, ymin, xmax, ymax):
    N = pos.shape[0]
    for i in range(N):
        p = pos[i]
        v = vel[i]
        if p[0] < xmin + radius:
            v[0] = abs(v[0]) * 0.95
            p[0] = xmin + radius + 0.001
        if p[0] > xmax - radius:
            v[0] = -abs(v[0]) * 0.95
            p[0] = xmax - radius - 0.001
        if p[1] < ymin + radius:
            v[1] = abs(v[1]) * 0.95
            p[1] = ymin + radius + 0.001
        if p[1] > ymax - radius:
            v[1] = -abs(v[1]) * 0.95
            p[1] = ymax - radius - 0.001
        pos[i] = p
        vel[i] = v

@njit(fastmath=True, cache=True)
def reflect_circular_numba(pos, vel, radius, cx, cy, R):
    N = pos.shape[0]
    R_eff = R - radius
    for i in range(N):
        p = pos[i]
        v = vel[i]
        dx = p[0] - cx
        dy = p[1] - cy
        dist2 = dx * dx + dy * dy
        if dist2 > R_eff * R_eff:
            dist = math.sqrt(dist2)
            if dist > 0:
                nx = dx / dist
                ny = dy / dist
                # Корректируем позицию
                p[0] = cx + nx * R_eff
                p[1] = cy + ny * R_eff
                # Отражаем скорость
                dot = v[0] * nx + v[1] * ny
                v[0] -= 2.0 * dot * nx * 0.95
                v[1] -= 2.0 * dot * ny * 0.95      
        pos[i] = p
        vel[i] = v

@njit(fastmath=True, cache=True)
def reflect_polygonal_numba(pos, vel, radius, poly):
    N = pos.shape[0]
    M = poly.shape[0]
    for i in range(N):
        p = pos[i]
        v = vel[i]
        min_dist = 1e10
        closest = np.array([0.0, 0.0])
        normal = np.array([0.0, 0.0])
        
        for a in range(M):
            b = (a + 1) % M
            x1, y1 = poly[a]
            x2, y2 = poly[b]
            ex = x2 - x1
            ey = y2 - y1
            el2 = ex * ex + ey * ey
            if el2 < 1e-12: continue
            
            tox = p[0] - x1
            toy = p[1] - y1
            t = (tox * ex + toy * ey) / el2
            t = min(max(t, 0.0), 1.0)
            cx = x1 + t * ex
            cy = y1 + t * ey
            dx = p[0] - cx
            dy = p[1] - cy
            dist = math.sqrt(dx * dx + dy * dy)
            
            if dist < min_dist:
                min_dist = dist
                closest[0] = cx
                closest[1] = cy
                if dist > 1e-12:
                    normal[0] = dx / dist
                    normal[1] = dy / dist
                else:
                    L = math.sqrt(el2)
                    normal[0] = -ey / L
                    normal[1] = ex / L
        
        if min_dist < radius:
            # Корректируем позицию
            overlap = radius - min_dist
            p[0] += overlap * normal[0]
            p[1] += overlap * normal[1]
            
            # Отражаем скорость
            dot = v[0] * normal[0] + v[1] * normal[1]
            if dot < 0:  # Только если движется внутрь
                v[0] -= 2.0 * dot * normal[0] * 0.95
                v[1] -= 2.0 * dot * normal[1] * 0.95
                
        pos[i] = p
        vel[i] = v


@dataclass
class System:
    vessel: Vessel
    N: int = 10
    radius: float = 3.0
    visual_radius: float = 3.0
    temp: float = 5
    dt: float = 0.1
    params: PotentialParams = field(default_factory=PotentialParams)
    mass: float = 1.0
    friction_gamma: float = 0.0
    enable_collisions: bool = True
    interaction_step: int = 1
    step: int = 0
    pos: np.ndarray | None = None
    vel: np.ndarray | None = None
    F: np.ndarray | None = None
    F_old: np.ndarray | None = None

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
        from matplotlib.path import Path
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

    def _fill_remaining_particles(self, rng, placed_positions, success_count):
        xmin, ymin, xmax, ymax = self.vessel.bounds(margin=self.radius)
        for i in range(success_count, self.N):
            placed_array = np.array(placed_positions)
            for attempt in range(500):
                if attempt < 300:
                    p = np.array([rng.uniform(xmin + self.radius, xmax - self.radius),
                                  rng.uniform(ymin + self.radius, ymax - self.radius)])
                else:
                    if len(placed_positions) > 0:
                        base = placed_positions[rng.integers(0, len(placed_positions))]
                        angle = rng.uniform(0, 2 * np.pi)
                        dist = rng.uniform(2 * self.radius, 4 * self.radius)
                        p = base + dist * np.array([math.cos(angle), math.sin(angle)])
                    else:
                        continue
                if (self.vessel.contains(p) and
                        (len(placed_positions) == 0 or np.all(
                            np.linalg.norm(placed_array - p, axis=1) >= 2 * self.radius))):
                    placed_positions.append(p)
                    placed_array = np.array(placed_positions)
                    break
            else:
                self.pos[i] = np.array([0.0, 0.0])
        self.pos = np.array(placed_positions)

    def _polygon_area(self, poly):
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _place_forced(self, i, rng):
        if i == 0:
            if self.vessel.kind in ("rect", "Прямоугольник"):
                self.pos[i] = np.array([0.0, 0.0])
            elif self.vessel.kind in ("circle", "Круг"):
                self.pos[i] = np.array(self.vessel.circle[:2])
            elif self.vessel.kind in ("poly", "Многоугольник") and self.vessel.poly is not None:
                self.pos[i] = np.mean(self.vessel.poly, axis=0)
        else:
            self.pos[i] = np.array([0.0, 0.0])

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
        if self.pos is None or self.n() < 2:
            return np.array([])
        return pairwise_distances_fast_numba(self.pos, max_pairs)


# ====================== GUI ======================

class WinCheckBox(QtWidgets.QCheckBox):
    def mousePressEvent(self, e: QtGui.QMouseEvent):
        self.setChecked(not self.isChecked())
        self.update()
        e.accept()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        e.accept()


class MainMenuWidget(QtWidgets.QWidget):
    def __init__(self, parent, start_cb, lang_toggle_cb, get_lang_cb, authors_cb=None):
        super().__init__(parent)
        self.start_cb = start_cb
        self.authors_cb = authors_cb  # ← сохраняем колбэк (может быть None)
        self.lang_toggle_cb = lang_toggle_cb
        self.get_lang_cb = get_lang_cb
        self._build_ui()

    def _load_img_icon(self, filename, size=(32, 22)):
        try:
            path = os.path.join(os.path.dirname(__file__), "images", filename)
            img = Image.open(path).convert("RGBA")
            img.thumbnail(size, Image.Resampling.LANCZOS)
            return QtGui.QIcon(QtGui.QPixmap.fromImage(ImageQt.ImageQt(img)))
        except Exception:
            return None

    def _load_logo_pixmap(self, filename, size=(180, 180)):
        try:
            path = os.path.join(os.path.dirname(__file__), "images", filename)
            img = Image.open(path).convert("RGBA")
            data = np.array(img)
            mask = (data[:, :, 0] > 240) & (data[:, :, 1] > 240) & (data[:, :, 2] > 240)
            data[mask, 3] = 0
            img = Image.fromarray(data)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            return QtGui.QPixmap.fromImage(ImageQt.ImageQt(img))
        except Exception:
            return None

    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)

        left = QtWidgets.QVBoxLayout()
        center = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()

        layout.addLayout(left)
        layout.addLayout(center, stretch=1)
        layout.addLayout(right)

        # --- Левый столбец: логотип + флаг ---
        left_logo = QtWidgets.QLabel()
        pix = self._load_logo_pixmap("cmc_logo.png")
        if pix:
            left_logo.setPixmap(pix)
            left_logo.setFixedSize(180, 180)  # ФИКСИРУЕМ РАЗМЕР
            left_logo.setScaledContents(True)
        left.addWidget(
            left_logo,
            alignment=QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        left.addStretch(1)

        # Кнопка флага (левый нижний угол)
        self.btn_flag = QtWidgets.QPushButton()
        self.btn_flag.setFixedSize(44, 32)  # ФИКСИРУЕМ РАЗМЕР
        self.btn_flag.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.btn_flag.setStyleSheet("border:1px solid #ccc; border-radius:6px; background:#fff;")
        self.btn_flag.clicked.connect(self.lang_toggle_cb)
        left.addWidget(
            self.btn_flag,
            alignment=QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignBottom
        )

        # --- Правый столбец: логотип (если есть) ---

        right_logo = QtWidgets.QLabel()
        pix2 = self._load_logo_pixmap("fiz_logo.png") or self._load_logo_pixmap("fiz_logo.jpg")
        if pix2:
            right_logo.setPixmap(pix2)
            right_logo.setFixedSize(180, 180)  # ФИКСИРУЕМ РАЗМЕР
            right_logo.setScaledContents(True)
        right.addWidget(
            right_logo,
            alignment=QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop
        )

        right.addStretch(1)  # ← ДОБАВЬ ЭТУ СТРОЧКУ

        # --- Центр: заголовки и кнопки ---
        self.lbl_title = QtWidgets.QLabel()
        self.lbl_title.setStyleSheet("font-size:30pt; font-weight:600;")
        self.lbl_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # ВЫРАВНИВАНИЕ
        self.lbl_title.setWordWrap(True)  # ПЕРЕНАС СЛОВ

        self.lbl_subtitle = QtWidgets.QLabel()
        self.lbl_subtitle.setStyleSheet("font-size:20pt; color:#444")
        self.lbl_subtitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # ВЫРАВНИВАНИЕ
        self.lbl_subtitle.setWordWrap(True)  # ПЕРЕНАС СЛОВ

        center.addStretch(1)
        center.addWidget(self.lbl_title)
        center.addWidget(self.lbl_subtitle)

        # Кнопка Старт
        self.btn_start = QtWidgets.QPushButton()
        self.btn_start.setFixedSize(240, 50)  # ФИКСИРУЕМ РАЗМЕР
        self.btn_start.setStyleSheet(
            "background:#3271a8; color:white; font-weight:600; font-size:14pt; border-radius:6px; margin-top:10px;"
        )
        self.btn_start.clicked.connect(self.start_cb)
        center.addWidget(self.btn_start, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Кнопка Об авторах (опционально)
        if hasattr(self, "authors_cb") and self.authors_cb:
            self.btn_authors = QtWidgets.QPushButton()
            self.btn_authors.setFixedSize(240, 50)  # ФИКСИРУЕМ РАЗМЕР
            self.btn_authors.setStyleSheet(
                "background:#a6b2bd; color:white; font-weight:600; font-size:14pt; border-radius:6px; margin-top:10px;"
            )
            self.btn_authors.clicked.connect(self.authors_cb)
            center.addWidget(self.btn_authors, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Кнопка Выход
        self.btn_exit = QtWidgets.QPushButton()
        self.btn_exit.setFixedSize(240, 50)  # ФИКСИРУЕМ РАЗМЕР
        self.btn_exit.setStyleSheet(
            "background:#f78765; color:white; font-weight:600; font-size:14pt; border-radius:6px; margin-top:10px;"
        )
        self.btn_exit.clicked.connect(QtWidgets.QApplication.instance().quit)
        center.addWidget(self.btn_exit, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        center.addStretch(2)

        # Установим надписи и иконку флага согласно текущему языку
        self.update_language(self.get_lang_cb())

    def update_language(self, lang: Lang):
        s = STRINGS[lang.value]
        self.lbl_title.setText(s["menu.title"])
        self.lbl_subtitle.setText(s["menu.subtitle"])
        self.btn_start.setText(s["menu.start"])
        self.btn_exit.setText(s["menu.exit"])

        # Обновляем иконку флага для всех языков
        if lang == Lang.RU:
            icon_file = "rus.png"
        elif lang == Lang.EN:
            icon_file = "eng.png"
        elif lang == Lang.CN:
            icon_file = "china.png"
        else:
            icon_file = "rus.png"  # fallback

        icon = self._load_img_icon(icon_file)
        if icon:
            self.btn_flag.setIcon(icon)
            self.btn_flag.setIconSize(QtCore.QSize(28, 20))
        if hasattr(self, 'btn_authors') and self.btn_authors is not None:
            self.btn_authors.setText(s.get('menu.authors', 'Об авторах'))


class SimulationWidget(QtWidgets.QWidget):
    def __init__(self, parent, back_cb, get_lang_cb):
        super().__init__(parent)

        self.accumulated_x = np.array([])
        self.accumulated_y = np.array([])
        self.accumulated_distances = np.array([])
        self.max_accumulated_points = 100000  # Максимальное количество точек для предотвращения переполнения

        self.back_cb = back_cb
        self.get_lang_cb = get_lang_cb
        self.bins_count = 36  # Значение по умолчанию
        self._build_ui()
        self._init_simulation()
        self._start_timer()
        self.update_language(self.get_lang_cb())

    def _build_ui(self):
        main_l = QtWidgets.QHBoxLayout(self)
        self.settings_panel = QtWidgets.QWidget()
        self.settings_panel.setFixedWidth(300)  # ФИКСИРУЕМ ШИРИНУ
        self.settings_panel.setStyleSheet("background:#f7f7f8;")
        sp_layout = QtWidgets.QVBoxLayout(self.settings_panel)
        sp_layout.setContentsMargins(12, 12, 12, 12)

        self.lbl_settings_title = QtWidgets.QLabel()
        self.lbl_settings_title.setStyleSheet("font-size:14pt; font-weight:700;")
        sp_layout.addWidget(self.lbl_settings_title)
        sp_layout.addSpacing(6)

        def add_row(key_label, widget, key_unit="", key_help=""):
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel()  # текст поставим в update_language
            lbl.setObjectName(f"lbl_{key_label}")
            lbl.setStyleSheet("font-size:12pt;")
            lbl.setFixedWidth(100)  # ФИКСИРУЕМ ШИРИНУ
            if key_help:
                h = QtWidgets.QLabel()
                h.setObjectName(f"help_{key_label}")
                h.setWordWrap(True)
                h.setStyleSheet("color:#555;font-size:12pt;margin-top:2px;")
                sp_layout.addWidget(h)
            row.addWidget(lbl)
            row.addWidget(widget)
            widget.setStyleSheet("font-size:10pt;")
            if key_unit:
                u = QtWidgets.QLabel()
                u.setObjectName(f"unit_{key_label}")
                u.setFixedWidth(80)  # ФИКСИРУЕМ ШИРИНУ
                u.setStyleSheet("font-size:12pt;")
                row.addWidget(u)
            sp_layout.addLayout(row)

        self.spin_N = QtWidgets.QSpinBox()
        self.spin_N.setRange(5, 50)
        self.spin_N.setValue(10)
        self.spin_N.valueChanged.connect(self._on_settings_changed)  # ДИНАМИЧЕСКОЕ ИЗМЕНЕНИЕ
        add_row("N", self.spin_N, "sim.N.unit", "sim.N.help")

        self.edit_R = QtWidgets.QDoubleSpinBox()
        self.edit_R.setRange(0.5, 7.0)   # Изменить диапазон для нового масштаба
        self.edit_R.setSingleStep(0.1)
        self.edit_R.setDecimals(1)
        self.edit_R.setValue(3.0)
        self.edit_R.valueChanged.connect(self._on_settings_changed)  # ДИНАМИЧЕСКОЕ ИЗМЕНЕНИЕ
        add_row("R", self.edit_R, "sim.R.unit", "sim.R.help")

        self.edit_T = QtWidgets.QDoubleSpinBox()
        self.edit_T.setRange(1, 15)
        self.edit_T.setSingleStep(1)
        self.edit_T.setValue(5)
        self.edit_T.valueChanged.connect(self._on_settings_changed)  # ДИНАМИЧЕСКОЕ ИЗМЕНЕНИЕ
        add_row("T", self.edit_T, "sim.T.unit", "sim.T.help")

        self.edit_m = QtWidgets.QDoubleSpinBox()
        self.edit_m.setRange(0.01, 100.0)
        self.edit_m.setSingleStep(0.1)
        self.edit_m.setValue(1.0)
        self.edit_m.valueChanged.connect(self._on_settings_changed)  # ДИНАМИЧЕСКОЕ ИЗМЕНЕНИЕ
        add_row("m", self.edit_m, "sim.m.unit", "sim.m.help")

        self.check_collisions = WinCheckBox()
        self.check_collisions.stateChanged.connect(self._on_settings_changed)  # ДИНАМИЧЕСКОЕ ИЗМЕНЕНИЕ
        self.check_collisions.setStyleSheet("font-size:12pt;")
        self.check_collisions.setChecked(True)
        sp_layout.addWidget(self.check_collisions)

        self.pot_box = QtWidgets.QComboBox()
        self.pot_box.currentIndexChanged.connect(self._on_settings_changed)  # ДИНАМИЧЕСКОЕ ИЗМЕНЕНИЕ
        add_row("pot", self.pot_box, "", "sim.pot.help")

        self.edit_eps = QtWidgets.QDoubleSpinBox()
        self.edit_eps.setRange(0.0, 50.0)
        self.edit_eps.setValue(1.0)
        self.edit_eps.valueChanged.connect(self._on_settings_changed)  # ДИНАМИЧЕСКОЕ ИЗМЕНЕНИЕ
        add_row("eps", self.edit_eps, "sim.eps.unit", "sim.eps.help")

        self.edit_sigma = QtWidgets.QDoubleSpinBox()
        self.edit_sigma.setRange(2.0, 20.0)  # Изменить диапазон
        self.edit_sigma.setValue(15.0)
        self.edit_sigma.valueChanged.connect(self._on_settings_changed)  # ДИНАМИЧЕСКОЕ ИЗМЕНЕНИЕ
        add_row("sigma", self.edit_sigma, "sim.sigma.unit", "sim.sigma.help")

        self.edit_De = QtWidgets.QDoubleSpinBox()
        self.edit_De.setRange(0.0, 50.0)
        self.edit_De.setValue(1.0)
        self.edit_De.valueChanged.connect(self._on_settings_changed)  # ДИНАМИЧЕСКОЕ ИЗМЕНЕНИЕ
        add_row("De", self.edit_De, "sim.De.unit", "sim.De.help")

        self.edit_a = QtWidgets.QDoubleSpinBox()
        self.edit_a.setRange(0.05, 2.0)  # было (0.001, 0.5)
        self.edit_a.setValue(0.3)
        self.edit_a.valueChanged.connect(self._on_settings_changed)  # ДИНАМИЧЕСКОЕ ИЗМЕНЕНИЕ
        add_row("a", self.edit_a, "sim.a.unit", "sim.a.help")

        self.edit_r0 = QtWidgets.QDoubleSpinBox()
        self.edit_r0.setRange(1, 10.0)  # было (0.1, 50.0)
        self.edit_r0.setValue(2.5) 
        self.edit_r0.valueChanged.connect(self._on_settings_changed)  # ДИНАМИЧЕСКОЕ ИЗМЕНЕНИЕ
        add_row("r0", self.edit_r0, "sim.r0.unit", "sim.r0.help")

        self.vessel_box = QtWidgets.QComboBox()
        self.vessel_box.addItems(["Прямоугольник", "Круг", "Многоугольник"])
        self.vessel_box.currentIndexChanged.connect(self._on_vessel_changed)  # ОСОБАЯ ОБРАБОТКА
        self.vessel_box.setStyleSheet("font-size:12pt;color: black;")
        add_row("vessel", self.vessel_box, "", "sim.vessel.help")

        self.spin_interact_step = QtWidgets.QSpinBox()
        self.spin_interact_step.setRange(1, 1000)
        self.spin_interact_step.setValue(1)
        self.spin_interact_step.valueChanged.connect(self._on_settings_changed)  # ДИНАМИЧЕСКОЕ ИЗМЕНЕНИЕ
        #add_row("interact", self.spin_interact_step, "sim.interact.unit", "")

        # Добавляем спинбокс для количества бинов
        self.spin_bins = QtWidgets.QSpinBox()
        self.spin_bins.setRange(5, 100)
        self.spin_bins.setValue(self.bins_count)
        self.spin_bins.valueChanged.connect(self._on_bins_changed)
        add_row("bins", self.spin_bins, "", "sim.bins.help")

        sp_layout.addSpacing(8)
        self.btn_draw = QtWidgets.QPushButton()
        self.btn_draw.clicked.connect(self._enter_draw_mode)
        self.btn_draw.setFixedHeight(25)  # ФИКСИРУЕМ ВЫСОТУ
        self.btn_draw.setStyleSheet("font-size:11pt;")
        sp_layout.addWidget(self.btn_draw)

        self.btn_clear = QtWidgets.QPushButton()
        self.btn_clear.clicked.connect(self._clear_poly)
        self.btn_clear.setFixedHeight(25)  # ФИКСИРУЕМ ВЫСОТУ
        self.btn_clear.setStyleSheet("font-size:11pt;")
        sp_layout.addWidget(self.btn_clear)

        self.btn_run = QtWidgets.QPushButton()
        self.btn_run.clicked.connect(self._toggle_run)
        self.btn_run.setFixedHeight(25)  # ФИКСИРУЕМ ВЫСОТУ
        self.btn_run.setStyleSheet("font-size:11pt;")
        sp_layout.addWidget(self.btn_run)

        # self.btn_apply = QtWidgets.QPushButton()
          
        # self.btn_apply.setFixedHeight(25)  # ФИКСИРУЕМ ВЫСОТУ
        # self.btn_apply.setStyleSheet("font-size:11pt;")
        # sp_layout.addWidget(self.btn_apply)

        self.btn_back = QtWidgets.QPushButton()
        self.btn_back.clicked.connect(self.back_cb)
        self.btn_back.setFixedHeight(25)  # ФИКСИРУЕМ ВЫСОТУ
        self.btn_back.setStyleSheet("font-size:11pt;")
        sp_layout.addStretch(1)
        sp_layout.addWidget(self.btn_back)

        # self.btn_collapse = QtWidgets.QPushButton()
        # self.btn_collapse.clicked.connect(self._toggle_settings)
        # self.btn_collapse.setFixedHeight(25)  # ФИКСИРУЕМ ВЫСОТУ
        # self.btn_collapse.setStyleSheet("font-size:11pt;")
        # sp_layout.addWidget(self.btn_collapse)

        canvas_container = QtWidgets.QHBoxLayout()
        main_l.addWidget(self.settings_panel)
        main_l.addLayout(canvas_container, stretch=1)

        anim_container = QtWidgets.QVBoxLayout()
        canvas_container.addLayout(anim_container, stretch=2)

        hist_container = QtWidgets.QVBoxLayout()
        canvas_container.addLayout(hist_container, stretch=1)

        top_bar = QtWidgets.QHBoxLayout()
        self.btn_toggle_settings_small = QtWidgets.QPushButton()
        self.btn_toggle_settings_small.setFixedSize(110, 28)  # ФИКСИРУЕМ РАЗМЕР
        self.btn_toggle_settings_small.setStyleSheet("font-size:10pt;")
        self.btn_toggle_settings_small.clicked.connect(self._toggle_settings)
        top_bar.addWidget(self.btn_toggle_settings_small, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        top_bar.addStretch(1)
        anim_container.addLayout(top_bar)

        self.fig_anim = plt.Figure(figsize=(6, 6))
        self.ax_anim = self.fig_anim.add_subplot(111)
        self.ax_anim.set_xticks([])
        self.ax_anim.set_yticks([])
        self.ax_anim.set_aspect('equal')
        self.canvas_anim = FigureCanvas(self.fig_anim)
        self.canvas_anim.setMinimumSize(600, 600)  # ФИКСИРУЕМ МИНИМАЛЬНЫЙ РАЗМЕР
        anim_container.addWidget(self.canvas_anim)
        self.canvas_anim.mpl_connect("button_press_event", self._on_mouse)

        self.fig_hist, axes = plt.subplots(3, 1, figsize=(10, 10))
        for ax in axes:
            ax.tick_params(labelsize=9)
        self.ax_histx, self.ax_histy, self.ax_histd = axes
        self.fig_hist.tight_layout()
        self.canvas_hist = FigureCanvas(self.fig_hist)
        self.canvas_hist.setMinimumSize(400, 600)  # ФИКСИРУЕМ МИНИМАЛЬНЫЙ РАЗМЕР
        hist_container.addWidget(self.canvas_hist)

        self.draw_mode = False
        self.poly_points = []
        
        # Изначально обновляем состояние кнопок полигона
        self._update_polygon_buttons_state()

    def _on_bins_changed(self, value):
        self.bins_count = value
        self._update_histograms()

    def _on_settings_changed(self):
        # Динамическое применение настроек без перезапуска симуляции
        if hasattr(self, 'system') and self.system is not None:
            # Обновляем параметры системы
            self.system.N = int(self.spin_N.value())
            self.system.radius = float(self.edit_R.value())
            self.system.visual_radius = float(self.edit_R.value())
            self.system.temp = float(self.edit_T.value())
            self.system.mass = float(self.edit_m.value())
            self.system.enable_collisions = self.check_collisions.isChecked()
            self.system.interaction_step = int(self.spin_interact_step.value())
            
            # Обновляем параметры потенциала
            self.system.params.kind = str(self.pot_box.currentText())
            self.system.params.epsilon = float(self.edit_eps.value())
            self.system.params.sigma = float(self.edit_sigma.value())
            self.system.params.De = float(self.edit_De.value())
            self.system.params.a = float(self.edit_a.value())
            self.system.params.r0 = float(self.edit_r0.value())
            
            # Если изменилось количество частиц, пересоздаем систему
            # if self.system.n() != self.system.N:
            self._reinitialize_system()
            # else:
                # Иначе просто обновляем визуализацию
            # self._update_particle_visualization()

    def _on_vessel_changed(self):
        # Особая обработка изменения типа сосуда
        vessel_kind = str(self.vessel_box.currentText())
        
        # Обновляем состояние кнопок полигона
        self._update_polygon_buttons_state()
        
        # Если выбран не полигон, выходим из режима рисования
        if vessel_kind not in ("poly", "Многоугольник"):
            self.draw_mode = False
            self.poly_points = []
        
        # Применяем изменение сосуда
        self._on_settings_changed()

    def _update_polygon_buttons_state(self):
        """Обновляет состояние кнопок рисования полигона в зависимости от выбранного типа сосуда"""
        vessel_kind = str(self.vessel_box.currentText())
        is_polygon_mode = vessel_kind in ("poly", "Многоугольник")
        
        self.btn_draw.setEnabled(is_polygon_mode)
        self.btn_clear.setEnabled(is_polygon_mode)
        
        # Визуальная индикация отключенного состояния
        if not is_polygon_mode:
            self.btn_draw.setStyleSheet("font-size:11pt; background-color: #cccccc; color: #666666;")
            self.btn_clear.setStyleSheet("font-size:11pt; background-color: #cccccc; color: #666666;")
        else:
            self.btn_draw.setStyleSheet("font-size:11pt;")
            self.btn_clear.setStyleSheet("font-size:11pt;")

    def _reinitialize_system(self):
        """Полная переинициализация системы с текущими настройками"""
        N = int(self.spin_N.value())
        radius = float(self.edit_R.value()) 
        temp = float(self.edit_T.value())
        dt = 0.1
        mass = float(self.edit_m.value())
        enable_collisions = self.check_collisions.isChecked()
        interaction_step = int(self.spin_interact_step.value())
        pot_params = PotentialParams(
            kind=str(self.pot_box.currentText()),
            epsilon=float(self.edit_eps.value()),
            sigma=float(self.edit_sigma.value()),
            De=float(self.edit_De.value()),
            a=float(self.edit_a.value()),
            r0=float(self.edit_r0.value()),
        )
        vessel_kind = str(self.vessel_box.currentText())
        poly = self.system.vessel.poly if vessel_kind in ("poly", "Многоугольник") else None

        # Сбрасываем накопленные данные при изменении параметров
        self.accumulated_x = np.array([])
        self.accumulated_y = np.array([])
        self.accumulated_distances = np.array([])

        self._init_simulation(N=N, radius=radius, temp=temp, dt=dt, vessel_kind=vessel_kind, poly=poly,
                              potential_params=pot_params, mass=mass, enable_collisions=enable_collisions,
                              interaction_step=interaction_step)

    def _update_particle_visualization(self):
        """Обновляет визуализацию частиц без пересоздания системы"""
        if hasattr(self, 'particle_circles'):
            for c in self.particle_circles: 
                c.remove()
        
        self.particle_circles = []
        if self.system.pos is not None:
            for i in range(self.system.n()):
                circle = patches.Circle((self.system.pos[i, 0], self.system.pos[i, 1]),
                                        radius=self.system.visual_radius, facecolor="#215a93",
                                        edgecolor="black", linewidth=0.5, alpha=0.8)
                self.ax_anim.add_patch(circle)
                self.particle_circles.append(circle)
        
        self.canvas_anim.draw_idle()

    def _init_simulation(self, N=10, radius=3, temp=5, dt=0.1,
                         vessel_kind="Прямоугольник", poly=None,
                         potential_params=None, mass=1.0, enable_collisions=True, interaction_step=1):
        if potential_params is None:
            potential_params = PotentialParams()
        if vessel_kind in ("rect", "Прямоугольник"):
            vessel = Vessel(kind="Прямоугольник", rect=(-100, -100, 100, 100))
        elif vessel_kind in ("circle", "Круг"):
            vessel = Vessel(kind="Круг", circle=(0, 0, 100))
        elif vessel_kind in ("poly", "Многоугольник"):
            if poly is None:
                poly = np.array([[-80, -80], [80, -80], [0.0, 80]])
            vessel = Vessel(kind="Многоугольник", poly=poly)
        else:
            vessel = Vessel(kind="Прямоугольник", rect=(-100, -100, 100, 100))
        self.system = System(vessel=vessel, N=N, radius=radius, visual_radius=radius, temp=temp, dt=dt,
                             params=potential_params, mass=mass, enable_collisions=enable_collisions,
                             interaction_step=interaction_step)
        self.system.seed()

        if hasattr(self, 'particle_circles'):
            for c in self.particle_circles: c.remove()
        self.particle_circles = []
        for i in range(self.system.n()):
            circle = patches.Circle((self.system.pos[i, 0], self.system.pos[i, 1]),
                                    radius=self.system.visual_radius, facecolor="#215a93",
                                    edgecolor="black", linewidth=0.5)
            self.ax_anim.add_patch(circle)
            self.particle_circles.append(circle)
        self._draw_vessel_patch()
        self.ax_anim.relim()
        self.ax_anim.autoscale_view()
        self.canvas_anim.draw_idle()
        self.canvas_hist.draw_idle()
        self.running = True
        self._last_time = time.time()
        self._last_hist_update = 0
        self._last_canvas_update = 0

    def _start_timer(self):
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start()

    def _on_timer(self):
        if self.running and self.system is not None:
            self.system.integrate()
            self._redraw_particles()
            current_time = time.time()
            if current_time - self._last_hist_update > 1.5:
                self._update_histograms()
                self._last_hist_update = current_time

    def _redraw_particles(self):
        if hasattr(self, 'particle_circles') and self.system.pos is not None:
            for i, circle in enumerate(self.particle_circles):
                circle.center = (self.system.pos[i, 0], self.system.pos[i, 1])
            current_time = time.time()
            if current_time - self._last_canvas_update > 0.033:
                self.canvas_anim.draw_idle()
                self._last_canvas_update = current_time

    def _update_histograms(self):
        # Накопление данных о текущих позициях
        current_x = self.system.pos[:, 0]
        current_y = self.system.pos[:, 1]
        current_dists = self.system.pairwise_distances_fast(max_pairs=5000)

        # Добавляем текущие данные к накопленным
        self.accumulated_x = np.concatenate([self.accumulated_x, current_x])
        self.accumulated_y = np.concatenate([self.accumulated_y, current_y])
        if current_dists.size > 0:
            self.accumulated_distances = np.concatenate([self.accumulated_distances, current_dists])

        # Ограничиваем размер массивов для предотвращения переполнения памяти
        if len(self.accumulated_x) > self.max_accumulated_points:
            self.accumulated_x = self.accumulated_x[-self.max_accumulated_points:]
        if len(self.accumulated_y) > self.max_accumulated_points:
            self.accumulated_y = self.accumulated_y[-self.max_accumulated_points:]
        if len(self.accumulated_distances) > self.max_accumulated_points:
            self.accumulated_distances = self.accumulated_distances[-self.max_accumulated_points:]

        # Очищаем графики
        for ax in (self.ax_histx, self.ax_histy, self.ax_histd):
            ax.cla()
            ax.set_facecolor("#f9f9f9")
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.tick_params(labelsize=9)

        lang = self.get_lang_cb()
        s = STRINGS[lang.value]

        # Строим гистограммы из накопленных данных
        if len(self.accumulated_x) > 0:
            self.ax_histx.hist(self.accumulated_x, bins=self.bins_count, density=True, color="#8bb7d7", alpha=0.8)
        self.ax_histx.set_ylabel("p(x)", labelpad=5)
        self.ax_histx.set_title(s["sim.hist.x"], fontsize=10, fontweight='bold')

        if len(self.accumulated_y) > 0:
            self.ax_histy.hist(self.accumulated_y, bins=self.bins_count, density=True, color="#f9ad6c", alpha=0.8)
        self.ax_histy.set_ylabel("p(y)", labelpad=5)
        self.ax_histy.set_title(s["sim.hist.y"], fontsize=10, fontweight='bold')

        if len(self.accumulated_distances) > 0:
            self.ax_histd.hist(self.accumulated_distances, bins=self.bins_count, density=True, color="#8dd38d",
                               alpha=0.85)
        self.ax_histd.set_ylabel("p(r)", labelpad=5)
        self.ax_histd.set_xlabel("r")
        self.ax_histd.set_title(s["sim.hist.r"], fontsize=10, fontweight='bold')

        self.canvas_hist.draw_idle()

    def _toggle_run(self):
        lang = self.get_lang_cb()
        s = STRINGS[lang.value]
        self.running = not self.running
        self.btn_run.setText(s["sim.btn.pause"] if self.running else s["sim.btn.start"])

    def _toggle_settings(self):
        lang = self.get_lang_cb()
        s = STRINGS[lang.value]
        if self.settings_panel.isVisible():
            self.settings_panel.hide()
            # self.btn_collapse.setText(s["sim.btn.expand"])
        else:
            self.settings_panel.show()
            # self.btn_collapse.setText(s["sim.btn.collapse"])

    def _enter_draw_mode(self):
        vessel_kind = str(self.vessel_box.currentText())
        if vessel_kind in ("poly", "Многоугольник"):
            self.draw_mode = True
            self.poly_points = []

    def _clear_poly(self):
        vessel_kind = str(self.vessel_box.currentText())
        if vessel_kind in ("poly", "Многоугольник"):
            self.system.vessel.poly = None
            self._draw_vessel_patch()
            self.canvas_anim.draw_idle()

    def _on_mouse(self, event):
        if not self.draw_mode or event.inaxes != self.ax_anim: return
        vessel_kind = str(self.vessel_box.currentText())
        if vessel_kind not in ("poly", "Многоугольник"): return
            
        if event.button == 1:
            self.poly_points.append((event.xdata, event.ydata))
            self._preview_poly()
        elif event.button == 3:
            if len(self.poly_points) >= 3:
                self.system.vessel.poly = np.array(self.poly_points)
                self.draw_mode = False
                self.poly_points = []
                self._draw_vessel_patch()
                self.system.seed()
                self._redraw_particles()

    def _preview_poly(self):
        if len(self.poly_points) >= 2:
            pts = np.array(self.poly_points)
            if hasattr(self, 'vessel_artist') and self.vessel_artist is not None:
                try:
                    self.vessel_artist.remove()
                except Exception:
                    pass
            self.vessel_artist = patches.Polygon(pts, closed=False, fill=False, lw=1.5, ec="#888", ls='--')
            self.ax_anim.add_patch(self.vessel_artist)
            self.canvas_anim.draw_idle()

    def _draw_vessel_patch(self):
        if hasattr(self, 'vessel_artist') and self.vessel_artist is not None:
            try:
                self.vessel_artist.remove()
            except Exception:
                pass
        v = self.system.vessel
        if v.kind in ("rect", "Прямоугольник"):
            xmin, ymin, xmax, ymax = v.rect
            self.vessel_artist = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, lw=2, ec="#444")
            self.ax_anim.set_xlim(xmin - 10, xmax + 10)
            self.ax_anim.set_ylim(ymin - 10, ymax + 10)
        elif v.kind in ("circle", "Круг"):
            cx, cy, R = v.circle
            self.vessel_artist = patches.Circle((cx, cy), R, fill=False, lw=2, ec="#444")
            self.ax_anim.set_xlim(cx - R - 10, cx + R + 10)
            self.ax_anim.set_ylim(cy - R - 10, cy + R + 10)
        elif v.kind in ("poly", "Многоугольник") and v.poly is not None:
            self.vessel_artist = patches.Polygon(v.poly, closed=True, fill=False, lw=2, ec="#444")
            xmin, ymin = v.poly.min(axis=0)
            xmax, ymax = v.poly.max(axis=0)
            self.ax_anim.set_xlim(xmin - 10, xmax + 10)
            self.ax_anim.set_ylim(ymin - 10, ymax + 10)
        else:
            self.vessel_artist = None
        if self.vessel_artist is not None:
            self.ax_anim.add_patch(self.vessel_artist)

    def update_language(self, lang: Lang):
        s = STRINGS[lang.value]
        self.lbl_settings_title.setText(s["sim.settings"])

        # Подписи слева (лейблы/юниты/хелпы)
        def set_lbl(obj_name, text_key):
            w = self.findChild(QtWidgets.QLabel, obj_name)
            if w: w.setText(STRINGS[lang.value].get(text_key, ""))

        set_lbl("lbl_N", "sim.N")
        set_lbl("unit_N", "sim.N.unit")
        set_lbl("help_N", "sim.N.help")
        set_lbl("lbl_R", "sim.R")
        set_lbl("unit_R", "sim.R.unit")
        set_lbl("help_R", "sim.R.help")
        set_lbl("lbl_T", "sim.T")
        set_lbl("unit_T", "sim.T.unit")
        set_lbl("help_T", "sim.T.help")
        set_lbl("lbl_m", "sim.m")
        set_lbl("unit_m", "sim.m.unit")
        set_lbl("help_m", "sim.m.help")
        self.check_collisions.setText(s["sim.collisions"])
        set_lbl("lbl_pot", "sim.pot")
        set_lbl("help_pot", "sim.pot.help")
        set_lbl("lbl_eps", "sim.eps")
        set_lbl("unit_eps", "sim.eps.unit")
        set_lbl("help_eps", "sim.eps.help")
        set_lbl("lbl_sigma", "sim.sigma")
        set_lbl("unit_sigma", "sim.sigma.unit")
        set_lbl("help_sigma", "sim.sigma.help")
        set_lbl("lbl_De", "sim.De")
        set_lbl("unit_De", "sim.De.unit")
        set_lbl("help_De", "sim.De.help")
        set_lbl("lbl_a", "sim.a")
        set_lbl("unit_a", "sim.a.unit")
        set_lbl("help_a", "sim.a.help")
        set_lbl("lbl_r0", "sim.r0")
        set_lbl("unit_r0", "sim.r0.unit")
        set_lbl("help_r0", "sim.r0.help")
        set_lbl("lbl_vessel", "sim.vessel")
        set_lbl("help_vessel", "sim.vessel.help")
        set_lbl("lbl_interact", "sim.interact")
        #set_lbl("unit_interact", "sim.interact.unit")
        set_lbl("lbl_bins", "sim.bins")
        set_lbl("help_bins", "sim.bins.help")

        self.btn_draw.setText(s["sim.btn.draw"])
        self.btn_clear.setText(s["sim.btn.clear"])
        self.btn_run.setText(s["sim.btn.pause"] if self.running else s["sim.btn.start"])
        # self.btn_apply.setText(s["sim.btn.apply"])
        self.btn_back.setText(s["sim.btn.back"])
        # self.btn_collapse.setText(s["sim.btn.collapse"] if self.settings_panel.isVisible() else s["sim.btn.expand"])
        self.btn_toggle_settings_small.setText(s["sim.top.settings"])
        self.ax_anim.set_title(s["sim.anim.title"])
        self.canvas_anim.draw_idle()

        # Обновляем названия потенциалов
        current_index = self.pot_box.currentIndex()
        if lang == Lang.RU:
            self.pot_box.clear()
            self.pot_box.addItems(["Нет", "Отталкивание", "Притяжение", "Леннард-Джонс", "Морзе"])
        elif lang == Lang.EN:
            self.pot_box.clear()
            self.pot_box.addItems(["None", "Repulsion", "Attraction", "Lennard-Jones", "Morse"])
        elif lang == Lang.CN:
            self.pot_box.clear()
            self.pot_box.addItems(["无", "排斥", "吸引", "伦纳德-琼斯", "莫尔斯"])
        self.pot_box.setCurrentIndex(current_index)

        # Обновляем названия сосудов
        current_vessel_index = self.vessel_box.currentIndex()
        self.vessel_box.clear()
        self.vessel_box.addItems([
            s["vessel.rect"],
            s["vessel.circle"],
            s["vessel.poly"]
        ])
        self.vessel_box.setCurrentIndex(current_vessel_index)
        
        # Обновляем состояние кнопок полигона после смены языка
        self._update_polygon_buttons_state()


class AuthorsWidget(QtWidgets.QWidget):
    def __init__(self, parent, back_cb, get_lang_cb):
        super().__init__(parent)
        self.back_cb = back_cb
        self.get_lang_cb = get_lang_cb
        self._build_ui()

    def _load_sticker(self, fname, size=420):
        try:
            path = os.path.join(os.path.dirname(__file__), "images", fname)
            img = Image.open(path).convert("RGBA")
            data = np.array(img)
            mask = (data[:, :, 0] > 240) & (data[:, :, 1] > 240) & (data[:, :, 2] > 240)
            data[mask, 3] = 0
            img = Image.fromarray(data)
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            return QtGui.QPixmap.fromImage(ImageQt.ImageQt(img))
        except Exception:
            return None

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Используем локализацию
        self.title_label = QtWidgets.QLabel()
        self.title_label.setStyleSheet("font-size:24pt; font-weight:700;")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        cards = QtWidgets.QHBoxLayout()
        layout.addLayout(cards)

        # -------- LEFT (author 1): имя сверху, фото снизу --------
        left = QtWidgets.QVBoxLayout()
        left.addSpacing(80)
        left.setSpacing(6)  # 🔹 уменьшили расстояние между текстом и фото
        left.setContentsMargins(0, 0, 0, 0)  # 🔹 убрали внутренние поля

        self.name1_label = QtWidgets.QLabel()
        self.name1_label.setStyleSheet("font-size:16pt; font-weight:600;")
        self.name1_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.name1_label)  # <-- СНАЧАЛА ТЕКСТ

        pix = self._load_sticker("author1.png") or self._load_sticker("author1.jpg")
        if pix:
            lbl = QtWidgets.QLabel()
            lbl.setPixmap(pix)
            lbl.setFixedSize(400, 500)
            lbl.setScaledContents(True)
            left.addWidget(lbl, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)  # <-- ПОТОМ ФОТО

        cards.addLayout(left)

        # -------- RIGHT (author 2): имя сверху, фото снизу --------
        right = QtWidgets.QVBoxLayout()
        right.addSpacing(80)
        right.setSpacing(6)  # 🔹 уменьшили расстояние между текстом и фото
        right.setContentsMargins(0, 0, 0, 0)  # 🔹 убрали внутренние поля

        self.name2_label = QtWidgets.QLabel()
        self.name2_label.setStyleSheet("font-size:16pt; font-weight:600;")
        self.name2_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.name2_label)  # <-- СНАЧАЛА ТЕКСТ

        pix2 = self._load_sticker("author2.png") or self._load_sticker("author2.jpg")
        if pix2:
            lbl2 = QtWidgets.QLabel()
            lbl2.setPixmap(pix2)
            lbl2.setFixedSize(400, 500)
            lbl2.setScaledContents(True)
            right.addWidget(lbl2, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)  # <-- ПОТОМ ФОТО

        cards.addLayout(right)

        # Сделаем колонки одинаковой ширины
        cards.setStretch(0, 1)
        cards.setStretch(1, 1)

        # -------- BOSS (по центру снизу, только текст) --------
        self.boss_label = QtWidgets.QLabel()
        self.boss_label.setStyleSheet("font-size:16pt; font-weight:600;")
        self.boss_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Прижимаем низ: сначала растяжка, затем boss, затем кнопка
        layout.addStretch(1)
        layout.addWidget(self.boss_label, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(12)

        self.back_btn = QtWidgets.QPushButton()
        self.back_btn.setFixedSize(200, 48)
        self.back_btn.setStyleSheet("background:#313132;color:white;font-weight:600;font-size:12pt;border-radius:6px;")
        self.back_btn.clicked.connect(self.back_cb)
        layout.addWidget(self.back_btn, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(20)

        # Устанавливаем текст при инициализации
        self.update_language(self.get_lang_cb())

    def update_language(self, lang: Lang):
        s = STRINGS[lang.value]
        self.title_label.setText(s["authors.title"])
        self.name1_label.setText(s["authors.name1"])
        self.name2_label.setText(s["authors.name2"])
        self.boss_label.setText(s["boss"])
        self.back_btn.setText(s["authors.back"])


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.lang = Lang.RU

        self._build_ui()
        self.apply_language()
        self.setWindowState(self.windowState() | QtCore.Qt.WindowState.WindowFullScreen)

    def _build_ui(self):
        central = QtWidgets.QStackedWidget()
        self.setCentralWidget(central)

        # Pages
        self.menu = MainMenuWidget(
            self,
            start_cb=self.show_sim,
            lang_toggle_cb=self.toggle_language,
            get_lang_cb=lambda: self.lang,
            authors_cb=self.show_authors
        )
        self.sim = SimulationWidget(self, back_cb=self.show_menu, get_lang_cb=lambda: self.lang)
        self.authors = AuthorsWidget(self, back_cb=self.show_menu, get_lang_cb=lambda: self.lang)

        central.addWidget(self.menu)
        central.addWidget(self.sim)
        central.addWidget(self.authors)
        self.stack = central

        self.setWindowTitle(STRINGS[self.lang.value]["app.title"])
        self.show_menu()

    def show_menu(self):
        self.stack.setCurrentWidget(self.menu)

    def show_sim(self):
        self.stack.setCurrentWidget(self.sim)

    def show_authors(self):
        self.stack.setCurrentWidget(self.authors)

    def set_language(self, lang: Lang):
        self.lang = lang
        self.apply_language()

    def toggle_language(self):
        # Циклическое переключение: RU → EN → CN → RU → ...
        if self.lang == Lang.RU:
            self.lang = Lang.EN
        elif self.lang == Lang.EN:
            self.lang = Lang.CN
        else:  # CN
            self.lang = Lang.RU
        self.apply_language()

    def apply_language(self):
        s = STRINGS[self.lang.value]
        self.setWindowTitle(s["app.title"])
        if hasattr(self.menu, "update_language"):
            self.menu.update_language(self.lang)
        if hasattr(self.sim, "update_language"):
            self.sim.update_language(self.lang)
        if hasattr(self.authors, "update_language"):
            self.authors.update_language(self.lang)


def main():
    app = QtWidgets.QApplication(sys.argv)

    font_families = ["Microsoft YaHei", "Noto Sans CJK SC", "SimSun", "Arial"]
    for font_family in font_families:
        if QFont(font_family).exactMatch():
            font = QFont(font_family, 9)
            app.setFont(font)
            break

    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').disabled = True

    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # Важно для правильного отображения минуса
    matplotlib.use('Agg')

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
