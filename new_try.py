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
        "menu.theory": "Теория",
        "menu.exit": "Выход",

        "vessel.rect": "Прямоугольник",
        "vessel.circle": "Круг",
        "vessel.poly": "Многоугольник",

        "sim.settings": "Настройки симуляции",
        "sim.N": "N",
        "sim.N.unit": "1-100",
        "sim.N.help": "Число частиц",
        "sim.R": "R",
        "sim.R.unit": "0.5-10",
        "sim.R.help": "Радиус частицы",
        "sim.T": "T",
        "sim.T.unit": "0.1-50",
        "sim.T.help": "Температура",
        "sim.m": "m",
        "sim.m.unit": "0.01-100",
        "sim.m.help": "Масса частицы",
        "sim.collisions": "Столкновения частиц",
        "sim.pot": "Тип",
        "sim.pot.help": "Тип парного потенциала",
        "sim.eps": "ε",
        "sim.eps.unit": "0-100",
        "sim.eps.help": "Энергетический масштаб",
        "sim.sigma": "σ",
        "sim.sigma.unit": "1-30",
        "sim.sigma.help": "Характерное расстояние LJ",
        "sim.De": "De",
        "sim.De.unit": "0-100",
        "sim.De.help": "Глубина ямы Morse",
        "sim.a": "a",
        "sim.a.unit": "0.01-5",
        "sim.a.help": "Жёсткость Morse",
        "sim.r0": "r₀",
        "sim.r0.unit": "1-20",
        "sim.r0.help": "Равновесное расстояние (Morse)",
        "sim.vessel": "Сосуд",
        "sim.vessel.help": "Форма сосуда",
        "sim.interact": "Интеракции",
        "sim.interact.unit": "к-я итерация",
        "sim.bins": "Бины",
        "sim.bins.help": "Количество бинов для гистограмм",

        "sim.btn.draw": "Рисовать",
        "sim.btn.clear": "Очистить",
        "sim.btn.pause": "Пауза",
        "sim.btn.start": "Старт",
        "sim.btn.apply": "Применить",
        "sim.btn.back": "Назад",
        "sim.btn.collapse": "Свернуть",
        "sim.btn.expand": "Развернуть",
        "sim.top.settings": "≡ Настройки",
        "sim.anim.title": "Область моделирования",

        "sim.hist.x": "Распределение X",
        "sim.hist.y": "Распределение Y",
        "sim.hist.r": "Распределение расстояний",
        "sim.hist.title": "Гистограммы",

        "sim.btn.select_particles": "Выбор частиц",
        "sim.btn.cancel_selection": "Отменить",

        "sim.histograms.title": "Гистограммы",
        "sim.display_mode.distance": "Расстояние",
        "sim.display_mode.projections": "Проекции на оси",
        "sim.display_mode.signed": "Расстояние со знаком",
        "sim.btn.reset_data": "Сбросить",
        "sim.btn.reset_data.tooltip": "Очистить все накопленные данные гистограмм",

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
        "menu.theory": "Theory",
        "menu.exit": "Exit",

        "vessel.rect": "Rectangle",
        "vessel.circle": "Circle",
        "vessel.poly": "Polygon",

        "sim.settings": "Simulation settings",
        "sim.N": "N",
        "sim.N.unit": "5-100",
        "sim.N.help": "Number of particles",
        "sim.R": "R",
        "sim.R.unit": "0.5-10",
        "sim.R.help": "Particle radius",
        "sim.T": "T",
        "sim.T.unit": "0.1-50",
        "sim.T.help": "Temperature",
        "sim.m": "m",
        "sim.m.unit": "0.01-100",
        "sim.m.help": "Particle mass",
        "sim.collisions": "Particle collisions",
        "sim.pot": "Type",
        "sim.pot.help": "Pair potential type",
        "sim.eps": "ε",
        "sim.eps.unit": "0-100",
        "sim.eps.help": "Energy scale",
        "sim.sigma": "σ",
        "sim.sigma.unit": "1-30",
        "sim.sigma.help": "Characteristic LJ distance",
        "sim.De": "De",
        "sim.De.unit": "0-100",
        "sim.De.help": "Morse well depth",
        "sim.a": "a",
        "sim.a.unit": "0.01-5",
        "sim.a.help": "Morse stiffness",
        "sim.r0": "r₀",
        "sim.r0.unit": "1-20",
        "sim.r0.help": "Equilibrium distance (Morse)",
        "sim.vessel": "Vessel",
        "sim.vessel.help": "Vessel shape",
        "sim.interact": "Interactions",
        "sim.interact.unit": "th iter",
        "sim.bins": "Bins",
        "sim.bins.help": "Number of bins for histograms",

        "sim.btn.draw": "Draw",
        "sim.btn.clear": "Clear",
        "sim.btn.pause": "Pause",
        "sim.btn.start": "Start",
        "sim.btn.apply": "Apply",
        "sim.btn.back": "Back",
        "sim.btn.collapse": "Collapse",
        "sim.btn.expand": "Expand",
        "sim.top.settings": "≡ Settings",
        "sim.anim.title": "Simulation area",

        "sim.hist.x": "Distribution of X",
        "sim.hist.y": "Distribution of Y",
        "sim.hist.r": "Distribution of pair distances",
        "sim.hist.title": "Histograms",

        "sim.btn.select_particles": "Select particles",
        "sim.btn.cancel_selection": "Cancel",

        "sim.histograms.title": "Histograms",
        "sim.display_mode.distance": "Distance",
        "sim.display_mode.projections": "Projections",
        "sim.display_mode.signed": "Signed distance",
        "sim.btn.reset_data": "Reset",
        "sim.btn.reset_data.tooltip": "Clear all accumulated histogram data",

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
        "menu.theory": "理论",
        "menu.exit": "退出",

        "vessel.rect": "矩形",
        "vessel.circle": "圆形",
        "vessel.poly": "多边形",

        "sim.settings": "模拟设置",
        "sim.N": "N",
        "sim.N.unit": "5-100",
        "sim.N.help": "粒子数量",
        "sim.R": "R",
        "sim.R.unit": "0.5-10",
        "sim.R.help": "粒子半径",
        "sim.T": "T",
        "sim.T.unit": "0.1-50",
        "sim.T.help": "温度",
        "sim.m": "m",
        "sim.m.unit": "0.01-100",
        "sim.m.help": "粒子质量",
        "sim.collisions": "粒子碰撞",
        "sim.pot": "类型",
        "sim.pot.help": "对势类型",
        "sim.eps": "ε",
        "sim.eps.unit": "0-100",
        "sim.eps.help": "能量尺度",
        "sim.sigma": "σ",
        "boss": "奇奇吉娜·奥尔加·亚历山德罗夫娜",
        "sim.sigma.unit": "1-30",
        "sim.sigma.help": "LJ特征距离",
        "sim.De": "De",
        "sim.De.unit": "0-100",
        "sim.De.help": "Morse势阱深度",
        "sim.a": "a",
        "sim.a.unit": "0.01-5",
        "sim.a.help": "Morse刚度",
        "sim.r0": "r₀",
        "sim.r0.unit": "1-20",
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
        "sim.anim.title": "模拟区域",

        "sim.hist.x": "X分布",
        "sim.hist.y": "Y分布",
        "sim.hist.r": "距离分布",
        "sim.hist.title": "直方图",

        "sim.btn.select_particles": "选择粒子",
        "sim.btn.cancel_selection": "取消选择",

        "sim.histograms.title": "直方图",
        "sim.display_mode.distance": "常规距离",
        "sim.display_mode.projections": "投影",
        "sim.display_mode.signed": "带符号距离",
        "sim.btn.reset_data": "重置数据",
        "sim.btn.reset_data.tooltip": "清除所有累积的直方图数据",

        "menu.authors": "关于作者",
        "authors.title": "开发团队",
        "authors.name1": "叶尼亚金·斯坦尼斯拉夫",
        "authors.name2": "科热米亚科娃·伊丽莎白",
        "boss": "科学指导：奇奇吉娜·奥尔加·亚历山德罗夫娜",
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
    a: float = 0.5  # было 0.02
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
                mag = k / r / r

            elif params_kind == 2:  # Притяжение ~ -k / r^2
                mag = -k / r / r

            elif params_kind == 3:  # Lennard-Jones (12-6)
                # F = 24ε(2(σ/r)^12 - (σ/r)^6) / r
                sr = sigma / r
                sr6 = sr ** 6
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
    def __init__(self, parent, start_cb, lang_toggle_cb, get_lang_cb, authors_cb=None, theory_cb=None):
        super().__init__(parent)
        self.start_cb = start_cb
        self.authors_cb = authors_cb  # ← сохраняем колбэк (может быть None)
        self.theory_cb = theory_cb  # ← новый колбэк для теории
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
        layout.setContentsMargins(60, 60, 60, 60)

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

        # Кнопка Теория (новая кнопка)
        if hasattr(self, "theory_cb") and self.theory_cb:
            self.btn_theory = QtWidgets.QPushButton()
            self.btn_theory.setFixedSize(240, 50)  # ФИКСИРУЕМ РАЗМЕР
            self.btn_theory.setStyleSheet(
                "background:#5a8d5a; color:white; font-weight:600; font-size:14pt; border-radius:6px; margin-top:10px;"
            )
            self.btn_theory.clicked.connect(self.theory_cb)
            center.addWidget(self.btn_theory, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

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
        if hasattr(self, 'btn_theory') and self.btn_theory is not None:
            self.btn_theory.setText(s.get('menu.theory', 'Теория'))


class SimulationWidget(QtWidgets.QWidget):
    def __init__(self, parent, back_cb, get_lang_cb):
        super().__init__(parent)

        self.accumulated_x = np.array([])
        self.accumulated_y = np.array([])
        self.accumulated_distances = np.array([])
        self.max_accumulated_points = 100000

        self.selected_particles = []
        self.distance_line = None
        self.particle_selection_mode = False
        self.selected_particles_data = {
            'x': np.array([]),
            'y': np.array([]),
            'distances': np.array([]),
            'dx': np.array([]),  # проекция X
            'dy': np.array([]),  # проекция Y
            'signed_distance': np.array([])  # расстояние со знаком
        }

        # Новый режим отображения графиков
        self.distance_display_mode = 0  # 0 - обычное расстояние, 1 - проекции, 2 - расстояние со знаком

        self.back_cb = back_cb
        self.get_lang_cb = get_lang_cb
        self.bins_count = 36
        self._build_ui()
        self._init_simulation()
        self._start_timer()
        self.update_language(self.get_lang_cb())
        self._reset_data()

    # --- новые вспомогательные методы для нормализации типа сосуда ---
    def _vessel_display_to_key(self, display_text: str) -> str:
        """Преобразует текст из vessel_box (локализованный) в канонический ключ: 'rect'/'circle'/'poly'."""
        lang = self.get_lang_cb()
        s = STRINGS[lang.value]
        txt = str(display_text)
        if txt == s.get("vessel.rect", "") or txt.lower() in ("прямоугольник", "rectangle", "rect"):
            return "rect"
        if txt == s.get("vessel.circle", "") or txt.lower() in ("круг", "circle", "circ"):
            return "circle"
        if txt == s.get("vessel.poly", "") or txt.lower() in ("многоугольник", "polygon", "poly", "多边形"):
            return "poly"
        # fallback: try to inspect substrings
        l = txt.lower()
        if "rect" in l or "прям" in l: return "rect"
        if "circ" in l or "круг" in l: return "circle"
        if "poly" in l or "много" in l or "多" in l: return "poly"
        return txt

    def _vessel_kind_to_key(self, kind_str: str) -> str:
        """Преобразует значение Vessel.kind (возможны локализованные и канонические значения) в канонический ключ."""
        # просто делегируем к display->key (покроет оба варианта)
        return self._vessel_display_to_key(kind_str)

    def _sanitize_selected_particles(self):
        """Убирает из self.selected_particles индексы, вышедшие за пределы текущей системы."""
        if not hasattr(self, 'selected_particles') or self.system is None or self.system.pos is None:
            self.selected_particles = []
            return
        
        N = self.system.n()
        
        # Если частиц меньше 2, автоматически выключаем режим выбора
        if N < 2 and self.particle_selection_mode:
            self.particle_selection_mode = False
            self.selected_particles = []
            # Обновляем текст кнопки
            lang = self.get_lang_cb()
            s = STRINGS[lang.value]
            self.btn_particle_select.setText(s.get("sim.btn.select_particles", "Выбор частиц"))
            # Удаляем линию расстояния если была
            if self.distance_line:
                self.distance_line.remove()
                self.distance_line = None
        
        # Фильтруем только валидные индексы
        valid_particles = [i for i in self.selected_particles if 0 <= i < N]
        
        # В режиме выбора частиц оставляем максимум 2 последние выбранные
        if self.particle_selection_mode and len(valid_particles) > 2:
            valid_particles = valid_particles[-2:]
        
        self.selected_particles = valid_particles

    def _build_ui(self):
        main_l = QtWidgets.QHBoxLayout(self)
        self.settings_panel = QtWidgets.QWidget()
        self.settings_panel.setFixedWidth(300)
        self.settings_panel.setStyleSheet("background:#f0f0f2; border-right: 2px solid #ddd;")
        sp_layout = QtWidgets.QVBoxLayout(self.settings_panel)
        sp_layout.setContentsMargins(8, 8, 8, 8)
        sp_layout.setSpacing(4)

        self.lbl_settings_title = QtWidgets.QLabel()
        self.lbl_settings_title.setStyleSheet("font-size:17pt; font-weight:700; color:#1a1a1a;")
        sp_layout.addWidget(self.lbl_settings_title)
        sp_layout.addSpacing(4)

        def add_slider_row(key_label, slider, label_value, key_unit="", key_help=""):
            """Компактная строка для слайдера: [Label] [Slider] [Value] [Unit]"""
            if key_help:
                h = QtWidgets.QLabel()
                h.setObjectName(f"help_{key_label}")
                h.setWordWrap(True)
                h.setStyleSheet("color:#666; font-size:12pt; margin-top:4px; padding:0px;")
                sp_layout.addWidget(h)
            
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(2, 0, 2, 0)
            row.setSpacing(4)
            
            lbl = QtWidgets.QLabel()
            lbl.setObjectName(f"lbl_{key_label}")
            lbl.setStyleSheet("font-size:13pt; font-weight:500; color:#2c2c2c;")
            lbl.setFixedWidth(45)
            row.addWidget(lbl)
            
            slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    background: #ddd;
                    height: 6px;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background: #3271a8;
                    width: 14px;
                    margin: -4px 0;
                    border-radius: 7px;
                }
                QSlider::handle:horizontal:hover {
                    background: #2a5f96;
                }
            """)
            row.addWidget(slider)
            
            label_value.setFixedWidth(45)
            label_value.setStyleSheet("font-size:12pt; color:#555; text-align:center;")
            row.addWidget(label_value)
            
            if key_unit:
                u = QtWidgets.QLabel()
                u.setObjectName(f"unit_{key_label}")
                u.setFixedWidth(50)
                u.setStyleSheet("font-size:12pt; color:#777;")
                row.addWidget(u)
            
            sp_layout.addLayout(row)

        def add_combobox_row(key_label, combobox, key_help=""):
            """Строка для комбобса: [Help сверху] [Label] [ComboBox]"""
            if key_help:
                h = QtWidgets.QLabel()
                h.setObjectName(f"help_{key_label}")
                h.setWordWrap(True)
                h.setStyleSheet("color:#666; font-size:12pt; margin:0px; padding:0px;")
                sp_layout.addWidget(h)
            
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(2, 0, 2, 0)
            row.setSpacing(4)
            
            lbl = QtWidgets.QLabel()
            lbl.setObjectName(f"lbl_{key_label}")
            lbl.setStyleSheet("font-size:13pt; font-weight:500; color:#2c2c2c;")
            lbl.setFixedWidth(45)
            row.addWidget(lbl)
            
            combobox.setStyleSheet("""
                QComboBox {
                    font-size:12pt; 
                    padding: 3px; 
                    border: 1px solid #bbb; 
                    border-radius: 3px;
                    background: white;
                }
            """)
            row.addWidget(combobox)
            sp_layout.addLayout(row)

        # N — ползунок (1-100)
        self.spin_N = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.spin_N.setRange(1, 100)
        self.spin_N.setValue(10)
        self.spin_N.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.spin_N_label = QtWidgets.QLabel("10")
        self.spin_N.sliderMoved.connect(lambda v: self.spin_N_label.setText(str(v)))
        self.spin_N.valueChanged.connect(lambda v: (self.spin_N_label.setText(str(v)), self._on_settings_changed()))
        add_slider_row("N", self.spin_N, self.spin_N_label, "sim.N.unit", "sim.N.help")

        # T — ползунок (0.1-50)
        self.edit_T = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.edit_T.setRange(1, 500)
        self.edit_T.setValue(50)
        self.edit_T.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.edit_T_label = QtWidgets.QLabel("5.0")
        self.edit_T.sliderMoved.connect(lambda v: self.edit_T_label.setText(f"{v/10:.1f}"))
        self.edit_T.valueChanged.connect(lambda v: (self.edit_T_label.setText(f"{v/10:.1f}"), self._on_settings_changed()))
        add_slider_row("T", self.edit_T, self.edit_T_label, "sim.T.unit", "sim.T.help")

        # m — ползунок (0.01-100)
        self.edit_m = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.edit_m.setRange(1, 10000)
        self.edit_m.setValue(1000)
        self.edit_m.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.edit_m_label = QtWidgets.QLabel("1.0")
        self.edit_m.sliderMoved.connect(lambda v: self.edit_m_label.setText(f"{v/1000:.2f}"))
        self.edit_m.valueChanged.connect(lambda v: (self.edit_m_label.setText(f"{v/1000:.2f}"), self._on_settings_changed()))
        add_slider_row("m", self.edit_m, self.edit_m_label, "sim.m.unit", "sim.m.help")

        self.check_collisions = WinCheckBox()
        self.check_collisions.stateChanged.connect(self._on_settings_changed)
        self.check_collisions.setStyleSheet("font-size:13pt; color:#2c2c2c; margin-top:4px;")
        self.check_collisions.setChecked(True)
        sp_layout.addWidget(self.check_collisions)

        self.pot_box = QtWidgets.QComboBox()
        self.pot_box.currentIndexChanged.connect(self._on_settings_changed)
        add_combobox_row("pot", self.pot_box, "sim.pot.help")

        # eps, sigma, De, a, r0 — ползунки
        self.edit_eps = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.edit_eps.setRange(0, 1000)
        self.edit_eps.setValue(100)
        self.edit_eps.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.edit_eps_label = QtWidgets.QLabel("10.0")
        self.edit_eps.sliderMoved.connect(lambda v: self.edit_eps_label.setText(f"{v/10:.1f}"))
        self.edit_eps.valueChanged.connect(lambda v: (self.edit_eps_label.setText(f"{v/10:.1f}"), self._on_settings_changed()))
        add_slider_row("eps", self.edit_eps, self.edit_eps_label, "sim.eps.unit", "sim.eps.help")

        self.edit_sigma = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.edit_sigma.setRange(10, 300)
        self.edit_sigma.setValue(150)
        self.edit_sigma.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.edit_sigma_label = QtWidgets.QLabel("15.0")
        self.edit_sigma.sliderMoved.connect(lambda v: self.edit_sigma_label.setText(f"{v/10:.1f}"))
        self.edit_sigma.valueChanged.connect(lambda v: (self.edit_sigma_label.setText(f"{v/10:.1f}"), self._on_settings_changed()))
        add_slider_row("sigma", self.edit_sigma, self.edit_sigma_label, "sim.sigma.unit", "sim.sigma.help")

        self.edit_De = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.edit_De.setRange(0, 1000)
        self.edit_De.setValue(200)
        self.edit_De.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.edit_De_label = QtWidgets.QLabel("20.0")
        self.edit_De.sliderMoved.connect(lambda v: self.edit_De_label.setText(f"{v/10:.1f}"))
        self.edit_De.valueChanged.connect(lambda v: (self.edit_De_label.setText(f"{v/10:.1f}"), self._on_settings_changed()))
        add_slider_row("De", self.edit_De, self.edit_De_label, "sim.De.unit", "sim.De.help")

        self.edit_a = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.edit_a.setRange(1, 500)
        self.edit_a.setValue(50)
        self.edit_a.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.edit_a_label = QtWidgets.QLabel("0.5")
        self.edit_a.sliderMoved.connect(lambda v: self.edit_a_label.setText(f"{v/100:.2f}"))
        self.edit_a.valueChanged.connect(lambda v: (self.edit_a_label.setText(f"{v/100:.2f}"), self._on_settings_changed()))
        add_slider_row("a", self.edit_a, self.edit_a_label, "sim.a.unit", "sim.a.help")

        self.edit_r0 = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.edit_r0.setRange(10, 200)
        self.edit_r0.setValue(25)
        self.edit_r0.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.edit_r0_label = QtWidgets.QLabel("2.5")
        self.edit_r0.sliderMoved.connect(lambda v: self.edit_r0_label.setText(f"{v/10:.1f}"))
        self.edit_r0.valueChanged.connect(lambda v: (self.edit_r0_label.setText(f"{v/10:.1f}"), self._on_settings_changed()))
        add_slider_row("r0", self.edit_r0, self.edit_r0_label, "sim.r0.unit", "sim.r0.help")

        self.vessel_box = QtWidgets.QComboBox()
        self.vessel_box.currentIndexChanged.connect(self._on_vessel_changed)
        add_combobox_row("vessel", self.vessel_box, "sim.vessel.help")

        self.spin_interact_step = QtWidgets.QSpinBox()
        self.spin_interact_step.setRange(1, 1000)
        self.spin_interact_step.setValue(1)
        self.spin_interact_step.setFixedHeight(22)
        self.spin_interact_step.valueChanged.connect(self._on_settings_changed)

        self.spin_bins = QtWidgets.QSpinBox()
        self.spin_bins.setRange(5, 100)
        self.spin_bins.setValue(self.bins_count)
        self.spin_bins.setFixedHeight(30)
        self.spin_bins.valueChanged.connect(self._on_bins_changed)
        add_combobox_row("bins", self.spin_bins, "sim.bins.help")

        sp_layout.addSpacing(6)

        # создаём кнопки, но не добавляем в settings panel — они будут в верхней панели анимации
        self.btn_draw = QtWidgets.QPushButton()
        self.btn_draw.clicked.connect(self._enter_draw_mode)
        self.btn_draw.setStyleSheet("""
            QPushButton {
                font-size:13pt; 
                font-weight:600;
                padding: 4px 8px;  
                background: #5a8dba; 
                color: white; 
                border: none; 
                border-radius: 4px;
            }
            QPushButton:hover { background: #4a7da8; }
            QPushButton:pressed { background: #3a6d98; }
            QPushButton:disabled { background: #cccccc; color: #666666; }
        """)

        self.btn_clear = QtWidgets.QPushButton()
        self.btn_clear.clicked.connect(self._clear_poly)
        self.btn_clear.setStyleSheet("""
            QPushButton {
                font-size:13pt; 
                font-weight:600;
                padding: 4px 8px; 
                background: #5a8dba; 
                color: white; 
                border: none; 
                border-radius: 4px;
            }
            QPushButton:hover { background: #4a7da8; }
            QPushButton:pressed { background: #3a6d98; }
            QPushButton:disabled { background: #cccccc; color: #666666; }
        """)

        self.btn_particle_select = QtWidgets.QPushButton()
        self.btn_particle_select.clicked.connect(self._toggle_particle_selection)
        self.btn_particle_select.setStyleSheet("""
            QPushButton {
                font-size:13pt; 
                font-weight:600;
                padding: 4px 8px;  
                background: #6b8e7f; 
                color: white; 
                border: none; 
                border-radius: 4px;
            }
            QPushButton:hover { background: #5b7e6f; }
            QPushButton:pressed { background: #4b6e5f; }
        """)

        self.btn_run = QtWidgets.QPushButton()
        self.btn_run.clicked.connect(self._toggle_run)
        self.btn_run.setStyleSheet("""
            QPushButton {
                font-size:13pt; 
                font-weight:600;
                padding: 4px 8px;
                background: #a85c3c; 
                color: white; 
                border: none; 
                border-radius: 4px;
            }
            QPushButton:hover { background: #984c2c; }
            QPushButton:pressed { background: #883c1c; }
        """)

        self.btn_reset_data = QtWidgets.QPushButton()
        self.btn_reset_data.clicked.connect(self._reset_data)
        self.btn_reset_data.setStyleSheet("""
            QPushButton {
                font-size:13pt; 
                font-weight:600;
                padding: 4px 8px;
                background: #c97c4c; 
                color: white; 
                border: none; 
                border-radius: 4px;
            }
            QPushButton:hover { background: #b96c3c; }
            QPushButton:pressed { background: #a95c2c; }
        """)

        self.btn_back = QtWidgets.QPushButton()
        self.btn_back.clicked.connect(self.back_cb)
        self.btn_back.setFixedHeight(25)
        self.btn_back.setStyleSheet("""
            QPushButton {
                font-size:13pt; 
                font-weight:600;
                background: #313132; 
                color: white; 
                border: none; 
                border-radius: 4px;
            }
            QPushButton:hover { background: #413132; }
            QPushButton:pressed { background: #211112; }
        """)
        sp_layout.addStretch(1)
        sp_layout.addWidget(self.btn_back)

        canvas_container = QtWidgets.QHBoxLayout()
        main_l.addWidget(self.settings_panel)
        main_l.addLayout(canvas_container, stretch=1)

        anim_container = QtWidgets.QVBoxLayout()
        canvas_container.addLayout(anim_container, stretch=2)

        hist_container = QtWidgets.QVBoxLayout()
        canvas_container.addLayout(hist_container, stretch=1)

        # Верхняя панель для анимации
        top_bar = QtWidgets.QHBoxLayout()
        self.btn_toggle_settings_small = QtWidgets.QPushButton()
        self.btn_toggle_settings_small.setFixedSize(110, 30)
        self.btn_toggle_settings_small.setStyleSheet("""
            QPushButton {
                font-size:13pt; 
                font-weight:600;
                background: #5a8dba; 
                color: white; 
                border: none; 
                border-radius: 5px;
            }
            QPushButton:hover {
                background: #4a7da8;
            }
        """)
        self.btn_toggle_settings_small.clicked.connect(self._toggle_settings)
        top_bar.addWidget(self.btn_toggle_settings_small, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        top_bar.addSpacing(6)
        self.btn_draw.setFixedSize(110, 30)
        top_bar.addWidget(self.btn_draw)
        self.btn_clear.setFixedSize(110, 30)
        top_bar.addWidget(self.btn_clear)
        self.btn_particle_select.setFixedSize(110, 30)
        top_bar.addWidget(self.btn_particle_select)
        self.btn_run.setFixedSize(110, 30)
        top_bar.addWidget(self.btn_run)
        self.btn_reset_data.setFixedSize(110, 30)
        top_bar.addWidget(self.btn_reset_data)
        top_bar.addStretch(1)
        anim_container.addLayout(top_bar)

        self.fig_anim = plt.Figure(figsize=(8, 8))
        self.ax_anim = self.fig_anim.add_subplot(111)
        self.ax_anim.set_xticks([])
        self.ax_anim.set_yticks([])
        self.ax_anim.set_aspect('equal')
        self.ax_anim.set_facecolor('#ffffff')
        self.fig_anim.patch.set_facecolor('#ffffff')
        self.canvas_anim = FigureCanvas(self.fig_anim)
        self.canvas_anim.setMinimumSize(650, 650)
        anim_container.addWidget(self.canvas_anim)
        self.canvas_anim.mpl_connect("button_press_event", self._on_mouse)

        # Верхняя панель для гистограмм с кнопкой переключения режима
        hist_top_bar = QtWidgets.QHBoxLayout()

        self.hist_title = QtWidgets.QLabel()
        self.hist_title.setStyleSheet("font-size:17pt; font-weight:bold; color:#1a1a1a;")
        hist_top_bar.addWidget(self.hist_title)

        self.btn_switch_display_mode = QtWidgets.QPushButton()
        self.btn_switch_display_mode.clicked.connect(self._switch_display_mode)
        self.btn_switch_display_mode.setFixedSize(170, 28)
        self.btn_switch_display_mode.setStyleSheet("""
            QPushButton {
                font-size:11pt; 
                font-weight:600;
                background: #ab9d98; 
                color: white; 
                border: none; 
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background: #a8897e;
            }
            QPushButton:pressed {
                background: #ab897d;
            }
        """)
        hist_top_bar.addWidget(self.btn_switch_display_mode, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        hist_container.addLayout(hist_top_bar)

        self.fig_hist, axes = plt.subplots(3, 1, figsize=(6, 8))
        for ax in axes:
            ax.tick_params(labelsize=9)
            ax.set_facecolor('#f9f9f9')
        self.ax_histx, self.ax_histy, self.ax_histd = axes
        self.fig_hist.patch.set_facecolor('#ffffff')
        self.fig_hist.tight_layout()
        self.canvas_hist = FigureCanvas(self.fig_hist)
        self.canvas_hist.setMinimumSize(400, 600)
        hist_container.addWidget(self.canvas_hist)

        self.draw_mode = False
        self.poly_points = []

        self._update_polygon_buttons_state()
        self._update_display_mode_button()
        # self._update_system_buttons_texts()

    def _update_reset_buttons(self):
        """Обновление текста кнопок сброса"""
        lang = self.get_lang_cb()
        s = STRINGS[lang.value]

        self.btn_reset_data.setText(s.get("sim.btn.reset_data", "Сбросить"))

        # Добавляем всплывающие подсказки
        self.btn_reset_data.setToolTip(
            s.get("sim.btn.reset_data.tooltip", "Очистить все накопленные данные гистограмм"))

    def _reset_hist_data(self):
        """Сброс накопленных данных, не трогая выбор частиц"""
        # Сбрасываем накопленные данные для всех частиц
        self.accumulated_x = np.array([])
        self.accumulated_y = np.array([])
        self.accumulated_distances = np.array([])
        self.accumulated_dx = np.array([])
        self.accumulated_dy = np.array([])
        self.accumulated_signed = np.array([])

        # Сбрасываем данные для выбранных частиц
        self.selected_particles_data = {
            'x': np.array([]),
            'y': np.array([]),
            'distances': np.array([]),
            'dx': np.array([]),
            'dy': np.array([]),
            'signed_distance': np.array([])
        }

        # Перерисовываем только гистограммы
        self._update_histograms()

    def _reset_data(self):
        """Сброс всех накопленных данных"""
        # Сбрасываем накопленные данные для всех частиц
        self.accumulated_x = np.array([])
        self.accumulated_y = np.array([])
        self.accumulated_distances = np.array([])
        self.accumulated_dx = np.array([])
        self.accumulated_dy = np.array([])
        self.accumulated_signed = np.array([])

        # Сбрасываем данные для выбранных частиц (полная инициализация всех ключей)
        self.selected_particles_data = {
            'x': np.array([]),
            'y': np.array([]),
            'distances': np.array([]),
            'dx': np.array([]),
            'dy': np.array([]),
            'signed_distance': np.array([])
        }

        # Очищаем выбор частиц
        # self.selected_particles = []

        # Удаляем линию расстояния
        if self.distance_line:
            self.distance_line.remove()
            self.distance_line = None

        # Перерисовываем частицы (убираем подсветку)
        self._redraw_particles()

        # Обновляем гистограммы
        self._update_histograms()

    def _on_bins_changed(self, value):
        self.bins_count = value
        self._update_histograms()

    def _switch_display_mode(self):
        """Переключение режима отображения графиков расстояния"""
        self.distance_display_mode = (self.distance_display_mode + 1) % 3
        self._update_display_mode_button()
        self._update_histograms()

    def _update_display_mode_button(self):
        """Обновление текста кнопки переключения режима"""
        lang = self.get_lang_cb()
        s = STRINGS[lang.value]

        if self.distance_display_mode == 0:
            self.btn_switch_display_mode.setText(s.get("sim.display_mode.distance", "Расстояние"))
            self.btn_switch_display_mode.setToolTip("Показать обычное евклидово расстояние между частицами")
        elif self.distance_display_mode == 1:
            self.btn_switch_display_mode.setText(s.get("sim.display_mode.projections", "Проекции X/Y"))
            self.btn_switch_display_mode.setToolTip("Показать проекции расстояния по осям X и Y")
        else:  # mode 2
            self.btn_switch_display_mode.setText(s.get("sim.display_mode.signed", "Расстояние со знаком"))
            self.btn_switch_display_mode.setToolTip("Показать ориентированное расстояние со знаком")

    def _toggle_particle_selection(self):
        """Включение/выключение режима выбора частиц"""
        self.particle_selection_mode = not self.particle_selection_mode
        lang = self.get_lang_cb()
        s = STRINGS[lang.value]

        if self.particle_selection_mode:
            self.btn_particle_select.setText(s.get("sim.btn.cancel_selection", "Отменить выбор"))
            # Очищаем предыдущие данные
            self.selected_particles_data = {
                'x': np.array([]),
                'y': np.array([]),
                'distances': np.array([]),
                'dx': np.array([]),
                'dy': np.array([]),
                'signed_distance': np.array([])
            }
        else:
            self.btn_particle_select.setText(s.get("sim.btn.select_particles", "Выбор частиц"))
            # Очищаем выбор и удаляем линию
            self.selected_particles = []
            if self.distance_line:
                self.distance_line.remove()
                self.distance_line = None
            # Очищаем данные выбранных частиц
            self.selected_particles_data = {
                'x': np.array([]),
                'y': np.array([]),
                'distances': np.array([]),
                'dx': np.array([]),
                'dy': np.array([]),
                'signed_distance': np.array([])
            }

        self._redraw_particles()
        self._update_histograms()

    def _on_settings_changed(self):
        if not (hasattr(self, 'system') and self.system is not None):
            return

        # старые значения
        old_N = self.system.N
        old_temp = self.system.temp
        old_mass = self.system.mass
        old_collisions = self.system.enable_collisions
        old_interact_step = self.system.interaction_step
        old_kind = self.system.params.kind
        old_eps = self.system.params.epsilon
        old_sigma = self.system.params.sigma
        old_De = self.system.params.De
        old_a = self.system.params.a
        old_r0 = self.system.params.r0
        old_vessel_key = self._vessel_kind_to_key(getattr(self.system.vessel, "kind", ""))

        # новые значения из ползунков
        new_N = int(self.spin_N.value())
        new_temp = float(self.edit_T.value()) / 10.0
        new_mass = float(self.edit_m.value()) / 1000.0

        self.system.N = new_N
        self.system.temp = new_temp
        self.system.mass = new_mass
        self.system.enable_collisions = self.check_collisions.isChecked()
        self.system.interaction_step = int(self.spin_interact_step.value())

        self.system.params.kind = str(self.pot_box.currentText())
        self.system.params.epsilon = float(self.edit_eps.value()) / 10.0
        self.system.params.sigma = float(self.edit_sigma.value()) / 10.0
        self.system.params.De = float(self.edit_De.value()) / 10.0
        self.system.params.a = float(self.edit_a.value()) / 100.0
        self.system.params.r0 = float(self.edit_r0.value()) / 10.0

        new_vessel_key = self._vessel_display_to_key(str(self.vessel_box.currentText()))

        self._sanitize_selected_particles()

        if new_N != old_N or new_vessel_key != old_vessel_key:
            reset_hist = (new_vessel_key != old_vessel_key)
            self._reinitialize_system(reset_hist=reset_hist)
            return

        # флаги изменений
        changed_temp = (new_temp != old_temp)
        changed_mass = (new_mass != old_mass)

        other_changed = (
            self.system.enable_collisions != old_collisions or
            self.system.interaction_step != old_interact_step or
            self.system.params.kind != old_kind or
            self.system.params.epsilon != old_eps or
            self.system.params.sigma != old_sigma or
            self.system.params.De != old_De or
            self.system.params.a != old_a or
            self.system.params.r0 != old_r0
        )

        # если изменилось ЧТО-ТО, кроме T и m → сбрасываем и пересчитываем гистограммы
        if other_changed:
            self._reset_hist_data()
            self._update_histograms()

        # если менялись T и/или m — только перескейлим скорости
        if self.system.vel is not None and (changed_temp or changed_mass):
            num = (new_temp / max(new_mass, 1e-12))
            den = (old_temp / max(old_mass, 1e-12))
            if den > 0:
                scale = math.sqrt(num / den)
                self.system.vel *= scale

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

    def _reinitialize_system(self, reset_hist: bool = False):
        """Полная переинициализация системы с текущими настройками."""
        N = int(self.spin_N.value())
        radius = 3
        temp = float(self.edit_T.value()) / 10.0
        dt = 0.1
        mass = float(self.edit_m.value()) / 1000.0
        enable_collisions = self.check_collisions.isChecked()
        interaction_step = int(self.spin_interact_step.value())
        pot_params = PotentialParams(
            kind=str(self.pot_box.currentText()),
            epsilon=float(self.edit_eps.value()) / 10.0,
            sigma=float(self.edit_sigma.value()) / 10.0,
            De=float(self.edit_De.value()) / 10.0,
            a=float(self.edit_a.value()) / 100.0,
            r0=float(self.edit_r0.value()) / 10.0,
        )
        
        vessel_key = self._vessel_display_to_key(str(self.vessel_box.currentText()))
        poly = self.system.vessel.poly if self._vessel_kind_to_key(getattr(self.system.vessel, "kind", "")) in ("poly", 'Многоугольник') else None

        # Сбрасываем накопленные данные при изменении параметров
        self._reset_hist_data()

        self._init_simulation(N=N, radius=radius, temp=temp, dt=dt, vessel_kind=vessel_key, poly=poly,
                            potential_params=pot_params, mass=mass, enable_collisions=enable_collisions,
                            interaction_step=interaction_step)
        
        # Если был включен режим выбора и есть хотя бы 2 частицы, выбираем две случайные
        if self.particle_selection_mode and self.system.n() >= 2:
            self.selected_particles = np.random.choice(self.system.n(), 2, replace=False).tolist()
            self._update_distance_display()
            self._redraw_particles()

    def _update_selected_particles_histograms(self):
        """Обновление гистограмм только для выбранных частиц"""
        if self.system.pos is None or len(self.selected_particles) == 0:
            return

        # Собираем данные выбранных частиц
        selected_x = []
        selected_y = []

        for idx in self.selected_particles:
            if idx < len(self.system.pos):
                selected_x.append(self.system.pos[idx, 0])
                selected_y.append(self.system.pos[idx, 1])
                # Накопление данных для выбранных частиц
                self.selected_particles_data['x'] = np.append(self.selected_particles_data['x'],
                                                              self.system.pos[idx, 0])
                self.selected_particles_data['y'] = np.append(self.selected_particles_data['y'],
                                                              self.system.pos[idx, 1])

        # Добавляем расстояние между выбранными частицами и проекции
        if len(self.selected_particles) == 2:
            pos1 = self.system.pos[self.selected_particles[0]]
            pos2 = self.system.pos[self.selected_particles[1]]

            # Обычное расстояние
            distance = np.linalg.norm(pos1 - pos2)
            self.selected_particles_data['distances'] = np.append(self.selected_particles_data['distances'], distance)

            # Проекции
            dx = pos2[0] - pos1[0]  # проекция на X
            dy = pos2[1] - pos1[1]  # проекция на Y
            self.selected_particles_data['dx'] = np.append(self.selected_particles_data['dx'], dx)
            self.selected_particles_data['dy'] = np.append(self.selected_particles_data['dy'], dy)

            # Расстояние со знаком (ориентированное расстояние)
            # Определяем направление от первой частицы ко второй
            angle = np.arctan2(dy, dx)
            signed_distance = distance * np.sign(np.cos(angle))  # знак зависит от направления по X
            self.selected_particles_data['signed_distance'] = np.append(
                self.selected_particles_data['signed_distance'], signed_distance
            )

        # Ограничиваем размер массивов
        max_points = 1000
        for key in self.selected_particles_data:
            if len(self.selected_particles_data[key]) > max_points:
                self.selected_particles_data[key] = self.selected_particles_data[key][-max_points:]

        # Строим гистограммы для выбранных частиц в зависимости от режима
        lang = self.get_lang_cb()
        s = STRINGS[lang.value]

        if len(self.selected_particles_data['x']) > 0:
            self.ax_histx.hist(self.selected_particles_data['x'], bins=self.bins_count, density=True,
                               color="#8bb7d7", alpha=0.8)
        self.ax_histx.set_ylabel("p(x)", labelpad=5, fontsize=12)
        self.ax_histx.set_title(f"{s['sim.hist.x']}", fontsize=12, fontweight='bold')

        if len(self.selected_particles_data['y']) > 0:
            self.ax_histy.hist(self.selected_particles_data['y'], bins=self.bins_count, density=True,
                               color="#f9ad6c", alpha=0.8)
        self.ax_histy.set_ylabel("p(y)", labelpad=5, fontsize=12)
        self.ax_histy.set_title(f"{s['sim.hist.y']}", fontsize=12, fontweight='bold')

        # Разные режимы отображения для третьего графика
        if self.distance_display_mode == 0:  # Обычное расстояние
            if len(self.selected_particles_data['distances']) > 0:
                self.ax_histd.hist(self.selected_particles_data['distances'], bins=self.bins_count, density=True,
                                   color="#8dd38d", alpha=0.85)
            self.ax_histd.set_ylabel("p(r)", labelpad=5, fontsize=12)
            self.ax_histd.set_xlabel("r", fontsize=12)
            self.ax_histd.set_title(f"{s['sim.hist.r']}", fontsize=12, fontweight='bold')

        elif self.distance_display_mode == 1:  # Проекции
            # Показываем проекции dx и dy на одном графике
            has_dx = len(self.selected_particles_data['dx']) > 0
            has_dy = len(self.selected_particles_data['dy']) > 0

            if has_dx:
                self.ax_histd.hist(self.selected_particles_data['dx'], bins=self.bins_count, density=True,
                                   color="#FF6B6B", alpha=0.6, label="Δx")
            if has_dy:
                self.ax_histd.hist(self.selected_particles_data['dy'], bins=self.bins_count, density=True,
                                   color="#4ECDC4", alpha=0.6, label="Δy")

            self.ax_histd.set_ylabel("p(Δ)", labelpad=5, fontsize=12)
            self.ax_histd.set_xlabel("Δ", fontsize=12)
            self.ax_histd.set_title(s.get("sim.display_mode.projections", "Проекции расстояния"), fontsize=12, fontweight='bold')

            # Добавляем легенду только если есть данные
            if has_dx or has_dy:
                self.ax_histd.legend()

        else:  # Расстояние со знаком
            if len(self.selected_particles_data['signed_distance']) > 0:
                self.ax_histd.hist(self.selected_particles_data['signed_distance'], bins=self.bins_count, density=True,
                                   color="#8d6e63", alpha=0.85)
            self.ax_histd.set_ylabel("p(rₛ)", labelpad=5, fontsize=12)
            self.ax_histd.set_xlabel("rₛ", fontsize=12)
            self.ax_histd.set_title(s.get("sim.display_mode.signed", "Расстояние со знаком"), fontsize=12, fontweight='bold')
            # Добавляем вертикальную линию в нуле
            self.ax_histd.axvline(x=0, color='red', linestyle='--', alpha=0.7)

    def _update_particle_visualization(self):
        """Обновляет визуализацию частиц без пересоздания системы"""
        # удаляем старые арты и заново создаём круги, чтобы избежать рассинхрона по числу частиц
        if hasattr(self, 'particle_circles'):
            for c in self.particle_circles:
                try:
                    c.remove()
                except Exception:
                    pass
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
            # УВЕЛИЧИВАЕМ интервал обновления гистограмм с 1.5 до 2.5 секунд
            if current_time - self._last_hist_update > 1:
                self._update_histograms()
                self._last_hist_update = current_time

    def _redraw_particles(self):
        if hasattr(self, 'particle_circles') and self.system.pos is not None:
            # sanitize selection before use
            self._sanitize_selected_particles()
            if self.system.n() == 0:
                return
            n_circles = len(self.particle_circles)
            n_pos = self.system.n()
            n_iter = min(n_circles, n_pos)
            for i in range(n_iter):
                circle = self.particle_circles[i]
                circle.center = (self.system.pos[i, 0], self.system.pos[i, 1])

                # Подсвечиваем выбранные частицы (всегда, если они выбраны)
                if i in self.selected_particles:
                    circle.set_facecolor("#ff4444")  # Красный для выбранных
                    circle.set_edgecolor("darkred")
                    circle.set_linewidth(2)
                else:
                    circle.set_facecolor("#215a93")  # Синий для остальных
                    circle.set_edgecolor("black")
                    circle.set_linewidth(0.5)

            # Обновляем линию расстояния
            if self.particle_selection_mode and len(self.selected_particles) == 2:
                self._update_distance_display()

            current_time = time.time()
            if current_time - self._last_canvas_update > 0.033:
                self.canvas_anim.draw_idle()
                self._last_canvas_update = current_time
        else:
            # если рассинхрон (нет кругов) — создаём визуализацию заново
            self._update_particle_visualization()

    def _update_histograms(self):
        """Обновление гистограмм с учетом выбранных частиц"""
        # Очищаем графики
        for ax in (self.ax_histx, self.ax_histy, self.ax_histd):
            ax.cla()
            ax.set_facecolor("#f9f9f9")
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.tick_params(labelsize=9)
        # Добавляем проверку на пустую систему
        if self.system is None or self.system.pos is None or self.system.n() == 0:
            self.canvas_hist.draw_idle()
            return
            
        # sanitize selected indices before histogram updates
        self._sanitize_selected_particles()

        lang = self.get_lang_cb()
        s = STRINGS[lang.value]

        # Если в режиме выбора и есть выбранные частицы, показываем только их данные
        if self.particle_selection_mode and len(self.selected_particles) > 0:
            self._update_selected_particles_histograms()
        else:
            # Обычные гистограммы для всех частиц
            self._update_all_particles_histograms()

        self.canvas_hist.draw_idle()

    def _update_selected_particles_histograms(self):
        """Обновление гистограмм только для выбранных частиц"""
        if self.system.pos is None or len(self.selected_particles) == 0:
            return

        # Собираем данные выбранных частиц
        selected_x = []
        selected_y = []

        for idx in self.selected_particles:
            if idx < len(self.system.pos):
                selected_x.append(self.system.pos[idx, 0])
                selected_y.append(self.system.pos[idx, 1])
                # Накопление данных для выбранных частиц
                self.selected_particles_data['x'] = np.append(self.selected_particles_data['x'],
                                                              self.system.pos[idx, 0])
                self.selected_particles_data['y'] = np.append(self.selected_particles_data['y'],
                                                              self.system.pos[idx, 1])

        # Добавляем расстояние между выбранными частицами и проекции
        if len(self.selected_particles) == 2:
            pos1 = self.system.pos[self.selected_particles[0]]
            pos2 = self.system.pos[self.selected_particles[1]]

            # Обычное расстояние
            distance = np.linalg.norm(pos1 - pos2)
            self.selected_particles_data['distances'] = np.append(self.selected_particles_data['distances'], distance)

            # Проекции
            dx = pos2[0] - pos1[0]  # проекция на X
            dy = pos2[1] - pos1[1]  # проекция на Y
            self.selected_particles_data['dx'] = np.append(self.selected_particles_data['dx'], dx)
            self.selected_particles_data['dy'] = np.append(self.selected_particles_data['dy'], dy)

            # Расстояние со знаком (ориентированное расстояние)
            # Определяем направление от первой частицы ко второй
            angle = np.arctan2(dy, dx)
            signed_distance = distance * np.sign(np.cos(angle))  # знак зависит от направления по X
            self.selected_particles_data['signed_distance'] = np.append(
                self.selected_particles_data['signed_distance'], signed_distance
            )

        # Ограничиваем размер массивов
        max_points = 1000
        for key in self.selected_particles_data:
            if len(self.selected_particles_data[key]) > max_points:
                self.selected_particles_data[key] = self.selected_particles_data[key][-max_points:]

        # Строим гистограммы для выбранных частиц в зависимости от режима
        lang = self.get_lang_cb()
        s = STRINGS[lang.value]

        if len(self.selected_particles_data['x']) > 0:
            self.ax_histx.hist(self.selected_particles_data['x'], bins=self.bins_count, density=True,
                               color="#8bb7d7", alpha=0.8)
        self.ax_histx.set_ylabel("p(x)", labelpad=5, fontsize=12)
        self.ax_histx.set_title(f"{s['sim.hist.x']}", fontsize=12, fontweight='bold')

        if len(self.selected_particles_data['y']) > 0:
            self.ax_histy.hist(self.selected_particles_data['y'], bins=self.bins_count, density=True,
                               color="#f9ad6c", alpha=0.8)
        self.ax_histy.set_ylabel("p(y)", labelpad=5, fontsize=12)
        self.ax_histy.set_title(f"{s['sim.hist.y']}", fontsize=12, fontweight='bold')

        # Разные режимы отображения для третьего графика
        if self.distance_display_mode == 0:  # Обычное расстояние
            if len(self.selected_particles_data['distances']) > 0:
                self.ax_histd.hist(self.selected_particles_data['distances'], bins=self.bins_count, density=True,
                                   color="#8dd38d", alpha=0.85)
            self.ax_histd.set_ylabel("p(r)", labelpad=5, fontsize=12)
            self.ax_histd.set_xlabel("r", fontsize=12)
            self.ax_histd.set_title(f"{s['sim.hist.r']}", fontsize=12, fontweight='bold')

        elif self.distance_display_mode == 1:  # Проекции
            # Показываем проекции dx и dy на одном графике
            has_dx = len(self.selected_particles_data['dx']) > 0
            has_dy = len(self.selected_particles_data['dy']) > 0

            if has_dx:
                self.ax_histd.hist(self.selected_particles_data['dx'], bins=self.bins_count, density=True,
                                   color="#FF6B6B", alpha=0.6, label="Δx")
            if has_dy:
                self.ax_histd.hist(self.selected_particles_data['dy'], bins=self.bins_count, density=True,
                                   color="#4ECDC4", alpha=0.6, label="Δy")

            self.ax_histd.set_ylabel("p(Δ)", labelpad=5, fontsize=12)
            self.ax_histd.set_xlabel("Δ", fontsize=12)
            self.ax_histd.set_title(s.get("sim.display_mode.projections", "Проекции расстояния"), fontsize=12, fontweight='bold')

            # Добавляем легенду только если есть данные
            if has_dx or has_dy:
                self.ax_histd.legend()

        else:  # Расстояние со знаком
            if len(self.selected_particles_data['signed_distance']) > 0:
                self.ax_histd.hist(self.selected_particles_data['signed_distance'], bins=self.bins_count, density=True,
                                   color="#8d6e63", alpha=0.85)
            self.ax_histd.set_ylabel("p(rₛ)", labelpad=5, fontsize=12)
            self.ax_histd.set_xlabel("rₛ", fontsize=12)
            self.ax_histd.set_title(s.get("sim.display_mode.signed", "Расстояние со знаком"), fontsize=12, fontweight='bold')
            # Добавляем вертикальную линию в нуле
            self.ax_histd.axvline(x=0, color='red', linestyle='--', alpha=0.7)

    def _update_all_particles_histograms(self):
        """Обновление гистограмм для всех частиц"""
        # sanitize selected indices to avoid OOB access when N changed rapidly
        self._sanitize_selected_particles()
        # Накопление данных о текущих позициях
        current_x = self.system.pos[:, 0]
        current_y = self.system.pos[:, 1]

        # Вычисляем разные типы расстояний в зависимости от режима
        if self.distance_display_mode == 0:  # Обычное расстояние
            current_dists = self.system.pairwise_distances_fast(max_pairs=5000)
            if current_dists.size > 0:
                self.accumulated_distances = np.concatenate([self.accumulated_distances, current_dists])

        elif self.distance_display_mode == 1:  # Проекции
            current_dx, current_dy = self._compute_pairwise_projections_fast(max_pairs=5000)
            if current_dx.size > 0:
                self.accumulated_dx = np.concatenate([self.accumulated_dx, current_dx])
            if current_dy.size > 0:
                self.accumulated_dy = np.concatenate([self.accumulated_dy, current_dy])

        else:  # Расстояние со знаком
            current_signed = self._compute_pairwise_signed_fast(max_pairs=5000)
            if current_signed.size > 0:
                self.accumulated_signed = np.concatenate([self.accumulated_signed, current_signed])

        # Добавляем текущие данные к накопленным
        self.accumulated_x = np.concatenate([self.accumulated_x, current_x])
        self.accumulated_y = np.concatenate([self.accumulated_y, current_y])

        # Ограничиваем размер массивов для предотвращения переполнения памяти
        if len(self.accumulated_x) > self.max_accumulated_points:
            self.accumulated_x = self.accumulated_x[-self.max_accumulated_points:]
        if len(self.accumulated_y) > self.max_accumulated_points:
            self.accumulated_y = self.accumulated_y[-self.max_accumulated_points:]
        if len(self.accumulated_distances) > self.max_accumulated_points:
            self.accumulated_distances = self.accumulated_distances[-self.max_accumulated_points:]
        if len(self.accumulated_dx) > self.max_accumulated_points:
            self.accumulated_dx = self.accumulated_dx[-self.max_accumulated_points:]
        if len(self.accumulated_dy) > self.max_accumulated_points:
            self.accumulated_dy = self.accumulated_dy[-self.max_accumulated_points:]
        if len(self.accumulated_signed) > self.max_accumulated_points:
            self.accumulated_signed = self.accumulated_signed[-self.max_accumulated_points:]

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

        # Разные режимы отображения для третьего графика
        if self.distance_display_mode == 0:  # Обычное расстояние
            if len(self.accumulated_distances) > 0:
                self.ax_histd.hist(self.accumulated_distances, bins=self.bins_count, density=True, color="#8dd38d",
                                   alpha=0.85)
            self.ax_histd.set_ylabel("p(r)", labelpad=5)
            self.ax_histd.set_xlabel("r")
            self.ax_histd.set_title(s["sim.hist.r"], fontsize=10, fontweight='bold')

        elif self.distance_display_mode == 1:  # Проекции
            # Показываем проекции dx и dy на одном графике
            has_dx = len(self.accumulated_dx) > 0
            has_dy = len(self.accumulated_dy) > 0

            if has_dx:
                self.ax_histd.hist(self.accumulated_dx, bins=self.bins_count, density=True,
                                   color="#FF6B6B", alpha=0.6, label="Δx")
            if has_dy:
                self.ax_histd.hist(self.accumulated_dy, bins=self.bins_count, density=True,
                                   color="#4ECDC4", alpha=0.6, label="Δy")

            self.ax_histd.set_ylabel("p(Δ)", labelpad=5)
            self.ax_histd.set_xlabel("Δ")
            self.ax_histd.set_title(s.get("sim.display_mode.projections", "Проекции расстояния"), fontsize=10, fontweight='bold')

            # Добавляем легенду только если есть данные
            if has_dx or has_dy:
                self.ax_histd.legend()

        else:  # Расстояние со знаком
            if len(self.accumulated_signed) > 0:
                self.ax_histd.hist(self.accumulated_signed, bins=self.bins_count, density=True,
                                   color="#8d6e63", alpha=0.85)
            self.ax_histd.set_ylabel("p(rₛ)", labelpad=5)
            self.ax_histd.set_xlabel("rₛ")
            self.ax_histd.set_title(s.get("sim.display_mode.signed", "Расстояние со знаком"), fontsize=10, fontweight='bold')
            # Добавляем вертикальную линию в нуле
            self.ax_histd.axvline(x=0, color='red', linestyle='--', alpha=0.7)

    def _compute_pairwise_projections_fast(self, max_pairs=5000):
        """Вычисляет проекции расстояний для случайной выборки пар частиц"""
        N = self.system.n()
        if N < 2:
            return np.array([]), np.array([])

        pos = self.system.pos
        dx_list = np.zeros(max_pairs)
        dy_list = np.zeros(max_pairs)
        count = 0

        for _ in range(max_pairs * 2):
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            if i != j:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dx_list[count] = dx
                dy_list[count] = dy
                count += 1
                if count >= max_pairs:
                    break

        return dx_list[:count], dy_list[:count]

    def _compute_pairwise_signed_fast(self, max_pairs=5000):
        """Вычисляет расстояния со знаком для случайной выборки пар частиц"""
        N = self.system.n()
        if N < 2:
            return np.array([])

        pos = self.system.pos
        signed_list = np.zeros(max_pairs)
        count = 0

        for _ in range(max_pairs * 2):
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            if i != j:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                distance = math.sqrt(dx * dx + dy * dy)
                # Знак зависит от направления по X
                signed_distance = distance * (1 if dx >= 0 else -1)
                signed_list[count] = signed_distance
                count += 1
                if count >= max_pairs:
                    break

        return signed_list[:count]

    def _compute_quick_distances_sample(self, sample_size):
        """Быстрое вычисление случайной выборки расстояний"""
        N = self.system.n()
        if N < 2:
            return np.array([])

        pos = self.system.pos
        distances = np.zeros(sample_size)

        for i in range(sample_size):
            # Берем две случайные различные частицы
            idx1, idx2 = np.random.choice(N, 2, replace=False)
            dx = pos[idx2, 0] - pos[idx1, 0]
            dy = pos[idx2, 1] - pos[idx1, 1]
            distances[i] = math.sqrt(dx * dx + dy * dy)

        return distances

    def _compute_sampled_distances(self, sample_size):
        """Вычисляет расстояния для случайной выборка пар частиц"""
        N = self.system.n()
        if N < 2:
            return np.array([])

        pos = self.system.pos
        distances = np.zeros(sample_size)
        count = 0

        for _ in range(sample_size * 2):  # Даем немного больше попыток
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            if i != j:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                distances[count] = math.sqrt(dx * dx + dy * dy)
                count += 1
                if count >= sample_size:
                    break

        return distances[:count]

    def _accumulate_with_limit(self, accumulated, new_data):
        """Добавляет новые данные, ограничивая общий размер"""
        if len(accumulated) + len(new_data) > self.max_accumulated_points:
            # Удаляем старые данные, чтобы освободить место
            keep_count = self.max_accumulated_points - len(new_data)
            if keep_count > 0:
                accumulated = accumulated[-keep_count:]
            else:
                accumulated = np.array([])

        return np.concatenate([accumulated, new_data]) if accumulated.size > 0 else new_data.copy()

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
        else:
            self.settings_panel.show()

    def _enter_draw_mode(self):
        # проверяем по каноническому ключу текущий выбранный сосуд
        v_k = self._vessel_display_to_key(str(self.vessel_box.currentText()))
        if v_k in ('poly', 'Многоугольник'):
            self.draw_mode = True
            self.poly_points = []

    def _clear_poly(self):
        # очищаем полигон только если мы в режиме poly (канонический ключ)
        if self._vessel_display_to_key(str(self.vessel_box.currentText())) in ('poly', 'Многоугольник'):
            self.system.vessel.poly = None
            # сохраняем kind как poly/rect/circle — не меняем здесь
            self._draw_vessel_patch()
            self.canvas_anim.draw_idle()

    def _on_mouse(self, event):
        if event.inaxes != self.ax_anim:
            return

        # Если включен режим выбора частиц, обрабатываем выбор
        if self.particle_selection_mode and event.button == 1:
            self._select_particle(event)

        # Остальная логика для рисования полигонов
        v_k = self._vessel_display_to_key(str(self.vessel_box.currentText()))
        if not self.draw_mode or v_k != "poly":
            return

        if event.button == 1:
            self.poly_points.append((event.xdata, event.ydata))
            self._preview_poly()
        elif event.button == 3:
            if len(self.poly_points) >= 3:
                self.system.vessel.poly = np.array(self.poly_points)
                # сохраняем канонический тип у сосудa
                self.system.vessel.kind = "poly"
                self.draw_mode = False
                self.poly_points = []
                self._draw_vessel_patch()
                self.system.seed()
                self._redraw_particles()

    def _select_particle(self, event):
        """Выбор частицы для измерения расстояния"""
        if self.system.pos is None or len(self.system.pos) == 0:
            return

        # Находим ближайшую частицу к точке клика
        click_pos = np.array([event.xdata, event.ydata])
        distances = np.linalg.norm(self.system.pos - click_pos, axis=1)
        closest_idx = np.argmin(distances)

        # Если частица достаточно близко к клику
        if distances[closest_idx] < self.system.visual_radius * 3:
            if closest_idx in self.selected_particles:
                # Убираем частицу из выбора
                self.selected_particles.remove(closest_idx)
            else:
                # Добавляем частицу в выбор (максимум 2)
                if len(self.selected_particles) < 2:
                    self.selected_particles.append(closest_idx)

            self._update_distance_display()
            self._redraw_particles()

    def _update_distance_display(self):
        """Обновление отображения расстояния между выбранными частицами"""
        # Удаляем старую линию
        if self.distance_line:
            self.distance_line.remove()
            self.distance_line = None

        # Рисуем новую линию, если выбрано 2 частицы
        if len(self.selected_particles) == 2 and self.system.pos is not None:
            pos1 = self.system.pos[self.selected_particles[0]]
            pos2 = self.system.pos[self.selected_particles[1]]

            # Создаем линию между частицами
            self.distance_line = patches.ConnectionPatch(
                pos1, pos2, 'data', 'data',
                arrowstyle='-', color='red', linewidth=2, alpha=0.7
            )
            self.ax_anim.add_patch(self.distance_line)

            # Добавляем информацию о расстоянии в зависимости от режима
            distance = np.linalg.norm(pos1 - pos2)
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            mid_point = (pos1 + pos2) / 2

            if self.distance_display_mode == 0:
                text = f'r = {distance:.1f}'
            elif self.distance_display_mode == 1:
                text = f'Δx = {dx:.1f}\nΔy = {dy:.1f}'
            else:  # mode 2
                signed_dist = distance * np.sign(dx)
                text = f'rₛ = {signed_dist:.1f}'

            # Отображаем текст с информацией
            # self.ax_anim.text(mid_point[0], mid_point[1], text,
            #                  fontsize=9, color='red', ha='center', va='center',
            #                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        self.canvas_anim.draw_idle()

        self.canvas_anim.draw_idle()

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
        # приводим kind к каноническому ключу
        vk = self._vessel_kind_to_key(getattr(v, "kind", ""))
        if vk == "rect":
            xmin, ymin, xmax, ymax = v.rect
            self.vessel_artist = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, lw=2, ec="#444")
            self.ax_anim.set_xlim(xmin - 10, xmax + 10)
            self.ax_anim.set_ylim(ymin - 10, ymax + 10)
        elif vk == "circle":
            cx, cy, R = v.circle
            self.vessel_artist = patches.Circle((cx, cy), R, fill=False, lw=2, ec="#444")
            self.ax_anim.set_xlim(cx - R - 10, cx + R + 10)
            self.ax_anim.set_ylim(cy - R - 10, cy + R + 10)
        elif vk in ('poly', 'Многоугольник') and v.poly is not None:
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

        self.hist_title.setText(s.get("sim.histograms.title", "Гистограммы"))

        # Обновляем кнопку переключения режима
        self._update_display_mode_button()
        self._update_reset_buttons()

        if hasattr(self, 'btn_particle_select'):
            if self.particle_selection_mode:
                self.btn_particle_select.setText(s.get("sim.btn.cancel_selection", "Отменить выбор"))
            else:
                self.btn_particle_select.setText(s.get("sim.btn.select_particles", "Выбор частиц"))

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
        set_lbl("lbl_bins", "sim.bins")
        set_lbl("help_bins", "sim.bins.help")

        self.btn_draw.setText(s["sim.btn.draw"])
        self.btn_clear.setText(s["sim.btn.clear"])
        self.btn_run.setText(s["sim.btn.pause"] if self.running else s["sim.btn.start"])
        self.btn_back.setText(s["sim.btn.back"])
        self.btn_toggle_settings_small.setText(s["sim.top.settings"])
        self.ax_anim.set_title(s["sim.anim.title"])
        self.canvas_anim.draw_idle()

        # Обновляем названия потенциалов
        current_index = max(0, self.pot_box.currentIndex())
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
        current_vessel_index = max(0, self.vessel_box.currentIndex())
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
        left.setContentsMargins(2, 0, 2, 0)  # 🔹 убрали внутренние поля

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
        right.setContentsMargins(2, 0, 2, 0)  # 🔹 убрали внутренние поля

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
            authors_cb=self.show_authors,
            theory_cb=self.open_theory_pdf  # ← добавляем колбэк для теории
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

    def open_theory_pdf(self):
        """Открывает PDF-файл с теорией"""
        try:
            pdf_path = os.path.join(os.path.dirname(__file__), "theory.pdf")
            if os.path.exists(pdf_path):
                if sys.platform == "win32":
                    os.startfile(pdf_path)
                elif sys.platform == "darwin":  # macOS
                    os.system(f'open "{pdf_path}"')
                else:  # linux
                    os.system(f'xdg-open "{pdf_path}"')
            else:
                # Если файл не найден, показываем сообщение
                msg = QtWidgets.QMessageBox(self)
                msg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
                msg.setWindowTitle("Файл не найден")
                msg.setText("Файл theory.pdf не найден в папке с программой.")
                msg.exec()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Не удалось открыть файл теории: {str(e)}")

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
            font = QFont(font_family, 11)  # УВЕЛИЧИЛИ БАЗОВЫЙ РАЗМЕР ШРИФТА
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
