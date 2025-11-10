# Improved and fixed version of the particle-distance simulation GUI
# - dt is now fixed (0.2) and not user-editable
# - field 'm' added to settings for particle mass

import sys
import os
import time
from dataclasses import dataclass, field
import numpy as np
from PIL import Image, ImageQt
from math import sqrt, exp

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib import patches

from numba import jit, njit, prange
import numba as nb
import math


# -------------------- Physics helpers (numpy, simple, robust) --------------------
@dataclass
class PotentialParams:
    kind: str = "Нет"
    k: float = 1.0  # Верните нормальное значение
    epsilon: float = 1.0
    sigma: float = 0.02
    De: float = 1.0
    a: float = 2.0
    r0: float = 0.025
    rcut_lr: float = 0.3  # Уменьшено, но не так радикально


@dataclass
class Vessel:
    kind: str = "Прямоугольник"  # internal kinds: 'rect','circle','poly'
    rect: tuple = (-1.0, -1.0, 1.0, 1.0)
    circle: tuple = (0.0, 0.0, 1.0)
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
        """ Возвращает границы сосуда с optional margin.
          Args: margin: дополнительный отступ от границ (например, радиус частицы)
         """ 
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
            return (-1 + margin, -1 + margin, 1 - margin, 1 - margin)


@njit(fastmath=True, cache=True)
def compute_forces_numba(pos, params_kind, k, epsilon, sigma, De, a, r0, rcut_lr, mass):
    N = pos.shape[0]
    F = np.zeros((N, 2), dtype=np.float64)

 # Предварительные вычисления
    rcut = 0.0
    if params_kind == 3:  # Леннард-Джонс
        rcut = 2.5 * max(sigma, 1e-6)
    elif params_kind == 4:  # Морзе
        rcut = r0 + max(3.0 / a, 0.05)
    elif params_kind in (1, 2):  # Отталкивание/Притяжение
        rcut = rcut_lr  # Теперь это разумное значение ~0.2

    rcut2 = rcut * rcut if rcut > 0 else 1e10
    rmin = 0.1 * sigma if sigma > 0 else 1e-4

    for i in prange(N):
        xi, yi = pos[i]
        for j in range(i + 1, N):
            xj, yj = pos[j]
            dx = xj - xi
            dy = yj - yi
            r2 = dx * dx + dy * dy

            if r2 > rcut2 or r2 < 1e-12:
                continue

            r = math.sqrt(r2)
            if r < rmin:
                r = rmin
                r2 = r * r

            invr = 1.0 / r
            nx = dx * invr
            ny = dy * invr

            mag = 0.0
            if params_kind == 1:  # Отталкивание
                mag = k / r2
                # УДАЛЕНО ограничение силы - теперь не нужно
            elif params_kind == 2:  # Притяжение
                mag = -k / r2
                # УДАЛЕНО ограничение силы
            elif params_kind == 3:  # Леннард-Джонс
                sr = sigma / r
                sr6 = sr * sr * sr * sr * sr * sr
                sr12 = sr6 * sr6
                mag = 24.0 * epsilon * (2.0 * sr12 - sr6) / r
            elif params_kind == 4:  # Морзе
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
        for k in range(max_pairs * 2):
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
            v[0] = abs(v[0])
            p[0] = xmin + radius
        if p[0] > xmax - radius:
            v[0] = -abs(v[0])
            p[0] = xmax - radius
        if p[1] < ymin + radius:
            v[1] = abs(v[1])
            p[1] = ymin + radius
        if p[1] > ymax - radius:
            v[1] = -abs(v[1])
            p[1] = ymax - radius

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
                dot = v[0] * nx + v[1] * ny
                v[0] -= 2.0 * dot * nx
                v[1] -= 2.0 * dot * ny
                p[0] = cx + nx * R_eff
                p[1] = cy + ny * R_eff

        pos[i] = p
        vel[i] = v


@njit(fastmath=True, cache=True)
def reflect_polygonal_numba(pos, vel, radius, poly):
    """
    Отражает частицы от границ полигона с учётом радиуса.
    Упрощённая и исправленная версия.
    """
    N = pos.shape[0]
    M = poly.shape[0]
    
    for i in range(N):
        p = pos[i]
        v = vel[i]
        
        # Проверяем все рёбра полигона
        for a in range(M):
            b = (a + 1) % M
            
            # Координаты начала и конца ребра
            x1, y1 = poly[a]
            x2, y2 = poly[b]
            
            # Вектор ребра
            edge_x = x2 - x1
            edge_y = y2 - y1
            edge_length = math.sqrt(edge_x * edge_x + edge_y * edge_y)
            
            if edge_length < 1e-12:
                continue
                
            # Нормализованный вектор ребра
            edge_x /= edge_length
            edge_y /= edge_length
            
            # Нормаль к ребру (направлена ВНУТРЬ полигона)
            # Для правильного полигона (вершины по часовой стрелке) нормаль направлена внутрь
            normal_x = -edge_y
            normal_y = edge_x
            
            # Вектор от начала ребра к частице
            to_particle_x = p[0] - x1
            to_particle_y = p[1] - y1
            
            # Расстояние от частицы до ребра (со знаком)
            distance = to_particle_x * normal_x + to_particle_y * normal_y
            
            # Если частица вышла за границу (учитывая радиус)
            if distance < radius:
                # Проекция точки на ребро
                t = (to_particle_x * edge_x + to_particle_y * edge_y) / edge_length
                t = min(max(t, 0.0), 1.0)
                
                # Ближайшая точка на ребре
                closest_x = x1 + t * (x2 - x1)
                closest_y = y1 + t * (y2 - y1)
                
                # Вектор от ближайшей точки к частице
                dx = p[0] - closest_x
                dy = p[1] - closest_y
                dist_to_edge = math.sqrt(dx * dx + dy * dy)
                
                if dist_to_edge < 1e-12:
                    # Если частица точно на ребре, используем вычисленную нормаль
                    norm_length = math.sqrt(normal_x * normal_x + normal_y * normal_y)
                    if norm_length > 1e-12:
                        normal_x /= norm_length
                        normal_y /= norm_length
                else:
                    # Нормаль от ребра к частице
                    normal_x = dx / dist_to_edge
                    normal_y = dy / dist_to_edge
                
                # Отражение скорости
                dot = v[0] * normal_x + v[1] * normal_y
                if dot < 0:  # Только если движется наружу
                    v[0] -= 2.0 * dot * normal_x
                    v[1] -= 2.0 * dot * normal_y
                
                # Сдвигаем частицу внутрь
                overlap = radius - distance
                p[0] += overlap * normal_x
                p[1] += overlap * normal_y
        
        pos[i] = p
        vel[i] = v


@dataclass
class System:
    vessel: Vessel
    N: int = 50
    radius: float = 0.015
    temp: float = 0.5
    dt: float = 0.002  # fixed integration timestep (user cannot change)
    params: PotentialParams = field(default_factory=PotentialParams)
    mass: float = 1.0
    friction_gamma: float = 0.0

    pos: np.ndarray | None = None
    vel: np.ndarray | None = None
    F: np.ndarray | None = None
    F_old: np.ndarray | None = None

    def n(self):
        return 0 if self.pos is None else self.pos.shape[0]

    def seed(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # Инициализация массивов
        self.pos = np.zeros((self.N, 2), dtype=np.float64)
        self.vel = np.zeros((self.N, 2), dtype=np.float64)
        self.F = np.zeros((self.N, 2), dtype=np.float64)
        self.F_old = np.zeros((self.N, 2), dtype=np.float64)

        # Границы сосуда с учётом радиуса частиц
        xmin, ymin, xmax, ymax = self.vessel.bounds(margin=self.radius)

        for i in range(self.N):
            for attempt in range(1000):
                # Прямоугольник
                if self.vessel.kind in ("rect", "Прямоугольник"):
                    p = np.array([
                        rng.uniform(xmin + self.radius, xmax - self.radius),
                        rng.uniform(ymin + self.radius, ymax - self.radius)
                    ])
                # Круг
                elif self.vessel.kind in ("circle", "Круг"):
                    cx, cy, R = self.vessel.circle
                    ang = rng.uniform(0, 2 * np.pi)
                    rad = math.sqrt(rng.uniform(0, (R - self.radius) ** 2))  # равномерно по площади
                    p = np.array([cx, cy]) + rad * np.array([math.cos(ang), math.sin(ang)])
                # Многоугольник (пока как прямоугольник, можно улучшить later)
                else:
                    p = np.array([
                        rng.uniform(xmin + self.radius, xmax - self.radius),
                        rng.uniform(ymin + self.radius, ymax - self.radius)
                    ])

                # Проверка, что точка внутри сосуда
                if self.vessel.contains(p):
                    # Проверка на пересечение с другими частицами
                    if i == 0 or np.all(np.linalg.norm(self.pos[:i] - p, axis=1) >= 2 * self.radius):
                        self.pos[i] = p
                        break
            else:
                # Если не удалось разместить после 1000 попыток — ставим в центр
                self.pos[i] = np.array([0.0, 0.0])
                print(f"Warning: Particle {i} placed at center due to placement conflicts.")

        # Инициализация скоростей с нормальным распределением
        sigma_v = math.sqrt(self.temp / max(self.mass, 1e-12))
        self.vel = rng.normal(0.0, sigma_v, size=(self.N, 2))

        # Отладочный вывод
        print(f"Seeded system: N = {self.N}, pos[0] = {self.pos[0]}, vel[0] = {self.vel[0]}")


    def compute_forces(self):
        if self.pos is None:
            return np.zeros((0, 2))

        kind_code = self._potential_kind_code()
        return compute_forces_numba(
            self.pos, kind_code, self.params.k, self.params.epsilon,
            self.params.sigma, self.params.De, self.params.a, self.params.r0,
            self.params.rcut_lr, self.mass
        )

    def integrate(self):
        if self.pos is None:
            print("Integrate: pos is None")
            return

        N = self.n()
        F_new = self.compute_forces()

        # 1. Вычисляем промежуточную скорость и позицию
        vel_half = self.vel + 0.5 * (F_new + self.F_old) / self.mass * self.dt
        pos_new = self.pos + vel_half * self.dt

        # 2. Обрабатываем столкновения со стенками
        if self.vessel.kind in ("rect", "Прямоугольник"):
            xmin, ymin, xmax, ymax = self.vessel.rect
            collisions = reflect_rectangular_numba(pos_new, vel_half, self.radius, xmin, ymin, xmax, ymax)
        elif self.vessel.kind in ("circle", "Круг"):
            cx, cy, R = self.vessel.circle
            collisions = reflect_circular_numba(pos_new, vel_half, self.radius, cx, cy, R)
        elif self.vessel.kind in ("poly", "Многоугольник") and self.vessel.poly is not None:
            collisions = reflect_polygonal_numba(pos_new, vel_half, self.radius, self.vessel.poly)
        else:
            xmin, ymin, xmax, ymax = self.vessel.bounds()
            collisions = reflect_rectangular_numba(pos_new, vel_half, self.radius, xmin, ymin, xmax, ymax)

        # 3. Если есть столкновения, пересчитываем позицию
        # if collisions > 0:
            # Пересчитываем позицию с учетом отраженной скорости
            # pos_new = self.pos + vel_half * self.dt

        # 4. Демпфирование
        if self.friction_gamma > 0:
            damp = math.exp(-self.friction_gamma * self.dt)
            vel_new = vel_half * damp
        else:
            vel_new = vel_half

        self.F_old[:] = F_new
        self.pos[:] = pos_new
        self.vel[:] = vel_new

    def pairwise_distances_fast(self, max_pairs=10000):
        if self.pos is None or self.n() < 2:
            return np.array([])
        return pairwise_distances_fast_numba(self.pos, max_pairs)

    def _rcut(self):
        p = self.params
        if p.kind == "Леннард-Джонс":
            return 2.5 * max(p.sigma, 1e-6)
        if p.kind == "Морзе":
            return p.r0 + max(3.0 / p.a, 0.05)
        if p.kind in ("Отталкивание", "Притяжение"):
            return p.rcut_lr
        return 0.0

    def _potential_kind_code(self):
        kinds = {"Нет": 0, "Отталкивание": 1, "Притяжение": 2, "Леннард-Джонс": 3, "Морзе": 4}
        return kinds.get(self.params.kind, 0)


# -------------------- GUI (PyQt) --------------------
class MainMenuWidget(QtWidgets.QWidget):
    def __init__(self, parent, start_cb, authors_cb):
        super().__init__(parent)
        self.start_cb = start_cb
        self.authors_cb = authors_cb
        self._build_ui()

    def _load_logo(self, filename, size=(180, 180)):
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
        right = QtWidgets.QVBoxLayout()
        center = QtWidgets.QVBoxLayout()
        layout.addLayout(left)
        layout.addLayout(center, stretch=1)
        layout.addLayout(right)

        left_logo = QtWidgets.QLabel()
        pix = self._load_logo("cmc_logo.png")
        if pix: left_logo.setPixmap(pix)
        left.addWidget(left_logo, alignment=QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)

        right_logo = QtWidgets.QLabel()
        pix2 = self._load_logo("fiz_logo.png") or self._load_logo("fiz_logo.jpg")
        if pix2: right_logo.setPixmap(pix2)
        right.addWidget(right_logo, alignment=QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        title = QtWidgets.QLabel("Распределение расстояний между частицами")
        title.setStyleSheet("font-size:30pt; font-weight:600;")
        subtitle = QtWidgets.QLabel("в сосудах различной формы")
        subtitle.setStyleSheet("font-size:20pt; color: #444")
        center.addStretch(1)
        center.addWidget(title, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        center.addWidget(subtitle, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        btn_start = QtWidgets.QPushButton("Начать симуляцию")
        btn_start.setFixedSize(240, 50)
        btn_start.setStyleSheet(
            "background:#3271a8; color:white; font-weight:600; font-size:12pt; border-radius:6px; margin-top:10px;")
        btn_start.clicked.connect(self.start_cb)
        center.addWidget(btn_start, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        btn_authors = QtWidgets.QPushButton("Об авторах")
        btn_authors.setFixedSize(240, 50)
        btn_authors.setStyleSheet(
            "background:#a6b2bd; color:white; font-weight:600; font-size:12pt; border-radius:6px; margin-top:10px;")
        btn_authors.clicked.connect(self.authors_cb)
        center.addWidget(btn_authors, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        btn_exit = QtWidgets.QPushButton("Выход")
        btn_exit.setFixedSize(240, 50)
        btn_exit.setStyleSheet(
            "background:#f78765; color:white; font-weight:600; font-size:12pt; border-radius:6px; margin-top:10px;")
        btn_exit.clicked.connect(QtWidgets.QApplication.instance().quit)
        center.addWidget(btn_exit, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        center.addStretch(2)


class AuthorsWidget(QtWidgets.QWidget):
    def __init__(self, parent, back_cb):
        super().__init__(parent)
        self.back_cb = back_cb
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
        title = QtWidgets.QLabel("Команда разработчиков")
        title.setStyleSheet("font-size:24pt; font-weight:700;")
        layout.addWidget(title, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        cards = QtWidgets.QHBoxLayout()
        layout.addLayout(cards)

        left = QtWidgets.QVBoxLayout()
        pix = self._load_sticker("author1.png") or self._load_sticker("author1.jpg")
        if pix:
            lbl = QtWidgets.QLabel()
            lbl.setPixmap(pix)
            left.addWidget(lbl, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        name = QtWidgets.QLabel("Енягин Станислав")
        name.setStyleSheet("font-size:16pt; font-weight:600;")
        left.addWidget(name, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        cards.addLayout(left)

        right = QtWidgets.QVBoxLayout()
        pix2 = self._load_sticker("author2.png") or self._load_sticker("author2.jpg")
        if pix2:
            lbl2 = QtWidgets.QLabel()
            lbl2.setPixmap(pix2)
            right.addWidget(lbl2, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        name2 = QtWidgets.QLabel("Кожемякова Елизавета")
        name2.setStyleSheet("font-size:16pt; font-weight:600;")
        right.addWidget(name2, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        cards.addLayout(right)

        layout.addStretch(1)
        back = QtWidgets.QPushButton("Вернуться")
        back.setFixedSize(200, 48)
        back.setStyleSheet("background:#313132;color:white;font-weight:600;border-radius:6px;")
        back.clicked.connect(self.back_cb)
        layout.addWidget(back, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(20)


class SimulationWidget(QtWidgets.QWidget):
    def __init__(self, parent, back_cb):
        super().__init__(parent)
        self.back_cb = back_cb
        self._build_ui()           # 1. Сначала строим UI
        self._init_simulation()    # 2. Потом инициализируем систему и scatter
        self._start_timer()


    def _build_ui(self):
        main_l = QtWidgets.QHBoxLayout(self)
        self.settings_panel = QtWidgets.QWidget()
        self.settings_panel.setFixedWidth(300)
        self.settings_panel.setStyleSheet("background:#f7f7f8;")
        sp_layout = QtWidgets.QVBoxLayout(self.settings_panel)
        sp_layout.setContentsMargins(12, 12, 12, 12)
        title = QtWidgets.QLabel("Настройки симуляции")
        title.setStyleSheet("font-size:14pt; font-weight:700;")
        sp_layout.addWidget(title)
        sp_layout.addSpacing(6)

        def add_row(label, widget, unit="", help_text=""):
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label)
            lbl.setFixedWidth(80)
            if help_text:
                h = QtWidgets.QLabel(help_text)
                h.setWordWrap(True)
                h.setStyleSheet("color:#555;font-size:9pt;margin-top:2px;")
                sp_layout.addWidget(h)
            row.addWidget(lbl)
            row.addWidget(widget)
            if unit:
                u = QtWidgets.QLabel(unit)
                u.setFixedWidth(80)
                row.addWidget(u)
            sp_layout.addLayout(row)

        self.spin_N = QtWidgets.QSpinBox()
        self.spin_N.setRange(5, 500)
        self.spin_N.setValue(100)
        add_row("N", self.spin_N, "шт", "Число частиц")
        self.edit_R = QtWidgets.QDoubleSpinBox()
        self.edit_R.setRange(0.003, 0.12)
        self.edit_R.setSingleStep(0.001)
        self.edit_R.setDecimals(3)
        self.edit_R.setValue(0.02)
        add_row("R", self.edit_R, "ед", "Радиус частицы (визуальная единица)")
        self.edit_T = QtWidgets.QDoubleSpinBox()
        self.edit_T.setRange(0.01, 5.0)
        self.edit_T.setSingleStep(0.1)
        self.edit_T.setValue(0.5)
        add_row("T", self.edit_T, "kT", "Температура (в единицах)")

        # Removed dt input: dt is fixed at 0.2
        # New: mass input (m)
        self.edit_m = QtWidgets.QDoubleSpinBox()
        self.edit_m.setRange(0.01, 100.0)
        self.edit_m.setSingleStep(0.1)
        self.edit_m.setValue(1.0)
        add_row("m", self.edit_m, "ед", "Масса частицы")

        pot_box = QtWidgets.QComboBox()
        pot_box.addItems(["Нет", "Отталкивание", "Притяжение", "Леннард-Джонс", "Морзе"])
        self.pot_box = pot_box
        add_row("Pot", pot_box, "", "Тип парного потенциала")
        self.edit_eps = QtWidgets.QDoubleSpinBox()
        self.edit_eps.setRange(0.0, 50.0)
        self.edit_eps.setValue(1.0)
        add_row("ε", self.edit_eps, "кДж/моль", "Энергетический масштаб")
        self.edit_sigma = QtWidgets.QDoubleSpinBox()
        self.edit_sigma.setRange(0.001, 0.2)
        self.edit_sigma.setValue(0.02)
        add_row("σ", self.edit_sigma, "ед", "Характерное расстояние LJ")
        self.edit_De = QtWidgets.QDoubleSpinBox()
        self.edit_De.setRange(0.0, 50.0)
        self.edit_De.setValue(1.0)
        add_row("De", self.edit_De, "кДж/моль", "Глубина ямы Morse")
        self.edit_a = QtWidgets.QDoubleSpinBox()
        self.edit_a.setRange(0.1, 50.0)
        self.edit_a.setValue(2.0)
        add_row("a", self.edit_a, "1/ед", "Жёсткость Morse")
        self.edit_r0 = QtWidgets.QDoubleSpinBox()
        self.edit_r0.setRange(0.001, 0.5)
        self.edit_r0.setValue(0.025)
        add_row("r₀", self.edit_r0, "ед", "Равновесное расстояние (Morse)")

        vessel_box = QtWidgets.QComboBox()
        vessel_box.addItems(["Прямоугольник", "Круг", "Многоугольник"])
        self.vessel_box = vessel_box
        add_row("Сосуд", vessel_box, "", "Форма сосуда")

        sp_layout.addSpacing(8)
        btn_draw = QtWidgets.QPushButton("Рисовать полигон")
        btn_draw.clicked.connect(self._enter_draw_mode)
        sp_layout.addWidget(btn_draw)
        btn_clear = QtWidgets.QPushButton("Очистить полигон")
        btn_clear.clicked.connect(self._clear_poly)
        sp_layout.addWidget(btn_clear)
        self.btn_run = QtWidgets.QPushButton("Пауза")
        self.btn_run.clicked.connect(self._toggle_run)
        sp_layout.addWidget(self.btn_run)
        btn_apply = QtWidgets.QPushButton("Применить")
        btn_apply.clicked.connect(self._apply_settings)
        sp_layout.addWidget(btn_apply)
        btn_back = QtWidgets.QPushButton("Назад")
        btn_back.clicked.connect(self.back_cb)
        sp_layout.addStretch(1)
        sp_layout.addWidget(btn_back)
        self.btn_collapse = QtWidgets.QPushButton("Свернуть")
        self.btn_collapse.clicked.connect(self._toggle_settings)
        sp_layout.addWidget(self.btn_collapse)

        canvas_container = QtWidgets.QHBoxLayout()
        main_l.addWidget(self.settings_panel)
        main_l.addLayout(canvas_container, stretch=1)

        anim_container = QtWidgets.QVBoxLayout()
        canvas_container.addLayout(anim_container, stretch=2)

        hist_container = QtWidgets.QVBoxLayout()
        canvas_container.addLayout(hist_container, stretch=1)

        top_bar = QtWidgets.QHBoxLayout()
        self.btn_toggle_settings_small = QtWidgets.QPushButton("≡ Настройки")
        self.btn_toggle_settings_small.setFixedSize(110, 28)
        self.btn_toggle_settings_small.clicked.connect(self._toggle_settings)
        top_bar.addWidget(self.btn_toggle_settings_small, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        top_bar.addStretch(1)
        anim_container.addLayout(top_bar)

        self.fig_anim = plt.Figure(figsize=(6, 6))
        self.ax_anim = self.fig_anim.add_subplot(111)
        self.ax_anim.set_aspect('equal')
        self.ax_anim.set_title("Сосуд — область частиц")
        self.canvas_anim = FigureCanvas(self.fig_anim)
        anim_container.addWidget(self.canvas_anim)
        self.canvas_anim.mpl_connect("button_press_event", self._on_mouse)

        self.fig_hist, axes = plt.subplots(3, 1, figsize=(10, 10))
        for ax in axes:
            ax.tick_params(labelsize=9)
        self.ax_histx, self.ax_histy, self.ax_histd = axes
        self.fig_hist.tight_layout()
        self.canvas_hist = FigureCanvas(self.fig_hist)
        hist_container.addWidget(self.canvas_hist)

        self.draw_mode = False
        self.poly_points = []

    def _init_simulation(self, N=50, radius=0.02, temp=0.3, dt=0.002,  # temp = 0.3 вместо 0.1
                     vessel_kind="Прямоугольник", poly=None,
                     potential_params=None, mass=1.0):

        if potential_params is None:
            potential_params = PotentialParams()

        if vessel_kind in ("rect", "Прямоугольник"):
            vessel = Vessel(kind="Прямоугольник", rect=(-1, -1, 1, 1))
        elif vessel_kind in ("circle", "Круг"):
            vessel = Vessel(kind="Круг", circle=(0, 0, 1))
        elif vessel_kind in ("poly", "Многоугольник"):
            if poly is None:
                poly = np.array([[-0.8, -0.8], [0.8, -0.8], [0.0, 0.8]])
            vessel = Vessel(kind="Многоугольник", poly=poly)
        else:
            vessel = Vessel(kind="Прямоугольник", rect=(-1, -1, 1, 1))

        self.system = System(
            vessel=vessel,
            N=N,
            radius=radius,
            temp=temp,
            dt=dt,
            params=potential_params,
            mass=mass
        )
        self.system.seed()

        if hasattr(self, 'scat') and self.scat is not None:
            self.scat.remove()

        scatter_s = max(2.0, (radius * 1000) ** 2)
        self.scat = self.ax_anim.scatter(self.system.pos[:, 0], self.system.pos[:, 1],
                                         s=scatter_s, c="#215a93", edgecolors='k', linewidths=0.4)

        self._draw_vessel_patch()
        self.ax_anim.relim()
        self.ax_anim.autoscale_view()
        self.canvas_anim.draw_idle()
        self.canvas_hist.draw_idle()
        self.running = True
        self._last_time = time.time()

    def _start_timer(self):
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start()

    def _on_timer(self):
        if self.running:
            if self.system is None:
                print("SYSTEM IS NONE!")
                return
            self.system.integrate()

            self._redraw_particles()

            current_time = time.time()
            if not hasattr(self, '_last_hist_update'):
                self._last_hist_update = 0

            if current_time - self._last_hist_update > 0.3:
                self._update_histograms()
                self._last_hist_update = current_time

    def _redraw_particles(self):
        if hasattr(self, 'scat') and self.system.pos is not None:
            self.scat.set_offsets(self.system.pos)
            if not hasattr(self, '_last_canvas_update'):
                self._last_canvas_update = 0

            current_time = time.time()
            if current_time - self._last_canvas_update > 0.033:
                self.canvas_anim.draw_idle()
                self._last_canvas_update = current_time

    def _update_histograms(self):
         # ===== Очистка осей =====
        for ax in (self.ax_histx, self.ax_histy, self.ax_histd):
            ax.cla()
            ax.set_facecolor("#f9f9f9")  # светлый фон
            ax.grid(True, linestyle='--', alpha=0.3)  # лёгкая сетка
            ax.tick_params(labelsize=9)

        # ===== Гистограмма X =====
        x = self.system.pos[:, 0]
        self.ax_histx.hist(x, bins=36, density=True, color="#8bb7d7", alpha=0.8)
        self.ax_histx.set_ylabel("p(x)", labelpad=5)
        self.ax_histx.set_title("Распределение X", fontsize=10, fontweight='bold')

        # ===== Гистограмма Y =====
        y = self.system.pos[:, 1]
        self.ax_histy.hist(y, bins=36, density=True, color="#f9ad6c", alpha=0.8)
        self.ax_histy.set_ylabel("p(y)", labelpad=5)
        self.ax_histy.set_title("Распределение Y", fontsize=10, fontweight='bold')

        # ===== Гистограмма расстояний =====
        dists = self.system.pairwise_distances_fast(max_pairs=5000)
        if dists.size > 0:
            self.ax_histd.hist(dists, bins=36, density=True, color="#8dd38d", alpha=0.85)
        self.ax_histd.set_ylabel("p(r)", labelpad=5)
        self.ax_histd.set_xlabel("r")
        self.ax_histd.set_title("Распределение расстояний", fontsize=10, fontweight='bold')

        # ===== Обновление канваса =====
        self.canvas_hist.draw_idle()


    def _apply_settings(self):
        N = int(self.spin_N.value())
        radius = float(self.edit_R.value())
        temp = float(self.edit_T.value())
        # dt is fixed
        dt = 0.002
        mass = float(self.edit_m.value())

        pot_params = PotentialParams(
            kind=str(self.pot_box.currentText()),
            epsilon=float(self.edit_eps.value()),
            sigma=float(self.edit_sigma.value()),
            De=float(self.edit_De.value()),
            a=float(self.edit_a.value()),
            r0=float(self.edit_r0.value())
        )

        vessel_kind = str(self.vessel_box.currentText())
        poly = self.system.vessel.poly if vessel_kind in ("poly", "Многоугольник") else None

        self._init_simulation(N=N, radius=radius, temp=temp, dt=dt,
                              vessel_kind=vessel_kind, poly=poly,
                              potential_params=pot_params, mass=mass)

        self._update_histograms()

    def _toggle_run(self):
        self.running = not self.running
        self.btn_run.setText("Пауза" if self.running else "Старт")

    def _toggle_settings(self):
        if self.settings_panel.isVisible():
            self.settings_panel.hide()
            self.btn_collapse.setText("Развернуть")
        else:
            self.settings_panel.show()
            self.btn_collapse.setText("Свернуть")

    def _enter_draw_mode(self):
        self.draw_mode = True
        self.poly_points = []

    def _clear_poly(self):
        self.system.vessel.poly = None
        self._draw_vessel_patch()
        self.canvas_anim.draw_idle()

    def _on_mouse(self, event):
        if not self.draw_mode: return
        if event.inaxes != self.ax_anim: return
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
            self.ax_anim.set_xlim(xmin - 0.1, xmax + 0.1)
            self.ax_anim.set_ylim(ymin - 0.1, ymax + 0.1)
        elif v.kind in ("circle", "Круг"):
            cx, cy, R = v.circle
            self.vessel_artist = patches.Circle((cx, cy), R, fill=False, lw=2, ec="#444")
            self.ax_anim.set_xlim(cx - R - 0.1, cx + R + 0.1)
            self.ax_anim.set_ylim(cy - R - 0.1, cy + R + 0.1)
        elif v.kind in ("poly", "Многоугольник") and v.poly is not None:
            self.vessel_artist = patches.Polygon(v.poly, closed=True, fill=False, lw=2, ec="#444")
            xmin, ymin = v.poly.min(axis=0)
            xmax, ymax = v.poly.max(axis=0)
            self.ax_anim.set_xlim(xmin - 0.1, xmax + 0.1)
            self.ax_anim.set_ylim(ymin - 0.1, ymax + 0.1)
        else:
            self.vessel_artist = None
        if self.vessel_artist is not None:
            self.ax_anim.add_patch(self.vessel_artist)


# -------------------- Application shell --------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распределение расстояний между частицами в сосудах различной формы")
        self.showFullScreen()
        self._build_ui()

    def _build_ui(self):
        central = QtWidgets.QStackedWidget()
        self.setCentralWidget(central)
        self.menu = MainMenuWidget(self, start_cb=self.show_sim, authors_cb=self.show_authors)
        self.sim = SimulationWidget(self, back_cb=self.show_menu)
        self.authors = AuthorsWidget(self, back_cb=self.show_menu)
        central.addWidget(self.menu)
        central.addWidget(self.sim)
        central.addWidget(self.authors)
        self.stack = central
        self.show_menu()

    def show_menu(self):
        self.stack.setCurrentWidget(self.menu)

    def show_sim(self):
        self.stack.setCurrentWidget(self.sim)

    def show_authors(self):
        self.stack.setCurrentWidget(self.authors)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
