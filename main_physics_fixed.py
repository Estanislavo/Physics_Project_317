from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
from dataclasses import dataclass, field
from numba import njit
import tkinter as tk
from tkinter import font as tkFont
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from PIL import Image, ImageTk
import os


@njit(fastmath=True, cache=True)
def compute_forces_numba(pos, N, rcut2, kind, k, epsilon, sigma, De, a, r0, mass):
    F = np.zeros((N, 2), dtype=np.float32)

    if kind == 0:  # none
        return F

    for i in range(N):
        for j in range(i+1, N):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            r2 = dx*dx + dy*dy + 1e-12

            if r2 > rcut2:
                continue

            r = np.sqrt(r2)

            # clamp минимального расстояния ради стабильности
            rmin = 0.1 * sigma if sigma > 0.0 else 1e-3
            if r < rmin:
                r = rmin
                r2 = r*r

            invr = 1.0 / r
            nx = dx * invr  # Направление от i к j
            ny = dy * invr

            # ИСПРАВЛЕННЫЕ ПОТЕНЦИАЛЫ:

            if kind == 1:  # repel (1/r^2) - ОТТАЛКИВАНИЕ
                mag = k / r2
            elif kind == 2:  # attract (1/r^2)
                mag = -k / r2
            elif kind == 3:  # Lennard-Jones
                sr = sigma / r
                sr6 = sr ** 6
                sr12 = sr ** 12
                mag = 24.0 * epsilon * (2.0 * sr12 - sr6) / r
            elif kind == 4:  # Morse
                d = r - r0
                ea = np.exp(-a * d)
                mag = 2.0 * De * a * ea * (1.0 - ea)
            else:
                mag = 0.0

            fx = mag * nx
            fy = mag * ny

            F[i, 0] -= fx
            F[i, 1] -= fy
            F[j, 0] += fx
            F[j, 1] += fy

    return F

@njit(fastmath=True, cache=True)
def integrate_velocity_verlet_numba(pos, vel, F, F_old, dt, mass, N, gamma):
    """Velocity Verlet с демпированием"""
    # стандартный verlet-полуприём
    vel_half = vel + 0.5 * (F + F_old) / mass * dt
    pos_new = pos + vel_half * dt
    # обновлённые силы рассчитываются вне этой функции (в вашем коде)
    # но для совместимости здесь применяем демпинг к скорости
    damp = 1.0
    if gamma > 0.0:
        damp = np.exp(-gamma * dt)
    vel_new = vel_half * damp
    return pos_new, vel_new

@njit(fastmath=True, cache=True)
def reflect_rect_numba(pos, vel, xmin, ymin, xmax, ymax, radius):
    """Отражение от стен прямоугольника"""
    if pos[0] - radius < xmin:
        vel[0] = abs(vel[0])
        pos[0] = xmin + radius
    elif pos[0] + radius > xmax:
        vel[0] = -abs(vel[0])
        pos[0] = xmax - radius

    if pos[1] - radius < ymin:
        vel[1] = abs(vel[1])
        pos[1] = ymin + radius
    elif pos[1] + radius > ymax:
        vel[1] = -abs(vel[1])
        pos[1] = ymax - radius

    return pos, vel

@njit(fastmath=True, cache=True)
def reflect_circle_numba(pos, vel, cx, cy, R, radius):
    """Отражение от круга"""
    dx = pos[0] - cx
    dy = pos[1] - cy
    dist2 = dx*dx + dy*dy
    R_eff = R - radius

    if dist2 > R_eff*R_eff:
        dist = np.sqrt(dist2)
        if dist > 0:
            nx = dx / dist
            ny = dy / dist

            dot = vel[0]*nx + vel[1]*ny
            vel[0] -= 2.0 * dot * nx
            vel[1] -= 2.0 * dot * ny

            pos[0] = cx + nx * R_eff
            pos[1] = cy + ny * R_eff

    return pos, vel


@njit(fastmath=True, cache=True)
def reflect_poly_numba(pos: np.ndarray, vel: np.ndarray, poly: np.ndarray, radius: float):
    """ИСПРАВЛЕННАЯ версия отражения от полигона"""
    n_vertices = len(poly)
    min_dist = 1e10
    normal_x = 0.0
    normal_y = 0.0
    
    for i in range(n_vertices):
        j = (i + 1) % n_vertices
        
        # Начало и конец отрезка
        Ax, Ay = poly[i]
        Bx, By = poly[j]
        
        # Вектор AB
        ABx = Bx - Ax
        ABy = By - Ay
        AB_sq = ABx*ABx + ABy*ABy
        
        if AB_sq < 1e-12:
            continue
            
        # Вектор AP  
        APx = pos[0] - Ax
        APy = pos[1] - Ay
        
        # Проекция на AB
        t = (APx*ABx + APy*ABy) / AB_sq
        t = max(0.0, min(1.0, t))  # Ограничиваем отрезком
        
        # Ближайшая точка на отрезке
        Px = Ax + t * ABx
        Py = Ay + t * ABy
        
        # Вектор от ближайшей точки к частице
        dx = pos[0] - Px
        dy = pos[1] - Py
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist < min_dist:
            min_dist = dist
            if dist > 1e-8:
                normal_x = dx / dist
                normal_y = dy / dist
            else:
                # Если на линии - используем перпендикуляр к AB
                AB_len = np.sqrt(AB_sq)
                normal_x = -ABy / AB_len
                normal_y = ABx / AB_len
    
    # Отражение
    if min_dist < radius + 1e-6:
        dot = vel[0]*normal_x + vel[1]*normal_y
        vel[0] -= 2.0 * dot * normal_x
        vel[1] -= 2.0 * dot * normal_y
        
        # Коррекция позиции
        correction = (radius - min_dist) * 1.01
        pos[0] += correction * normal_x
        pos[1] += correction * normal_y
        
    return pos, vel

@njit(cache=True)
def pairwise_distances_numba(pos, N):
    """Вычисление расстояний"""
    distances = np.zeros(N*(N-1)//2, dtype=np.float32)
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            distances[idx] = np.sqrt(dx*dx + dy*dy)
            idx += 1
    return distances

# ==================== VESSEL ====================
@dataclass
class Vessel:
    kind: str = "rect"
    rect: tuple = (-1.0, -1.0, 1.0, 1.0)
    circle: tuple = (0.0, 0.0, 1.0)
    poly: np.ndarray | None = None

    def contains(self, p: np.ndarray) -> bool:
        if self.kind == "rect":
            xmin, ymin, xmax, ymax = self.rect
            return (xmin <= p[0] <= xmax) and (ymin <= p[1] <= ymax)
        if self.kind == "circle":
            cx, cy, r = self.circle
            return (p[0]-cx)**2 + (p[1]-cy)**2 <= r**2 + 1e-12
        if self.kind == "poly" and self.poly is not None and len(self.poly) >= 3:
            return bool(Path(self.poly, closed=True).contains_point(p))
        return False

    def bounds(self):
        if self.kind == "rect":
            return self.rect
        if self.kind == "circle":
            cx, cy, R = self.circle
            return (cx-R, cy-R, cx+R, cy+R)
        if self.kind == "poly" and self.poly is not None:
            xmin, ymin = self.poly.min(axis=0)
            xmax, ymax = self.poly.max(axis=0)
            return (xmin, ymin, xmax, ymax)
        return (-1, -1, 1, 1)

    def reflect_particle(self, p: np.ndarray, v: np.ndarray, radius: float):
        if self.kind == "rect":
            return reflect_rect_numba(p, v, *self.rect, radius)
        elif self.kind == "circle":
            return reflect_circle_numba(p, v, *self.circle, radius)
        elif self.kind == "poly" and self.poly is not None and len(self.poly) >= 3:
            return reflect_poly_numba(p, v, self.poly, radius)  # ← ЭТО НЕ РАБОТАЕТ!
        return p, v

# ==================== POTENTIAL PARAMS ====================
@dataclass
class PotentialParams:
    kind: str = "none"
    k: float = 1.0      # [кДж/моль·нм²] для электростатики/гравитации
    epsilon: float = 1.0 # [кДж/моль] глубина потенциальной ямы
    sigma: float = 0.34  # [нм] характерное расстояние (диаметр атома)
    De: float = 1.0     # [кДж/моль] глубина ямы Морзе
    a: float = 2.0      # [1/нм] параметр жёсткости Морзе
    r0: float = 0.38    # [нм] равновесное расстояние Морзе
    rcut_lr: float = 2.5 # [нм] радиус обрезания дальних взаимодействий

# ==================== SYSTEM ====================
@dataclass
class System:
    vessel: Vessel
    N: int = 50
    radius: float = 0.03  # [нм] уменьшен радиус частицы (с 0.17 до 0.02)
    temp: float = 2.0     # [kT] безразмерная температура
    dt: float = 0.001     # [пс] шаг по времени
    params: PotentialParams = field(default_factory=PotentialParams)
    mass: float = 39.95   # [г/моль] масса частицы (аргон)
    friction_gamma: float = 0.1  # [1/пс] коэффициент трения

    pos: np.ndarray | None = None
    vel: np.ndarray | None = None
    F: np.ndarray | None = None
    F_old: np.ndarray | None = None

    def n(self) -> int:
        return 0 if self.pos is None else self.pos.shape[0]

    def seed(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        self.pos = np.zeros((self.N, 2), dtype=np.float32)
        self.vel = np.zeros((self.N, 2), dtype=np.float32)
        self.F = np.zeros((self.N, 2), dtype=np.float32)
        self.F_old = np.zeros((self.N, 2), dtype=np.float32)

        def random_point_inside():
            if self.vessel.kind == "rect":
                xmin, ymin, xmax, ymax = self.vessel.rect
                for _ in range(10000):
                    p = np.array([rng.uniform(xmin+self.radius, xmax-self.radius),
                                  rng.uniform(ymin+self.radius, ymax-self.radius)])
                    if self.vessel.contains(p):
                        return p
            elif self.vessel.kind == "circle":
                cx, cy, R = self.vessel.circle
                for _ in range(10000):
                    ang = rng.uniform(0, 2*np.pi)
                    rad = np.sqrt(rng.uniform(0, (R-self.radius)**2))  # ОДИН корень!
                    p = np.array([cx, cy]) + rad * np.array([np.cos(ang), np.sin(ang)])
            elif self.vessel.kind == "poly" and self.vessel.poly is not None:
                poly = self.vessel.poly
                xmin, ymin = poly.min(axis=0)
                xmax, ymax = poly.max(axis=0)
                for _ in range(10000):
                    p = np.array([rng.uniform(xmin+self.radius, xmax-self.radius),
                                  rng.uniform(ymin+self.radius, ymax-self.radius)])
                    if self.vessel.contains(p):
                        return p
                return poly.mean(axis=0)
            return np.array([0.0, 0.0])

        for i in range(self.N):
            self.pos[i] = random_point_inside()

        sigma_v = np.sqrt(self.temp / max(self.mass, 1e-12))
        self.vel = rng.normal(0.0, sigma_v, size=(self.N, 2)).astype(np.float32)

    def _rcut(self):
        p = self.params
        if p.kind == "lj":
            return 2.5 * max(p.sigma, 1e-6)
        if p.kind == "morse":
            return p.r0 + max(3.0/p.a, 0.05)
        if p.kind in ("repel", "attract"):
            return p.rcut_lr
        return 0.0

    def _potential_kind_code(self) -> int:
        kinds = {"none": 0, "repel": 1, "attract": 2, "lj": 3, "morse": 4}
        return kinds.get(self.params.kind, 0)

    def step(self):
        """Главный цикл - ВСЕ в Numba!"""
        if self.pos is None:
            return

        N = self.n()
        rcut = self._rcut()
        rcut2 = rcut*rcut if rcut > 0 else 1e10
        kind_code = self._potential_kind_code()

        p = self.params

        F_new = compute_forces_numba(
            self.pos, N, rcut2, kind_code,  # ← УБРАТЬ vel!
            p.k, p.epsilon, p.sigma, p.De, p.a, p.r0, self.mass
        )

        pos_new, vel_new = integrate_velocity_verlet_numba(
            self.pos, self.vel, F_new, self.F_old, self.dt, self.mass, N, self.friction_gamma
        )

        for i in range(N):
            pos_new[i], vel_new[i] = self.vessel.reflect_particle(
                pos_new[i], vel_new[i], self.radius
            )

        self.F_old[:] = F_new
        self.pos[:] = pos_new
        self.vel[:] = vel_new

    def pairwise_distances_fast(self, max_pairs=10000) -> np.ndarray:
        N = self.n()
        if N < 2:
            return np.array([])

        total = N*(N-1)//2
        if total <= max_pairs:
            return pairwise_distances_numba(self.pos, N)
        else:
            rng = np.random.default_rng()
            indices = rng.integers(0, N, size=max_pairs*2).reshape(-1, 2)
            indices = indices[indices[:, 0] < indices[:, 1]]
            d = np.linalg.norm(self.pos[indices[:, 0]] - self.pos[indices[:, 1]], axis=1)
            return d

# ==================== MAIN MENU ====================
class MainMenuWindow:
    def __init__(self, root, on_start_callback=None, on_authors_callback=None):
        self.root = root
        self.on_start_callback = on_start_callback
        self.on_authors_callback = on_authors_callback
        
        if isinstance(self.root, tk.Frame):
            self.container = self.root
        else:
            self.container = self.root

        # Современный светло-серый фон
        self.container.configure(bg='#f5f5f7')
        self.container.pack_propagate(False)

        # Создаем центральный фрейм для контента
        content_frame = tk.Frame(self.container, bg='#f5f5f7')
        content_frame.place(relx=0.5, rely=0.45, anchor='center')

        left_logo_frame = tk.Frame(self.container, bg='#f5f5f7')
        left_logo_frame.place(relx=0.1, rely=0.5, anchor='w')
        
        right_logo_frame = tk.Frame(self.container, bg='#f5f5f7')
        right_logo_frame.place(relx=0.9, rely=0.5, anchor='e')

        images_dir = os.path.join(os.path.dirname(__file__), 'images')
        try:
            # Левый логотип
            left_img = Image.open(os.path.join(images_dir, 'cmc_logo.png'))
            # Конвертируем в RGBA если изображение не прозрачное
            left_img = left_img.convert('RGBA')
            
            # Делаем белый фон прозрачным
            data = left_img.getdata()
            new_data = []
            for item in data:
                # Если пиксель близок к белому - делаем его прозрачным
                if item[0] > 240 and item[1] > 240 and item[2] > 240:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append(item)
            left_img.putdata(new_data)
            
            left_img.thumbnail((200, 200), Image.Resampling.LANCZOS)
            left_photo = ImageTk.PhotoImage(left_img)
            left_label = tk.Label(left_logo_frame, image=left_photo, bg='#f5f5f7')
            left_label.image = left_photo
            left_label.pack()
            
            # Правый логотип
            right_img = Image.open(os.path.join(images_dir, 'fiz_logo.jpg'))  # меняем расширение на .png
            right_img = right_img.convert('RGBA')
            
            # Аналогично для правого логотипа
            data = right_img.getdata()
            new_data = []
            for item in data:
                if item[0] > 240 and item[1] > 240 and item[2] > 240:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append(item)
            right_img.putdata(new_data)
            
            right_img.thumbnail((200, 200), Image.Resampling.LANCZOS)
            right_photo = ImageTk.PhotoImage(right_img)
            right_label = tk.Label(right_logo_frame, image=right_photo, bg='#f5f5f7')
            right_label.image = right_photo
            right_label.pack()
        except Exception as e:
            print(f"Ошибка загрузки логотипов: {e}")

        # Улучшенные шрифты
        title_font = tkFont.Font(family="Helvetica", size=36, weight="bold")
        subtitle_font = tkFont.Font(family="Helvetica", size=24)
        desc_font = tkFont.Font(family="Helvetica", size=16)
        btn_font = tkFont.Font(family="Helvetica", size=16, weight="bold")

        # Заголовок с тенью
        title_frame = tk.Frame(content_frame, bg='#f5f5f7')
        title_frame.pack(pady=(0, 30))
        
        title = tk.Label(title_frame, 
                        text="Распределение расстояний между частицами",
                        font=title_font, 
                        bg='#f5f5f7', 
                        fg='#1d1d1f')
        title.pack()
        
        subtitle = tk.Label(title_frame,
                          text="в сосудах различной формы",
                          font=subtitle_font,
                          bg='#f5f5f7',
                          fg='#1d1d1f')
        subtitle.pack(pady=(5, 0))

        # Обновленное описание с более научным подходом
        desc_frame = tk.Frame(content_frame, bg='#f5f5f7', padx=20)
        desc_frame.pack(pady=(0, 40))
       
        # Кнопки с современным дизайном и hover-эффектом
        btn_start = tk.Button(content_frame, 
                             text="Начать симуляцию",
                             font=btn_font,
                             bg='#0071e3',
                             activebackground='#0077ED',
                             activeforeground='white',
                             width=20,
                             height=2,
                             relief='flat',
                             command=self.on_start)
        btn_start.pack(pady=(0, 15))

        btn_authors = tk.Button(content_frame,
                               text="Об авторах",
                               font=btn_font,
                               bg='#313132',
                               activebackground='#424245',
                               activeforeground='white',
                               width=20,
                               height=2,
                               relief='flat',
                               command=self.on_authors)
        btn_authors.pack()

        # Кнопка выхода
        exit_btn = tk.Button(content_frame,
                           text="Выход",
                           font=btn_font,
                           bg='#FF3B30',  # Apple-style red
                           activebackground='#FF453A',
                           activeforeground='white',
                           width=20,
                           height=2,
                           relief='flat',
                           command=self.root.quit)
        exit_btn.pack(pady=(15, 0))

        def on_enter(e):
            e.widget['background'] = '#FF453A'
            
        def on_leave(e):
            e.widget['background'] = '#FF3B30'
            
        exit_btn.bind("<Enter>", on_enter)
        exit_btn.bind("<Leave>", on_leave)


        # Добавляем hover эффекты для кнопок
        def on_enter(e):
            e.widget['background'] = '#0077ED' if e.widget['text'] == "Начать симуляцию" else '#424245'

        def on_leave(e):
            e.widget['background'] = '#0071e3' if e.widget['text'] == "Начать симуляцию" else '#313132'

        for btn in (btn_start, btn_authors):
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)

    def on_start(self):
        if callable(self.on_start_callback):
            self.on_start_callback()

    def on_authors(self):
        if callable(self.on_authors_callback):
            self.on_authors_callback()
            return
        # иначе — старое поведение (на случай, если callback не передан)
        try:
            images_dir = os.path.join(os.path.dirname(__file__), 'images')
        except Exception:
            images_dir = 'images'
        # fallback: открываем Toplevel с тем же содержанием
        authors_window = tk.Toplevel(self.root)
        authors_window.title("Авторы проекта")
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        authors_window.geometry(f"{sw}x{sh}+0+0")
        authors_window.configure(bg='#f5f5f7')
        authors_window.resizable(True, True)
        try:
            authors_window.grab_set()
        except Exception:
            pass

        title_font = tkFont.Font(family="Helvetica", size=28, weight="bold")
        name_font = tkFont.Font(family="Helvetica", size=20, weight="bold")
        role_font = tkFont.Font(family="Helvetica", size=14)

        header = tk.Label(authors_window, text="Авторы",
                          font=title_font, bg='#f5f5f7', fg='#1d1d1f')
        header.pack(pady=20)

        cards_frame = tk.Frame(authors_window, bg='#f5f5f7')
        cards_frame.pack(fill=tk.BOTH, expand=True, padx=60, pady=10)

        def load_sticker(path, size):
            try:
                img = Image.open(path).convert('RGBA')
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                data = img.getdata()
                new_data = []
                for item in data:
                    if item[0] > 240 and item[1] > 240 and item[2] > 240:
                        new_data.append((255, 255, 255, 0))
                    else:
                        new_data.append(item)
                img.putdata(new_data)
                return ImageTk.PhotoImage(img)
            except Exception:
                return None

        photo_size = 420
        left_card = tk.Frame(cards_frame, bg='white', padx=30, pady=30)
        left_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=30, pady=20)
        right_card = tk.Frame(cards_frame, bg='white', padx=30, pady=30)
        right_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=30, pady=20)

        left_photo = load_sticker(os.path.join(images_dir, 'author1.png'), photo_size) or \
                     load_sticker(os.path.join(images_dir, 'author1.jpg'), photo_size)
        if left_photo:
            lbl = tk.Label(left_card, image=left_photo, bg='white')
            lbl.image = left_photo
            lbl.pack(pady=(0, 15))
        tk.Label(left_card, text="Иванов Иван", font=name_font, bg='white', fg='#1d1d1f').pack()
        tk.Label(left_card, text="Ведущий разработчик\n(физика и код)", font=role_font, bg='white',
                 fg='#515154').pack(pady=(6, 0))

        right_photo = load_sticker(os.path.join(images_dir, 'author2.png'), photo_size) or \
                      load_sticker(os.path.join(images_dir, 'author2.jpg'), photo_size)
        if right_photo:
            lbl2 = tk.Label(right_card, image=right_photo, bg='white')
            lbl2.image = right_photo
            lbl2.pack(pady=(0, 15))
        tk.Label(right_card, text="Петров Петр", font=name_font, bg='white', fg='#1d1d1f').pack()
        tk.Label(right_card, text="Физик-теоретик\n(модели и анализ)", font=role_font, bg='white',
                 fg='#515154').pack(pady=(6, 0))

def show_main_menu():
    # create a single Tk window with stacked pages (no top bar)
    root = tk.Tk()
    root.title("Распределение расстояний между частицами")
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}+0+0")
    root.resizable(True, True)

    # container for pages
    container = tk.Frame(root)
    container.pack(fill='both', expand=True)

    # create pages as frames stacked in the same container
    page_menu = tk.Frame(container, bg='#f0f0f0')
    page_sim = tk.Frame(container, bg='#ffffff')
    page_authors = tk.Frame(container, bg='#f0f0f0')

    # place all pages in same location; we'll lift the active one
    for p in (page_menu, page_sim, page_authors):
        p.place(relx=0, rely=0, relwidth=1, relheight=1)

    def show_page(page_name: str):
        if page_name == 'menu':
            page_menu.lift()
        elif page_name == 'sim':
            page_sim.lift()
        elif page_name == 'authors':
            page_authors.lift()

    # Instantiate MainMenuWindow; on_authors теперь переключает на страницу authors
    MainMenuWindow(page_menu, on_start_callback=lambda: show_page('sim'),
                   on_authors_callback=lambda: show_page('authors'))

    # Instantiate simulation embedded into page_sim
    sim_app = run_simulation(page_sim, on_back=lambda: show_page('menu'))

    # --- Build authors page content (big photos) ---
    try:
        images_dir = os.path.join(os.path.dirname(__file__), 'images')
    except Exception:
        images_dir = 'images'

    title_font = tkFont.Font(family="Helvetica", size=28, weight="bold")
    name_font = tkFont.Font(family="Helvetica", size=20, weight="bold")
    role_font = tkFont.Font(family="Helvetica", size=16)

    # Используем основной фон страницы для всех внутренних элементов
    authors_bg = page_authors.cget('bg')

    header = tk.Label(page_authors, text="Авторы",
                      font=title_font, fg='#1d1d1f', bg=authors_bg)
    header.pack(pady=20)

    cards_frame = tk.Frame(page_authors, bg=authors_bg)
    cards_frame.pack(fill=tk.BOTH, expand=True, padx=60, pady=10)

    def load_sticker(path, size):
        try:
            img = Image.open(path).convert('RGBA')
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            data = img.getdata()
            new_data = []
            for item in data:
                # make near-white pixels transparent
                if item[0] > 240 and item[1] > 240 and item[2] > 240:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append(item)
            img.putdata(new_data)
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    photo_size = 520  # большие фото
    # карточки: по центру, равномерно, фон — основной страницы
    left_card = tk.Frame(cards_frame, bg=authors_bg, padx=30, pady=30)
    left_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=30, pady=20)
    right_card = tk.Frame(cards_frame, bg=authors_bg, padx=30, pady=30)
    right_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=30, pady=20)

    left_photo = load_sticker(os.path.join(images_dir, 'author1.png'), photo_size) or \
                 load_sticker(os.path.join(images_dir, 'author1.jpg'), photo_size)
    if left_photo:
        lbl = tk.Label(left_card, image=left_photo, bg=authors_bg, bd=0)
        lbl.image = left_photo
        lbl.pack(pady=(0, 15))
    tk.Label(left_card, text="Енягин Станислав", font=name_font, bg=authors_bg, fg='#1d1d1f').pack()

    right_photo = load_sticker(os.path.join(images_dir, 'author2.png'), photo_size) or \
                  load_sticker(os.path.join(images_dir, 'author2.jpg'), photo_size)
    if right_photo:
        lbl2 = tk.Label(right_card, image=right_photo, bg=authors_bg, bd=0)
        lbl2.image = right_photo
        lbl2.pack(pady=(0, 15))
    tk.Label(right_card, text="Кожемякова Елизавета", font=name_font, bg=authors_bg, fg='#1d1d1f').pack()

    # Кнопка возврата на главную (стиль сохранён)
    btn_font_auth = tkFont.Font(family="Helvetica", size=16, weight="bold")
    back_btn = tk.Button(page_authors, text="Вернуться", font=btn_font_auth,
                         bg='#313132',
                         activebackground='#424245', activeforeground='white',
                         width=20, height=2, relief='flat',
                         command=lambda: show_page('menu'))
    back_btn.pack(pady=18)

    def _on_enter_back(e):
        e.widget['background'] = '#424245'
    def _on_leave_back(e):
        e.widget['background'] = '#313132'
    back_btn.bind("<Enter>", _on_enter_back)
    back_btn.bind("<Leave>", _on_leave_back)

    show_page('menu')

    root.mainloop()

class SimulationApp:
    HIST_EVERY = 5

    def __init__(self, parent_frame, on_back_callback=None):
        # Add callback parameter
        self.parent = parent_frame
        self.on_back_callback = on_back_callback
        
        self.fig = plt.figure(figsize=(13.6, 7.2))
        try:
            # when running standalone this works; when embedded manager may not be available
            self.fig.canvas.manager.set_window_title("Симуляция - ULTRA-OPTIMIZED (Numba + Physics Fixed)")
        except Exception:
            pass

        self.title_font = {'fontname': 'Helvetica Neue', 'size': 14, 'weight': 'bold'}
        self.subtitle_font = {'fontname': 'Helvetica Neue', 'size': 12}
        self.label_font = {'fontname': 'Helvetica Neue', 'size': 10}
        self.info_font = {'fontname': 'Helvetica Neue', 'size': 9}

        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1, 1])
        self.ax_anim = self.fig.add_subplot(gs[:, 0])
        self.ax_histx = self.fig.add_subplot(gs[0, 1])
        self.ax_histy = self.fig.add_subplot(gs[1, 1])
        self.ax_histd = self.fig.add_subplot(gs[2, 1])

        ax_btn_back = self.fig.add_axes([0.03, 0.95, 0.08, 0.04])
        self.btn_back = Button(ax_btn_back, 'Назад')
        self.btn_back.label.set_fontfamily('Helvetica Neue')
        self.btn_back.label.set_fontsize(10)
        self.btn_back.on_clicked(self._on_back)

        self.fig.subplots_adjust(left=0.1, right=0.97, top=0.93, bottom=0.27, wspace=0.32, hspace=0.40)

        # embed the Matplotlib figure into the provided Tk frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=1)

        self.ax_anim.set_aspect('equal', adjustable='box')
        self.ax_anim.set_title("Симуляция движения частиц", fontsize=10, pad=5)
        
        self.ax_histx.set_title("")
        self.ax_histy.set_title("")
        self.ax_histd.set_title("")

        # Общий заголовок для гистограмм
        self.fig.text(0.805, 0.95, "Распределение параметров",
                     ha='center', va='center',
                     **self.subtitle_font)

        # Заголовки гистограмм
        label_style = dict(
            transform=self.ax_histx.transAxes,
            rotation=0,
            va='center',
            ha='right',
            fontname='Helvetica Neue',
            fontsize=11,
            fontweight='bold'
        )
        
        self.ax_histx.text(-0.15, 0.5, "X", **label_style)
        self.ax_histy.text(-0.15, 0.5, "Y", **label_style)
        self.ax_histd.text(-0.15, 0.5, "R", **label_style)

        # Информационный текст
        self.info_text = self.ax_anim.text(
            0.02, 0.98, "",
            transform=self.ax_anim.transAxes,
            va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            **self.info_font
        )

        self.vessel = Vessel(kind="rect", rect=(-1, -1, 1, 1))
        self.system = System(self.vessel, N=100, radius=0.03, temp=2.0, dt=0.001)
        self.system.seed()

        self.vessel_patch = None
        self._draw_vessel_patch()

        self.scat = self.ax_anim.scatter(self.system.pos[:, 0], 
                                        self.system.pos[:, 1],
                                        s=(self.system.radius * 800)**2,  # уменьшен множитель с 300 до 800
                                        alpha=0.85,
                                        edgecolor='k',
                                        linewidths=0.5,  # уменьшена толщина обводки
                                        c="#215a93"  
                                    )

        self.info_text = self.ax_anim.text(0.02, 0.98, "", transform=self.ax_anim.transAxes,
                                           va='top', ha='left', fontsize=9,
                                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

        self.running = True
        self._build_widgets()

        # Режим рисования полигона
        self.draw_mode = False
        self.poly_points = []
        self.cid = self.canvas.mpl_connect('button_press_event', self._on_mouse)

        self._frame = 0
        self._start_time = time.time()

        # use FuncAnimation but ensure canvas is redrawn for the embedded widget
        self.ani = FuncAnimation(self.fig, self._update, interval=16, blit=False, save_count=100)
        # force an initial draw
        self.canvas.draw()

    def _seed_particles_inside(self):
        def is_inside(x, y, r):
            p = np.array([x, y])
            return self.vessel.contains(p) and \
                np.all(np.abs(p) + r <= 1.0)  # проверка с учетом радиуса
        
        N = self.system.N
        r = self.system.radius
        pos = np.zeros((N, 2))
        
        for i in range(N):
            while True:
                # Генерируем с запасом от краев
                x = np.random.uniform(-0.9, 0.9)
                y = np.random.uniform(-0.9, 0.9)
                if is_inside(x, y, r):
                    pos[i] = [x, y]
                    break
        
        self.system.pos = pos
        # Начальные скорости
        self.system.vel = np.random.randn(N, 2) * np.sqrt(self.system.temp)

    def _on_back(self, event=None):
        if callable(self.on_back_callback):
            self.on_back_callback()

    def _create_settings_panel(self):
        """Создаёт виджеты боковой панели настроек (в том же окне)."""
        sf = self.settings_frame
        for w in sf.winfo_children():
            w.destroy()

        hdr = tk.Label(sf, text="Настройки симуляции", bg=sf.cget('bg'),
                       font=tkFont.Font(family="Helvetica", size=16, weight="bold"))
        hdr.pack(pady=(12, 6), padx=12)

        small = tkFont.Font(family="Helvetica", size=10)

        # helper to add label + entry + unit + help text
        def add_field(name, var, unit, help_text):
            frm = tk.Frame(sf, bg=sf.cget('bg'))
            frm.pack(fill='x', padx=12, pady=6)
            tk.Label(frm, text=name, bg=sf.cget('bg'), anchor='w', font=small).grid(row=0, column=0, sticky='w')
            ent = tk.Entry(frm, textvariable=var, width=10)
            ent.grid(row=0, column=1, sticky='e', padx=(6,6))
            tk.Label(frm, text=unit, bg=sf.cget('bg'), font=small).grid(row=0, column=2, sticky='w')
            tk.Label(frm, text=help_text, bg=sf.cget('bg'), fg='#555', font=('Helvetica', 8), wraplength=280, justify='left').grid(row=1, column=0, columnspan=3, sticky='w', pady=(3,0))

        # Переменные
        self.var_N = tk.IntVar(value=50)
        self.var_radius = tk.DoubleVar(value=0.03)
        self.var_temp = tk.DoubleVar(value=1.0)
        self.var_dt = tk.DoubleVar(value=0.001)
        self.var_gamma = tk.DoubleVar(value=self.system.friction_gamma if hasattr(self, 'system') else 0.1)

        self.var_pot_kind = tk.StringVar(value=(self.system.params.kind if hasattr(self, 'system') else 'none'))
        self.var_eps = tk.DoubleVar(value=(self.system.params.epsilon if hasattr(self, 'system') else 1.0))
        self.var_sigma = tk.DoubleVar(value=(self.system.params.sigma if hasattr(self, 'system') else 0.34))
        self.var_De = tk.DoubleVar(value=(self.system.params.De if hasattr(self, 'system') else 1.0))
        self.var_a = tk.DoubleVar(value=(self.system.params.a if hasattr(self, 'system') else 2.0))
        self.var_r0 = tk.DoubleVar(value=(self.system.params.r0 if hasattr(self, 'system') else 0.38))
        self.var_k = tk.DoubleVar(value=(self.system.params.k if hasattr(self, 'system') else 1.0))

        add_field("N", self.var_N, "шт", "Число частиц в системе (шт).")
        add_field("R", self.var_radius, "нм", "Радиус частицы (нм). Должен быть достаточно мал по сравнению с размером сосуда.")
        add_field("T", self.var_temp, "kT", "Температура (в единицах kT) — задаёт масштаб начальных скоростей.")
        add_field("dt", self.var_dt, "пс", "Шаг интегрирования (пс). Меньше — стабильнее, но медленнее.")
        add_field("γ", self.var_gamma, "1/пс", "Коэффициент трения (демпирования). 0 — без демпирования.")

        # Потенциалы
        pot_frm = tk.Frame(sf, bg=sf.cget('bg'))
        pot_frm.pack(fill='x', padx=12, pady=(8,4))
        tk.Label(pot_frm, text="Потенциал", bg=sf.cget('bg'), font=small).grid(row=0, column=0, sticky='w')
        pot_opts = ('none', 'repel', 'attract', 'lj', 'morse')
        pot_menu = ttk.Combobox(pot_frm, values=pot_opts, textvariable=self.var_pot_kind, state='readonly', width=12)
        pot_menu.grid(row=0, column=1, sticky='e', padx=6)
        tk.Label(pot_frm, text="Выбор типа парного взаимодействия.", bg=sf.cget('bg'), fg='#555', font=('Helvetica', 8)).grid(row=1, column=0, columnspan=2, sticky='w', pady=(3,0))

        add_field("ε", self.var_eps, "кДж/моль", "Глубина ямы Lennard-Jones / Morse (ε или De).")
        add_field("σ", self.var_sigma, "нм", "Характерное расстояние LJ (σ) — ориентируйтесь ~ 2·R.")
        add_field("De", self.var_De, "кДж/моль", "Глубина ямы для потенциала Морзе (De).")
        add_field("a", self.var_a, "1/нм", "Жёсткость Морзе (a) — чем больше, тем жёстче краткая часть.")
        add_field("r₀", self.var_r0, "нм", "Равновесное расстояние Морзе (r₀).")
        add_field("k", self.var_k, "кДж/моль·нм²", "Коэффициент для репел/аттракта (1/r² тип).")

        # Кнопки Применить / Закрыть панели
        btns = tk.Frame(sf, bg=sf.cget('bg'))
        btns.pack(fill='x', padx=12, pady=12)
        apply_btn = tk.Button(btns, text="Применить", bg='#0071e3', fg='white', relief='flat', command=self._apply_settings)
        apply_btn.pack(side='left', padx=6)
        close_btn = tk.Button(btns, text="Свернуть", bg='#313132', fg='white', relief='flat', command=self._toggle_settings)
        close_btn.pack(side='right', padx=6)

    def _populate_settings_panel(self):
        """Заполнить поля значениями из system (вызывается после seed)."""
        if not hasattr(self, 'system') or self.system is None:
            return
        self.var_N.set(self.system.N)
        self.var_radius.set(float(self.system.radius))
        self.var_temp.set(float(self.system.temp))
        self.var_dt.set(float(self.system.dt))
        self.var_gamma.set(float(self.system.friction_gamma))
        p = self.system.params
        self.var_pot_kind.set(p.kind)
        self.var_eps.set(p.epsilon)
        self.var_sigma.set(p.sigma)
        self.var_De.set(p.De)
        self.var_a.set(p.a)
        self.var_r0.set(p.r0)
        self.var_k.set(p.k)

    def _toggle_settings(self):
        """Свернуть/развернуть боковую панель настроек."""
        if self.settings_frame.winfo_viewable():
            self.settings_frame.forget()
        else:
            self.settings_frame.pack(side='left', fill='y')

    def _apply_settings(self):
        """Применить настройки из панели к системе и перезапустить/применить."""
        try:
            self.system.N = int(self.var_N.get())
            self.system.radius = float(self.var_radius.get())
            self.system.temp = float(self.var_temp.get())
            self.system.dt = float(self.var_dt.get())
            self.system.friction_gamma = float(self.var_gamma.get())

            p = self.system.params
            p.kind = str(self.var_pot_kind.get())
            p.epsilon = float(self.var_eps.get())
            p.sigma = float(self.var_sigma.get())
            p.De = float(self.var_De.get())
            p.a = float(self.var_a.get())
            p.r0 = float(self.var_r0.get())
            p.k = float(self.var_k.get())

            # Применяем — пересоздаём начальные условия
            self._on_reset(None)
        except Exception as e:
            print("Ошибка при применении настроек:", e)

    def _draw_vessel_patch(self):
        if self.vessel_patch is not None:
            self.vessel_patch.remove()
            self.vessel_patch = None

        if self.vessel.kind == "rect":
            xmin, ymin, xmax, ymax = self.vessel.rect
            w = xmax-xmin
            h = ymax-ymin
            self.vessel_patch = patches.Rectangle((xmin, ymin), w, h, fill=False, lw=2.0, ec="#444")
            self.ax_anim.set_xlim(xmin-0.1, xmax+0.1)
            self.ax_anim.set_ylim(ymin-0.1, ymax+0.1)

        elif self.vessel.kind == "circle":
            cx, cy, R = self.vessel.circle
            self.vessel_patch = patches.Circle((cx, cy), R, fill=False, lw=2.0, ec="#444")
            self.ax_anim.set_xlim(cx-R-0.1, cx+R+0.1)
            self.ax_anim.set_ylim(cy-R-0.1, cy+R+0.1)

        elif self.vessel.kind == "poly" and self.vessel.poly is not None:
            self.vessel_patch = patches.Polygon(self.vessel.poly, closed=True, fill=False, lw=2.0, ec="#444")
            xmin, ymin = self.vessel.poly.min(axis=0)
            xmax, ymax = self.vessel.poly.max(axis=0)
            self.ax_anim.set_xlim(xmin-0.1, xmax+0.1)
            self.ax_anim.set_ylim(ymin-0.1, ymax+0.1)

        if self.vessel_patch is not None:
            self.ax_anim.add_patch(self.vessel_patch)

    def _build_widgets(self):
        ax_radio_vessel = self.fig.add_axes([0.03, 0.07, 0.23, 0.12])
        self.rb_vessel = RadioButtons(ax_radio_vessel, ('Прямоугольник', 'Круг', 'Многоугольник'), active=0)

        self.rb_vessel.on_clicked(self._on_vessel_change)

        # Кнопки для рисования полигона
        ax_btn_draw = self.fig.add_axes([0.03, 0.01, 0.10, 0.04])
        self.btn_draw = Button(ax_btn_draw, 'Рисовать')
        self.btn_draw.on_clicked(self._on_draw_click)

        # Изначально блокируем кнопку если режим не полигон
        if self.vessel.kind != "poly":
            self.btn_draw.set_active(False)
            self.btn_draw.label.set_color('gray')
        else:
            self.btn_draw.set_active(True)
            self.btn_draw.label.set_color('black')

        ax_btn_clear = self.fig.add_axes([0.15, 0.01, 0.10, 0.04])
        self.btn_clear = Button(ax_btn_clear, 'Очистить')
        self.btn_clear.on_clicked(self._on_clear_click)

        ax_radio_pot = self.fig.add_axes([0.27, 0.01, 0.24, 0.18])  # ← width=0.24 вместо 0.28
        self.rb_pot = RadioButtons(ax_radio_pot, ('Нет', 'Отталкивание', 'Притяжение', 'Леннард-Джонс', 'Морзе'), active=0)
        self.rb_pot.on_clicked(self._on_potential_change)

        # Слайдеры с единицами измерения
        axN = self.fig.add_axes([0.55, 0.17, 0.18, 0.025])
        axR = self.fig.add_axes([0.55, 0.135, 0.18, 0.025])
        axT = self.fig.add_axes([0.55, 0.1, 0.18, 0.025])
        axK = self.fig.add_axes([0.55, 0.065, 0.18, 0.025])

        self.sld_N = Slider(axN, "N", 10, 200, valinit=self.system.N, valstep=5)
        self.sld_R = Slider(axR, "R", 0.01, 0.05, valinit=self.system.radius)
        self.sld_T = Slider(axT, "T", 0.1, 5.0, valinit=self.system.temp)
        self.sld_K = Slider(axK, "k", 0.0, 10.0, valinit=self.system.params.k)

        for s in (self.sld_N, self.sld_R, self.sld_T, self.sld_K):
            s.on_changed(self._on_slider_change)

        self.ax_run = self.fig.add_axes([0.55, 0.01, 0.15, 0.04])

        if self.running:
            btn_label = 'Пауза'
        else:
            btn_label = 'Старт'
        self.btn_run = Button(self.ax_run, btn_label)
        self.btn_run.on_clicked(self._on_run_pause)

        ax_reset = self.fig.add_axes([0.72, 0.01, 0.25, 0.04])
        self.btn_reset = Button(ax_reset, 'Сбросить / Применить')
        self.btn_reset.on_clicked(self._on_reset)

        # Поля для параметров потенциалов
        ax_eps = self.fig.add_axes([0.82, 0.16, 0.06, 0.035])
        self.tb_eps = TextBox(ax_eps, 'ε ', initial=str(self.system.params.epsilon))
        
        ax_sig = self.fig.add_axes([0.91, 0.16, 0.06, 0.035])
        self.tb_sig = TextBox(ax_sig, 'σ ', initial=str(self.system.params.sigma))
        
        ax_De = self.fig.add_axes([0.82, 0.11, 0.06, 0.035])
        self.tb_De = TextBox(ax_De, 'De ', initial=str(self.system.params.De))
        
        ax_a = self.fig.add_axes([0.91, 0.11, 0.06, 0.035])
        self.tb_a = TextBox(ax_a, 'a ', initial=str(self.system.params.a))
        
        ax_r0 = self.fig.add_axes([0.82, 0.06, 0.06, 0.035])
        self.tb_r0 = TextBox(ax_r0, 'r₀ ', initial=str(self.system.params.r0))
        
        for tb in (self.tb_eps, self.tb_sig, self.tb_De, self.tb_a, self.tb_r0):
            tb.on_submit(self._on_param_submit)

    def _on_vessel_change(self, label):
        if label == 'Прямоугольник':
            self.vessel.kind = "rect"
            self.vessel.rect = (-1, -1, 1, 1)
        elif label == 'Круг':
            self.vessel.kind = "circle"
            self.vessel.circle = (0.0, 0.0, 1.0)
        elif label == 'Многоугольник':
            self.vessel.kind = "poly"
            if self.vessel.poly is None:
                # Создаем треугольник по умолчанию
                self.vessel.poly = np.array([[-0.8, -0.8], [0.8, -0.8], [0.0, 0.8]])

        if self.vessel.kind != "poly":
            self.btn_draw.set_active(False)
            self.btn_draw.label.set_color('gray')
        else:
            self.btn_draw.set_active(True)
            self.btn_draw.label.set_color('black')

        self._draw_vessel_patch()
        self._on_reset(None)

    def _on_draw_click(self, event):
        if self.vessel.kind != "poly":
            print("Рисовать можно только в режиме Многоугольник.")
            return
        self.draw_mode = True
        self.poly_points = []
        print("Режим рисования: ЛКМ — вершины, ПКМ — замкнуть.")

    def _on_clear_click(self, event):
        self.poly_points = []
        self.draw_mode = False
        self.vessel.poly = None
        if self.vessel_patch is not None:
            self.vessel_patch.remove()
            self.vessel_patch = None
            self.ax_anim.figure.canvas.draw_idle()

    def _on_potential_change(self, label):
        if label.startswith('Нет'):
            self.system.params.kind = "none"
        elif label.startswith('Отталкивание'):
            self.system.params.kind = "repel"
        elif label.startswith('Притяжение'):
            self.system.params.kind = "attract"
        elif label.startswith('Леннард'):
            self.system.params.kind = "lj"
        elif label.startswith('Морзе'):
            self.system.params.kind = "morse"

    def _on_slider_change(self, val):
        newN = int(self.sld_N.val)
        if self.system.pos is None or newN != self.system.pos.shape[0]:
            self.system.N = newN
            self._on_reset(None)
            return
        self.system.radius = float(self.sld_R.val)
        self.system.temp = float(self.sld_T.val)
        self.system.params.k = float(self.sld_K.val)

    def _on_run_pause(self, event):
        self.running = not self.running
        self.btn_run.label.set_text('Пауза' if self.running else 'Старт')

    def _on_reset(self, event):
        self.system.vessel = self.vessel
        self.system.seed()
        self._redraw_particles(force=True)

    def _on_param_submit(self, text):
        p = self.system.params
        try:
            p.epsilon = float(self.tb_eps.text)
        except: pass
        try:
            p.sigma = float(self.tb_sig.text)
        except: pass
        try:
            p.De = float(self.tb_De.text)
        except: pass
        try:
            p.a = float(self.tb_a.text)
        except: pass
        try:
            p.r0 = float(self.tb_r0.text)
        except: pass

    def _on_mouse(self, event):
        if not self.draw_mode or event.inaxes != self.ax_anim:
            return
        
        if event.button == 1:  # ЛКМ - добавить вершину
            self.poly_points.append((event.xdata, event.ydata))
            self._preview_poly()
        elif event.button == 3:  # ПКМ - замкнуть полигон
            if len(self.poly_points) >= 3:
                self.vessel.kind = "poly"
                self.vessel.poly = np.array(self.poly_points, dtype=float)
                self.draw_mode = False
                self.poly_points = []
                self._draw_vessel_patch()
                self._on_reset(None)
                print("Полигон замкнут.")
            else:
                print("Нужно минимум 3 точки.")

    def _preview_poly(self):
        """Предпросмотр полигона во время рисования"""
        if len(self.poly_points) >= 2:
            pts = np.array(self.poly_points, dtype=float)
            if self.vessel_patch is not None:
                self.vessel_patch.remove()
            self.vessel_patch = patches.Polygon(pts, closed=False, fill=False, lw=2.0, ec="#888", ls='--')
            self.ax_anim.add_patch(self.vessel_patch)
            self.ax_anim.figure.canvas.draw_idle()

    def _redraw_particles(self, force=False):
        off = self.scat.get_offsets()
        if force or off.shape[0] != self.system.n():
            self.scat.remove()
            self.scat = self.ax_anim.scatter(self.system.pos[:, 0], 
                                        self.system.pos[:, 1],
                                        s=(self.system.radius * 800)**2,  # тот же множитель
                                        alpha=0.85,
                                        edgecolor='k',
                                        linewidths=0.5,
                                        c="#215a93"  # более приятный синий цвет
                                    )
        else:
            self.scat.set_offsets(self.system.pos)

    def _update_hists(self):
        x = self.system.pos[:, 0]
        y = self.system.pos[:, 1]

        self.ax_histx.cla()
        self.ax_histy.cla()
        self.ax_histd.cla()
        self.ax_anim.set_aspect('equal', adjustable='box')
        
        self.ax_histx.set_title("")
        self.ax_histy.set_title("")
        self.ax_histd.set_title("")

        # Добавляем заголовки слева от графиков с более "заголовочным" оформлением,
        # но оставляем их слева (не сверху)
        label_style = dict(transform=self.ax_histx.transAxes,
                           rotation=0, va='center', ha='right',
                           fontsize=11, fontweight='bold')
        self.ax_histx.text(-0.15, 0.5, "X", **label_style)
        self.ax_histy.text(-0.15, 0.5, "Y", transform=self.ax_histy.transAxes,
                           rotation=0, va='center', ha='right',
                           fontsize=11, fontweight='bold')
        self.ax_histd.text(-0.15, 0.5, "R", transform=self.ax_histd.transAxes,
                           rotation=0, va='center', ha='right',
                           fontsize=11, fontweight='bold')
        
        # Убираем лишние метки осей для компактности
        self.ax_histx.set_xlabel('')
        self.ax_histy.set_xlabel('')
        self.ax_histd.set_xlabel('')

        self.ax_histx.hist(x, bins=20, density=True, alpha=0.7)
        self.ax_histy.hist(y, bins=20, density=True, alpha=0.7)

        d = self.system.pairwise_distances_fast(max_pairs=10000)
        if d.size:
            self.ax_histd.hist(d, bins=30, density=True, alpha=0.7)

    def _update(self, frame):
        if self.running:
            self.system.step()
            self._redraw_particles()
            self.canvas.draw_idle()

        if (frame % self.HIST_EVERY) == 0:
            self._update_hists()

        if frame % 60 == 0:
            elapsed = time.time() - self._start_time
            fps = 60 / elapsed if elapsed > 0 else 0
            pot = self.system.params.kind
            k_val = self.system.params.k
            self.info_text.set_text(f"N={self.system.n()} FPS={fps:.1f}\nPot={pot} k={k_val:.1f}")
            self._start_time = time.time()

        return []

def run_simulation(parent_frame, on_back=None):
    # Add callback parameter
    app = SimulationApp(parent_frame, on_back_callback=on_back)
    return app

def main():
    show_main_menu()

if __name__ == "__main__":
    main()