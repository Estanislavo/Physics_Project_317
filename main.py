
"""
Vessel Particle Simulation — FULL (fast + robust)
=================================================
- Формы сосуда: Rect / Circle / Polygon (рисование мышью).
- Потенциалы: None, Repel(1/r^2), Attract(1/r^2), Lennard-Jones, Morse.
- Ускорение: cell-list + отсечки по радиусу взаимодействия; редкая перерисовка гистограмм; подвыборка пар.
- Надёжные отражения для многоугольника: ближайшая точка на границе + отражение по внутренней нормали.
- Anti-tunneling: substepping — дробим шаг, чтобы частица не «перепрыгивала» стенку.
- Починен слайдер N (пересоздание системы), аккуратный layout.

Зависимости: numpy, matplotlib
Запуск: python vessel_particles_sim_full.py
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
from dataclasses import dataclass, field

# -------------------------- Utils --------------------------

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / n

def reflect_velocity(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    n = unit(n)
    return v - 2.0 * (v @ n.T) * n.squeeze()

# -------------------------- Vessel --------------------------

@dataclass
class Vessel:
    kind: str = "rect"
    rect: tuple = (-1.0, -1.0, 1.0, 1.0)
    circle: tuple = (0.0, 0.0, 1.0)
    poly: np.ndarray | None = None

    def contains(self, p: np.ndarray) -> bool:
        if self.kind == "rect":
            xmin,ymin,xmax,ymax = self.rect
            return (xmin <= p[0] <= xmax) and (ymin <= p[1] <= ymax)
        if self.kind == "circle":
            cx,cy,r = self.circle
            return (p[0]-cx)**2 + (p[1]-cy)**2 <= r**2 + 1e-12
        if self.kind == "poly" and self.poly is not None and len(self.poly) >= 3:
            return bool(Path(self.poly, closed=True).contains_point(p))
        return False

    def bounds(self):
        if self.kind == "rect":
            xmin,ymin,xmax,ymax = self.rect
            return (xmin, ymin, xmax, ymax)
        if self.kind == "circle":
            cx,cy,R = self.circle
            return (cx-R, cy-R, cx+R, cy+R)
        if self.kind == "poly" and self.poly is not None:
            xmin,ymin = self.poly.min(axis=0)
            xmax,ymax = self.poly.max(axis=0)
            return (xmin, ymin, xmax, ymax)
        return (-1,-1,1,1)

    def reflect(self, p_old: np.ndarray, p_new: np.ndarray, v: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        # Rect
        if self.kind == "rect":
            xmin,ymin,xmax,ymax = self.rect
            p = p_new.copy()
            if p[0] - radius < xmin or p[0] + radius > xmax:
                v[0] *= -1.0
                p[0] = np.clip(p[0], xmin+radius, xmax-radius)
            if p[1] - radius < ymin or p[1] + radius > ymax:
                v[1] *= -1.0
                p[1] = np.clip(p[1], ymin+radius, ymax-radius)
            return p, v

        # Circle
        if self.kind == "circle":
            cx,cy,R = self.circle
            d = p_new - np.array([cx,cy])
            dist = np.linalg.norm(d)
            if dist + radius > R:
                n = unit(d).squeeze()
                p = np.array([cx,cy]) + n * (R - radius - 1e-6)
                v = reflect_velocity(v, n)
                return p, v
            return p_new, v

        # Polygon
        if self.kind == "poly" and self.poly is not None and len(self.poly) >= 3:
            poly = self.poly
            path = Path(poly, closed=True)
            if path.contains_point(p_new):
                return p_new, v

            # Closest point on a segment AB to P
            def closest_point_on_segment(A, B, P):
                AB = B - A
                t = np.dot(P - A, AB) / (np.dot(AB, AB) + 1e-12)
                t = np.clip(t, 0.0, 1.0)
                Q = A + t * AB
                return Q, t

            # Find nearest boundary point Q
            best_d2 = np.inf; best_Q = None; best_edge = None
            m = len(poly)
            for i in range(m):
                A = poly[i]
                B = poly[(i+1) % m]
                Q, t = closest_point_on_segment(A, B, p_new)
                d2 = np.sum((p_new - Q)**2)
                if d2 < best_d2:
                    best_d2 = d2; best_Q = Q; best_edge = (A, B, t)

            A, B, t = best_edge
            edge = B - A
            n_out = np.array([edge[1], -edge[0]])
            n_out = n_out / (np.linalg.norm(n_out) + 1e-12)

            # Choose inward normal: test both directions
            for cand in (n_out, -n_out):
                p_test = best_Q + cand * (radius + 1e-4)
                if Path(poly, closed=True).contains_point(p_test):
                    n_in = cand
                    break
            else:
                centroid = poly.mean(axis=0)
                n_in = centroid - best_Q
                n_in = n_in / (np.linalg.norm(n_in)+1e-12)

            v_ref = reflect_velocity(v, n_in)
            p_corr = best_Q + n_in * (radius + 1e-4)

            # If still not inside (concave corners), push a bit more inward (few iters)
            it = 0
            while not Path(poly, closed=True).contains_point(p_corr) and it < 3:
                p_corr = p_corr + n_in * (radius + 1e-3)
                it += 1

            return p_corr, v_ref

        return p_new, v

# -------------------------- Potentials --------------------------

@dataclass
class PotentialParams:
    kind: str = "none"   # "none", "repel", "attract", "lj", "morse"
    k: float = 1.0       # for 1/r^2
    epsilon: float = 1.0
    sigma: float = 0.1
    De: float = 1.0
    a: float = 10.0
    r0: float = 0.2
    rcut_lr: float = 1.5 # cutoff for 1/r^2

# -------------------------- System (fast forces + substeps) --------------------------

@dataclass
class System:
    vessel: Vessel
    N: int = 50
    radius: float = 0.03
    temp: float = 1.0
    dt: float = 0.008
    params: PotentialParams = field(default_factory=PotentialParams)
    mass: float = 1.0
    pos: np.ndarray | None = None
    vel: np.ndarray | None = None

    # cell list state
    cell_size: float = 0.3
    bounds_cache: tuple = field(default_factory=lambda: (-1,-1,1,1))
    grid = None

    def n(self) -> int:
        return 0 if self.pos is None else self.pos.shape[0]

    def seed(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.pos = np.zeros((self.N,2))
        self.vel = np.zeros((self.N,2))

        def random_point_inside():
            if self.vessel.kind == "rect":
                xmin,ymin,xmax,ymax = self.vessel.rect
                for _ in range(10000):
                    p = np.array([rng.uniform(xmin+self.radius, xmax-self.radius),
                                  rng.uniform(ymin+self.radius, ymax-self.radius)])
                    if self.vessel.contains(p):
                        return p
            elif self.vessel.kind == "circle":
                cx,cy,R = self.vessel.circle
                for _ in range(10000):
                    ang = rng.uniform(0, 2*np.pi)
                    rad = rng.uniform(0, R-self.radius) ** 0.5
                    p = np.array([cx, cy]) + rad * np.array([np.cos(ang), np.sin(ang)])
                    if self.vessel.contains(p):
                        return p
            else:
                poly = self.vessel.poly
                xmin, ymin = poly.min(axis=0)
                xmax, ymax = poly.max(axis=0)
                for _ in range(10000):
                    p = np.array([rng.uniform(xmin+self.radius, xmax-self.radius),
                                  rng.uniform(ymin+self.radius, ymax-self.radius)])
                    if self.vessel.contains(p):
                        return p
                return poly.mean(axis=0)

        for i in range(self.N):
            self.pos[i] = random_point_inside()

        self.vel = rng.normal(0, 1.0, size=(self.N,2)) * np.sqrt(self.temp)
        self._update_cell_size()
        self._build_grid()

    # --- cutoffs and grid ---
    def _rcut(self):
        p = self.params
        if p.kind == "lj":
            return 2.5 * max(p.sigma, 1e-6)
        if p.kind == "morse":
            return p.r0 + max(3.0/p.a, 0.05)
        if p.kind in ("repel","attract"):
            return p.rcut_lr
        return 0.0

    def _update_cell_size(self):
        rcut = self._rcut()
        self.cell_size = max(rcut, 0.25) if rcut > 0 else 0.4
        self.bounds_cache = self.vessel.bounds()

    def _build_grid(self):
        xmin,ymin,xmax,ymax = self.bounds_cache
        cs = self.cell_size
        nx = max(int(np.ceil((xmax-xmin)/cs)), 1)
        ny = max(int(np.ceil((ymax-ymin)/cs)), 1)
        cells = [[] for _ in range(nx*ny)]
        xx = ((self.pos[:,0]-xmin)/cs).astype(int).clip(0, nx-1)
        yy = ((self.pos[:,1]-ymin)/cs).astype(int).clip(0, ny-1)
        for i,(cx,cy) in enumerate(zip(xx,yy)):
            cells[cy*nx + cx].append(i)
        self.grid = {"xmin":xmin,"ymin":ymin,"cs":cs,"nx":nx,"ny":ny,"cells":cells}

    def _neighbors(self, i):
        g = self.grid; cs=g["cs"]; nx=g["nx"]; ny=g["ny"]; xmin=g["xmin"]; ymin=g["ymin"]
        x,y = self.pos[i]
        cx = int((x-xmin)/cs); cy = int((y-ymin)/cs)
        cx = max(0, min(nx-1, cx)); cy = max(0, min(ny-1, cy))
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                ix = cx+dx; iy = cy+dy
                if 0 <= ix < nx and 0 <= iy < ny:
                    yield from g["cells"][iy*nx + ix]

    def _forces_fast(self) -> np.ndarray:
        N = self.n()
        F = np.zeros((N,2), dtype=float)
        p = self.params
        kind = p.kind
        rcut = self._rcut()
        rcut2 = rcut*rcut if rcut>0 else None
        if kind == "none" or N < 2:
            return F

        # rebuild grid this substep
        self._build_grid()

        for i in range(N):
            ri = self.pos[i]
            for j in self._neighbors(i):
                if j <= i:
                    continue
                rvec = self.pos[j] - ri
                r2 = rvec[0]*rvec[0] + rvec[1]*rvec[1] + 1e-12
                if rcut2 is not None and r2 > rcut2:
                    continue
                r = np.sqrt(r2)
                n = rvec / r
                if kind == "repel":
                    mag = p.k / r2
                elif kind == "attract":
                    mag = -p.k / r2
                elif kind == "lj":
                    sr = p.sigma / r
                    sr6 = sr**6
                    sr12 = sr6*sr6
                    mag = 24*p.epsilon*(2*sr12 - sr6)/r
                elif kind == "morse":
                    d = r - p.r0
                    ea = np.exp(-p.a * d)
                    mag = -2 * p.a * p.De * (1 - ea) * ea
                else:
                    mag = 0.0
                f = mag * n
                F[i] -= f
                F[j] += f
        return F

    # --- Integrator with substeps ---
    def step(self):
        if self.pos is None or self.vel is None:
            return
        self._update_cell_size()
        dt_total = self.dt
        vnorm = np.linalg.norm(self.vel, axis=1)
        vmax = float(vnorm.max(initial=0.0))
        max_disp = vmax * dt_total
        allowed = 0.5 * self.radius + 1e-6
        n_sub = max(1, int(np.ceil(max_disp / allowed)))
        dt = dt_total / n_sub

        for _ in range(n_sub):
            F0 = self._forces_fast()
            a0 = F0 / self.mass
            pos_new = self.pos + self.vel*dt + 0.5*a0*(dt*dt)

            # reflect with vessel per particle
            N = self.n()
            for i in range(N):
                p_old = self.pos[i].copy()
                v = self.vel[i].copy()
                p_new, v_new = self.vessel.reflect(p_old, pos_new[i], v, self.radius)
                pos_new[i] = p_new
                self.vel[i] = v_new

            F1 = self._forces_fast()
            a1 = F1 / self.mass
            self.vel = self.vel + 0.5*(a0 + a1)*dt
            self.pos = pos_new

    def pairwise_distances_sampled(self, max_pairs=4000) -> np.ndarray:
        N = self.n()
        if N < 2:
            return np.array([])
        total = N*(N-1)//2
        if total <= max_pairs:
            idx = []
            for i in range(N-1):
                for j in range(i+1,N):
                    idx.append((i,j))
            a = np.array(idx, dtype=int)
        else:
            rng = np.random.default_rng()
            a = np.column_stack([rng.integers(0,N,size=max_pairs),
                                 rng.integers(0,N,size=max_pairs)])
            a = a[a[:,0] < a[:,1]]
        if a.size == 0:
            return np.array([])
        d = np.linalg.norm(self.pos[a[:,0]] - self.pos[a[:,1]], axis=1)
        return d

# -------------------------- App / UI --------------------------

class App:
    HIST_EVERY = 3

    def __init__(self):
        self.fig = plt.figure(figsize=(13.6,7.2))
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], height_ratios=[1,1,1])
        self.ax_anim = self.fig.add_subplot(gs[:,0])
        self.ax_histx = self.fig.add_subplot(gs[0,1])
        self.ax_histy = self.fig.add_subplot(gs[1,1])
        self.ax_histd = self.fig.add_subplot(gs[2,1])
        self.fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.27, wspace=0.32, hspace=0.40)

        self.ax_anim.set_aspect('equal', adjustable='box')
        self.ax_anim.set_title("Сосуд и частицы")
        self.ax_histx.set_title("Распределение X", fontsize=10)
        self.ax_histy.set_title("Распределение Y", fontsize=10)
        self.ax_histd.set_title("Распределение расстояний", fontsize=10)

        # vessel & system
        self.vessel = Vessel(kind="rect", rect=(-1,-1,1,1))
        self.system = System(self.vessel, N=50, radius=0.03, temp=1.0, dt=0.008)
        self.system.seed()

        # vessel patch
        self.vessel_patch = None
        self._draw_vessel_patch()

        # particles scatter
        self.scat = self.ax_anim.scatter(self.system.pos[:,0], self.system.pos[:,1],
                                         s=(self.system.radius*800)**2, alpha=0.85,
                                         edgecolor='k', linewidths=0.5, c="#1f77b4")

        # info
        self.info_text = self.ax_anim.text(0.02, 0.98, "", transform=self.ax_anim.transAxes,
                                           va='top', ha='left', fontsize=9,
                                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, ec='none'))

        # widgets
        self._build_widgets()

        # polygon draw
        self.draw_mode = False
        self.poly_points = []
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self._on_mouse)

        # run animation
        self.running = True
        self._frame = 0
        self.ani = FuncAnimation(self.fig, self._update, interval=16, blit=False)

    # --- vessel draw ---
    def _draw_vessel_patch(self):
        if self.vessel_patch is not None:
            self.vessel_patch.remove(); self.vessel_patch = None
        if self.vessel.kind == "rect":
            xmin,ymin,xmax,ymax = self.vessel.rect
            w = xmax-xmin; h=ymax-ymin
            self.vessel_patch = patches.Rectangle((xmin,ymin), w,h, fill=False, lw=2.0, ec="#444")
            self.ax_anim.set_xlim(xmin-0.1, xmax+0.1)
            self.ax_anim.set_ylim(ymin-0.1, ymax+0.1)
        elif self.vessel.kind == "circle":
            cx,cy,R = self.vessel.circle
            self.vessel_patch = patches.Circle((cx,cy), R, fill=False, lw=2.0, ec="#444")
            self.ax_anim.set_xlim(cx-R-0.1, cx+R+0.1)
            self.ax_anim.set_ylim(cy-R-0.1, cy+R+0.1)
        elif self.vessel.kind == "poly" and self.vessel.poly is not None:
            self.vessel_patch = patches.Polygon(self.vessel.poly, closed=True, fill=False, lw=2.0, ec="#444")
            xmin,ymin = self.vessel.poly.min(axis=0); xmax,ymax = self.vessel.poly.max(axis=0)
            self.ax_anim.set_xlim(xmin-0.1, xmax+0.1)
            self.ax_anim.set_ylim(ymin-0.1, ymax+0.1)
        if self.vessel_patch is not None:
            self.ax_anim.add_patch(self.vessel_patch)
        self.ax_anim.figure.canvas.draw_idle()

    # --- widgets ---
    def _build_widgets(self):
        ax_radio_vessel = self.fig.add_axes([0.06, 0.09, 0.11, 0.13])
        self.rb_vessel = RadioButtons(ax_radio_vessel, ('Rect','Circle','Polygon'), active=0)
        self.rb_vessel.on_clicked(self._on_vessel_change)

        ax_btn_draw = self.fig.add_axes([0.18, 0.155, 0.11, 0.045])
        self.btn_draw = Button(ax_btn_draw, 'Рисовать сосуд'); self.btn_draw.on_clicked(self._on_draw_click)

        ax_btn_clear = self.fig.add_axes([0.18, 0.105, 0.11, 0.045])
        self.btn_clear = Button(ax_btn_clear, 'Очистить'); self.btn_clear.on_clicked(self._on_clear_click)

        ax_radio_pot = self.fig.add_axes([0.31, 0.09, 0.19, 0.13])
        self.rb_pot = RadioButtons(ax_radio_pot, ('None','Repel(1/r^2)','Attract(1/r^2)','Lennard-Jones','Morse'), active=0)
        self.rb_pot.on_clicked(self._on_potential_change)

        axN  = self.fig.add_axes([0.52, 0.165, 0.18, 0.03])
        axR  = self.fig.add_axes([0.52, 0.125, 0.18, 0.03])
        axT  = self.fig.add_axes([0.52, 0.085, 0.18, 0.03])
        axDT = self.fig.add_axes([0.52, 0.045, 0.18, 0.03])
        self.sld_N  = Slider(axN , "N",   5, 150, valinit=self.system.N,     valstep=1)
        self.sld_R  = Slider(axR , "R",   0.01, 0.08, valinit=self.system.radius, valstep=0.005)
        self.sld_T  = Slider(axT , "Temp",0.1, 5.0,  valinit=self.system.temp,   valstep=0.1)
        self.sld_dt = Slider(axDT, "dt",  0.001, 0.03, valinit=self.system.dt,   valstep=0.001)
        for s in (self.sld_N, self.sld_R, self.sld_T, self.sld_dt):
            s.on_changed(self._on_slider_change)

        ax_run = self.fig.add_axes([0.72, 0.155, 0.11, 0.045])
        self.btn_run = Button(ax_run, 'Старт/Пауза'); self.btn_run.on_clicked(self._on_run_pause)

        ax_reset = self.fig.add_axes([0.72, 0.105, 0.11, 0.045])
        self.btn_reset = Button(ax_reset, 'Сброс'); self.btn_reset.on_clicked(self._on_reset)

        ax_eps = self.fig.add_axes([0.85, 0.155, 0.06, 0.040])
        self.tb_eps = TextBox(ax_eps, 'ε', initial=str(self.system.params.epsilon))
        ax_sig = self.fig.add_axes([0.92, 0.155, 0.06, 0.040])
        self.tb_sig = TextBox(ax_sig, 'σ', initial=str(self.system.params.sigma))
        ax_De = self.fig.add_axes([0.85, 0.105, 0.06, 0.040])
        self.tb_De = TextBox(ax_De, 'De', initial=str(self.system.params.De))
        ax_a  = self.fig.add_axes([0.92, 0.105, 0.06, 0.040])
        self.tb_a  = TextBox(ax_a , 'a', initial=str(self.system.params.a))
        ax_r0 = self.fig.add_axes([0.85, 0.055, 0.06, 0.040])
        self.tb_r0 = TextBox(ax_r0, 'r0', initial=str(self.system.params.r0))
        for tb in (self.tb_eps, self.tb_sig, self.tb_De, self.tb_a, self.tb_r0):
            tb.on_submit(self._on_param_submit)

    # --- callbacks ---
    def _on_vessel_change(self, label):
        if label == 'Rect':
            self.vessel.kind = "rect"; self.vessel.rect = (-1,-1,1,1)
        elif label == 'Circle':
            self.vessel.kind = "circle"; self.vessel.circle = (0.0,0.0,1.0)
        else:
            self.vessel.kind = "poly"
            if self.vessel.poly is None:
                self.vessel.poly = np.array([[-0.8,-0.8],[0.8,-0.8],[0.0,0.8]])
        self._draw_vessel_patch(); self._on_reset(None)

    def _on_draw_click(self, event):
        if self.vessel.kind != "poly":
            print("Рисовать можно только в режиме Polygon."); return
        self.draw_mode = True; self.poly_points = []
        print("Режим рисования: ЛКМ — вершины, ПКМ — замкнуть.")

    def _on_clear_click(self, event):
        self.poly_points = []; self.draw_mode = False; self.vessel.poly = None
        if self.vessel_patch is not None:
            self.vessel_patch.remove(); self.vessel_patch = None
            self.ax_anim.figure.canvas.draw_idle()

    def _on_potential_change(self, label):
        if label.startswith('None'):       self.system.params.kind = "none"
        elif label.startswith('Repel'):    self.system.params.kind = "repel"
        elif label.startswith('Attract'):  self.system.params.kind = "attract"
        elif label.startswith('Lennard'):  self.system.params.kind = "lj"
        elif label.startswith('Morse'):    self.system.params.kind = "morse"
        self.system._update_cell_size()

    def _on_slider_change(self, val):
        newN = int(self.sld_N.val)
        if self.system.pos is None or newN != self.system.pos.shape[0]:
            self.system.N = newN
            self._on_reset(None)
            return
        self.system.radius = float(self.sld_R.val)
        self.system.temp   = float(self.sld_T.val)
        self.system.dt     = float(self.sld_dt.val)

    def _on_run_pause(self, event):
        self.running = not self.running

    def _on_reset(self, event):
        self.system.vessel = self.vessel
        self.system.seed()
        self._redraw_particles(force=True)

    def _on_param_submit(self, _):
        p = self.system.params
        try: p.epsilon = float(self.tb_eps.text)
        except: pass
        try: p.sigma = float(self.tb_sig.text)
        except: pass
        try: p.De = float(self.tb_De.text)
        except: pass
        try: p.a = float(self.tb_a.text)
        except: pass
        try: p.r0 = float(self.tb_r0.text)
        except: pass
        self.system._update_cell_size()

    def _on_mouse(self, event):
        if not self.draw_mode or event.inaxes != self.ax_anim:
            return
        if event.button == 1:
            self.poly_points.append((event.xdata, event.ydata))
            self._preview_poly()
        elif event.button == 3:
            if len(self.poly_points) >= 3:
                self.vessel.kind = "poly"
                self.vessel.poly = np.array(self.poly_points, dtype=float)
                self.draw_mode = False; self.poly_points = []
                self._draw_vessel_patch(); self._on_reset(None)
                print("Полигон замкнут.")
            else:
                print("Нужно минимум 3 точки.")

    def _preview_poly(self):
        if len(self.poly_points) >= 2:
            pts = np.array(self.poly_points, dtype=float)
            if self.vessel_patch is not None:
                self.vessel_patch.remove()
            self.vessel_patch = patches.Polygon(pts, closed=False, fill=False, lw=2.0, ec="#888", ls='--')
            self.ax_anim.add_patch(self.vessel_patch)
            self.ax_anim.figure.canvas.draw_idle()

    # --- render ---
    def _redraw_particles(self, force=False):
        off = self.scat.get_offsets()
        if force or off.shape[0] != self.system.n():
            self.scat.remove()
            self.scat = self.ax_anim.scatter(self.system.pos[:,0], self.system.pos[:,1],
                                             s=(self.system.radius*800)**2, alpha=0.85,
                                             edgecolor='k', linewidths=0.5, c="#1f77b4")
        else:
            self.scat.set_offsets(self.system.pos)
            self.scat.set_sizes(np.full(self.system.n(), (self.system.radius*800)**2))

    def _update_hists(self):
        x = self.system.pos[:,0]; y = self.system.pos[:,1]
        self.ax_histx.cla(); self.ax_histy.cla(); self.ax_histd.cla()
        self.ax_histx.set_title("Распределение X", fontsize=10)
        self.ax_histy.set_title("Распределение Y", fontsize=10)
        self.ax_histd.set_title("Распределение расстояний", fontsize=10)
        self.ax_histx.hist(x, bins=20, density=True)
        self.ax_histy.hist(y, bins=20, density=True)
        d = self.system.pairwise_distances_sampled(max_pairs=5000)
        if d.size:
            self.ax_histd.hist(d, bins=30, density=True)

    def _update(self, frame):
        if self.running:
            self.system.step()
            self._redraw_particles()
            if (frame % self.HIST_EVERY) == 0:
                self._update_hists()
            # info text
            pot = self.system.params.kind
            if pot == "none": pot_name = "None"
            elif pot == "repel": pot_name = "Repel(1/r^2)"
            elif pot == "attract": pot_name = "Attract(1/r^2)"
            elif pot == "lj": pot_name = f"LJ (ε={self.system.params.epsilon:.2f}, σ={self.system.params.sigma:.2f})"
            elif pot == "morse": pot_name = f"Morse (De={self.system.params.De:.2f}, a={self.system.params.a:.2f}, r0={self.system.params.r0:.2f})"
            else: pot_name = pot
            self.info_text.set_text(f"N = {self.system.n()}\nR = {self.system.radius:.3f}\nTemp = {self.system.temp:.2f}\ndt = {self.system.dt:.3f}\nPotential: {pot_name}")
        return []

def main():
    App()
    plt.show()

if __name__ == "__main__":
    main()
