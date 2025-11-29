import math
import numpy as np
from numba import njit, prange


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

