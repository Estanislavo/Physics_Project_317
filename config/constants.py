DEFAULT_RADIUS = 3.0
DEFAULT_TEMP = 5.0
DEFAULT_DT = 0.1
DEFAULT_MASS = 1.0
DEFAULT_N = 10

# Vessel bounds
VESSEL_RECT_BOUNDS = (-100.0, -100.0, 100.0, 100.0)
VESSEL_CIRCLE_CENTER_AND_RADIUS = (0.0, 0.0, 100.0)
VESSEL_POLYGON_DEFAULT = [[-100.0, -100.0], [100.0, -100.0], [0.0, 100.0]]

MAX_ACCUMULATED_POINTS = 100000

PARTICLE_COLOR = "#215a93"
SELECTED_PARTICLE_COLOR = "#ff4444"
HISTOGRAM_COLORS = {
    'x': "#8bb7d7",
    'y': "#f9ad6c", 
    'distance': "#8dd38d"
}

SLIDER_RANGES = {
    'N': (1, 100),
    'T': (1, 500),  # фактически 0.1-50
    'm': (1, 10000),  # фактически 0.01-100
    'eps': (0, 1000),
    'sigma': (10, 300),
    'De': (0, 1000),
    'a': (1, 500),
    'r0': (10, 200)
}