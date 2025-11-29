from dataclasses import dataclass


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

class PotentialManager:
    """Управление потенциалами взаимодействия"""
    
    @staticmethod
    def get_kind_mapping():
        return {
            "Нет": 0, "None": 0, "无": 0,
            "Отталкивание": 1, "Repulsion": 1, "排斥": 1,
            "Притяжение": 2, "Attraction": 2, "吸引": 2,
            "Леннард-Джонс": 3, "Lennard-Jones": 3, "伦纳德-琼斯": 3,
            "Морзе": 4, "Morse": 4, "莫尔斯": 4
        }
    
    @staticmethod
    def get_cutoff_radius(params_kind: int, sigma: float, r0: float, a: float):
        if params_kind == 3:  # Lennard-Jones
            return 3.0 * sigma
        elif params_kind == 4:  # Morse
            return r0 + 5.0 / a
        else:
            return 80.0