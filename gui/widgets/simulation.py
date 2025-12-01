from PyQt6 import QtWidgets, QtCore, QtGui
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from config.strings import STRINGS, Lang
import config.constants 
from physics.particle_system import System
from physics.potentials import PotentialParams
from physics.vessels import Vessel
from gui.widgets.custom_widgets import WinCheckBox
import time
import math


class SimulationWidget(QtWidgets.QWidget):
    def __init__(self, parent, back_cb, get_lang_cb):
        super().__init__(parent)

        self.accumulated_x = np.array([])
        self.accumulated_y = np.array([])
        self.accumulated_distances = np.array([])
        self.max_accumulated_points = config.constants.MAX_ACCUMULATED_POINTS 

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

        self.distance_display_mode = 0  # 0 - обычное расстояние, 1 - проекции, 2 - расстояние со знаком

        self.back_cb = back_cb
        self.get_lang_cb = get_lang_cb
        self.bins_count = 36
        self._build_ui()
        self._init_simulation()
        self._start_timer()
        self.update_language(self.get_lang_cb())
        self._reset_data()

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
        self.settings_panel.setMinimumWidth(300)
        self.settings_panel.setStyleSheet("background:#f0f0f2; border-right: 2px solid #ddd;")
        sp_layout = QtWidgets.QVBoxLayout(self.settings_panel)
        sp_layout.setContentsMargins(8, 8, 8, 8)
        sp_layout.setSpacing(4)

        self.lbl_settings_title = QtWidgets.QLabel()
        self.lbl_settings_title.setStyleSheet("font-size:20pt; font-weight:700; color:#1a1a1a;")
        sp_layout.addWidget(self.lbl_settings_title)
        sp_layout.addSpacing(4)

        def add_slider_row(key_label, slider, label_value, key_unit="", key_help=""):
            """Компактная строка для слайдера: [Label] [Slider] [Value] [Unit]"""
            if key_help:
                h = QtWidgets.QLabel()
                h.setObjectName(f"help_{key_label}")
                h.setWordWrap(True)
                h.setStyleSheet("color:#666; font-size:14.5pt; margin-top:4px; padding:0px;")
                sp_layout.addWidget(h)
            
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(2, 2, 2, 2)
            row.setSpacing(4)
            
            lbl = QtWidgets.QLabel()
            lbl.setObjectName(f"lbl_{key_label}")
            lbl.setStyleSheet("font-size:14.5pt; font-weight:500; color:#2c2c2c;")
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
            label_value.setStyleSheet("font-size:14.5pt; color:#555; text-align:center;")
            row.addWidget(label_value)
            
            if key_unit:
                u = QtWidgets.QLabel()
                u.setObjectName(f"unit_{key_label}")
                u.setFixedWidth(50)
                u.setStyleSheet("font-size:14.5pt; color:#777;")
                row.addWidget(u)
            
            sp_layout.addLayout(row)

        def add_combobox_row(key_label, combobox, key_help=""):
            """Строка для комбобса: [Help сверху] [Label] [ComboBox]"""
            if key_help:
                h = QtWidgets.QLabel()
                h.setObjectName(f"help_{key_label}")
                h.setWordWrap(True)
                h.setStyleSheet("color:#666; font-size:14.5pt; margin:0px; padding:0px;")
                sp_layout.addWidget(h)
            
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(2, 2, 2, 2)
            row.setSpacing(4)
            
            lbl = QtWidgets.QLabel()
            lbl.setObjectName(f"lbl_{key_label}")
            lbl.setStyleSheet("font-size:14.5pt; font-weight:500; color:#2c2c2c;")
            row.addWidget(lbl)
            
            combobox.setStyleSheet("""
                QComboBox, QSpinBox{
                    font-size:16pt; 
                    width: 180px;
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
        self.spin_N.setRange(*config.constants.SLIDER_RANGES['N'])
        self.spin_N.setValue(10)
        self.spin_N.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.spin_N_label = QtWidgets.QLabel("10")
        self.spin_N.sliderMoved.connect(lambda v: self.spin_N_label.setText(str(v)))
        self.spin_N.valueChanged.connect(lambda v: (self.spin_N_label.setText(str(v)), self._on_settings_changed()))
        add_slider_row("N", self.spin_N, self.spin_N_label, "sim.N.unit", "sim.N.help")

        # T — ползунок (0.1-50)
        self.edit_T = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.edit_T.setRange(*config.constants.SLIDER_RANGES['T'])
        self.edit_T.setValue(50)
        self.edit_T.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.edit_T_label = QtWidgets.QLabel("5.0")
        self.edit_T.sliderMoved.connect(lambda v: self.edit_T_label.setText(f"{v/10:.1f}"))
        self.edit_T.valueChanged.connect(lambda v: (self.edit_T_label.setText(f"{v/10:.1f}"), self._on_settings_changed()))
        add_slider_row("T", self.edit_T, self.edit_T_label, "sim.T.unit", "sim.T.help")

        # m — ползунок (0.01-100)
        self.edit_m = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.edit_m.setRange(*config.constants.SLIDER_RANGES['m'])
        self.edit_m.setValue(1000)
        self.edit_m.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.edit_m_label = QtWidgets.QLabel("1.0")
        self.edit_m.sliderMoved.connect(lambda v: self.edit_m_label.setText(f"{v/1000:.2f}"))
        self.edit_m.valueChanged.connect(lambda v: (self.edit_m_label.setText(f"{v/1000:.2f}"), self._on_settings_changed()))
        add_slider_row("m", self.edit_m, self.edit_m_label, "sim.m.unit", "sim.m.help")

        self.check_collisions = WinCheckBox()
        self.check_collisions.stateChanged.connect(self._on_settings_changed)
        self.check_collisions.setStyleSheet("font-size:16pt; color:#2c2c2c; margin-top:4px;")
        self.check_collisions.setChecked(True)
        sp_layout.addWidget(self.check_collisions)

        self.pot_box = QtWidgets.QComboBox()
        self.pot_box.currentIndexChanged.connect(self._on_settings_changed)
        add_combobox_row("pot", self.pot_box, "sim.pot.help")

        # eps, sigma, De, a, r0 — ползунки
        self.edit_eps = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.edit_eps.setRange(*config.constants.SLIDER_RANGES['eps'])
        self.edit_eps.setValue(100)
        self.edit_eps.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.edit_eps_label = QtWidgets.QLabel("10.0")
        self.edit_eps.sliderMoved.connect(lambda v: self.edit_eps_label.setText(f"{v/10:.1f}"))
        self.edit_eps.valueChanged.connect(lambda v: (self.edit_eps_label.setText(f"{v/10:.1f}"), self._on_settings_changed()))
        add_slider_row("eps", self.edit_eps, self.edit_eps_label, "sim.eps.unit", "sim.eps.help")

        self.edit_sigma = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.edit_sigma.setRange(*config.constants.SLIDER_RANGES['sigma'])
        self.edit_sigma.setValue(150)
        self.edit_sigma.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.edit_sigma_label = QtWidgets.QLabel("15.0")
        self.edit_sigma.sliderMoved.connect(lambda v: self.edit_sigma_label.setText(f"{v/10:.1f}"))
        self.edit_sigma.valueChanged.connect(lambda v: (self.edit_sigma_label.setText(f"{v/10:.1f}"), self._on_settings_changed()))
        add_slider_row("sigma", self.edit_sigma, self.edit_sigma_label, "sim.sigma.unit", "sim.sigma.help")

        self.edit_De = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.edit_De.setRange(*config.constants.SLIDER_RANGES['De'])
        self.edit_De.setValue(200)
        self.edit_De.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.edit_De_label = QtWidgets.QLabel("20.0")
        self.edit_De.sliderMoved.connect(lambda v: self.edit_De_label.setText(f"{v/10:.1f}"))
        self.edit_De.valueChanged.connect(lambda v: (self.edit_De_label.setText(f"{v/10:.1f}"), self._on_settings_changed()))
        add_slider_row("De", self.edit_De, self.edit_De_label, "sim.De.unit", "sim.De.help")

        self.edit_a = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.edit_a.setRange(*config.constants.SLIDER_RANGES['a'])
        self.edit_a.setValue(50)
        self.edit_a.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.edit_a_label = QtWidgets.QLabel("0.5")
        self.edit_a.sliderMoved.connect(lambda v: self.edit_a_label.setText(f"{v/100:.2f}"))
        self.edit_a.valueChanged.connect(lambda v: (self.edit_a_label.setText(f"{v/100:.2f}"), self._on_settings_changed()))
        add_slider_row("a", self.edit_a, self.edit_a_label, "sim.a.unit", "sim.a.help")

        self.edit_r0 = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.edit_r0.setRange(*config.constants.SLIDER_RANGES['r0'])
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

        self.btn_draw = QtWidgets.QPushButton()
        self.btn_draw.clicked.connect(self._enter_draw_mode)
        self.btn_draw.setMinimumSize(100, 50)
        self.btn_draw.setStyleSheet("""
            QPushButton {
                font-size:16pt; 
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
        self.btn_clear.setMinimumSize(120, 50)
        self.btn_clear.setStyleSheet("""
            QPushButton {
                font-size:16pt; 
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
        self.btn_particle_select.setMinimumSize(120, 50)
        self.btn_particle_select.setStyleSheet("""
            QPushButton {
                font-size:16pt; 
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
        self.btn_run.setMinimumSize(120, 50)
        self.btn_run.setStyleSheet("""
            QPushButton {
                font-size:16pt; 
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
        self.btn_reset_data.setMinimumSize(120, 50)
        self.btn_reset_data.setStyleSheet("""
            QPushButton {
                font-size:16pt; 
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
        self.btn_back.setMinimumHeight(40)
        self.btn_back.setStyleSheet("""
            QPushButton {
                font-size:16pt; 
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
        canvas_container.addLayout(hist_container, stretch=2)

        # Верхняя панель для анимации с двумя рядами кнопок
        top_bar = QtWidgets.QVBoxLayout()
        
        # Первый ряд кнопок
        top_bar_row1 = QtWidgets.QHBoxLayout()
        top_bar_row1.addStretch(1)
        
        self.btn_toggle_settings_small = QtWidgets.QPushButton()
        self.btn_toggle_settings_small.setFixedHeight(40)
        self.btn_toggle_settings_small.setMinimumWidth(120)
        self.btn_toggle_settings_small.setStyleSheet("""
            QPushButton {
                font-size:16pt; 
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
        top_bar_row1.addWidget(self.btn_toggle_settings_small)

        top_bar_row1.addSpacing(6)
        self.btn_draw.setFixedHeight(40)
        self.btn_draw.setMinimumWidth(120)
        self.btn_draw.setStyleSheet("""
            QPushButton {
                font-size:16pt; 
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
        top_bar_row1.addWidget(self.btn_draw)
        
        top_bar_row1.addSpacing(6)
        self.btn_clear.setFixedHeight(40)
        self.btn_clear.setMinimumWidth(120)
        self.btn_clear.setStyleSheet("""
            QPushButton {
                font-size:16pt; 
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
        top_bar_row1.addWidget(self.btn_clear)
        
        top_bar_row1.addStretch(1)
        
        # Второй ряд кнопок
        top_bar_row2 = QtWidgets.QHBoxLayout()
        top_bar_row2.addStretch(1)
        
        self.btn_particle_select.setFixedHeight(40)
        self.btn_particle_select.setMinimumWidth(120)
        self.btn_particle_select.setStyleSheet("""
            QPushButton {
                font-size:16pt; 
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
        top_bar_row2.addWidget(self.btn_particle_select)
        
        top_bar_row2.addSpacing(6)
        self.btn_run.setFixedHeight(40)
        self.btn_run.setMinimumWidth(120)
        self.btn_run.setStyleSheet("""
            QPushButton {
                font-size:16pt; 
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
        top_bar_row2.addWidget(self.btn_run)
        
        top_bar_row2.addSpacing(6)
        self.btn_reset_data.setFixedHeight(40)
        self.btn_reset_data.setMinimumWidth(120)
        self.btn_reset_data.setStyleSheet("""
            QPushButton {
                font-size:16pt; 
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
        top_bar_row2.addWidget(self.btn_reset_data)
        top_bar_row2.addStretch(1)
        
        top_bar.addLayout(top_bar_row1)
        top_bar.addSpacing(4)
        top_bar.addLayout(top_bar_row2)
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
        self.hist_title.setStyleSheet("font-size:22pt; font-weight:bold; color:#1a1a1a;")
        hist_top_bar.addWidget(self.hist_title)

        self.btn_switch_display_mode = QtWidgets.QPushButton()
        self.btn_switch_display_mode.clicked.connect(self._switch_display_mode)
        self.btn_switch_display_mode.setMinimumSize(170, 28)
        self.btn_switch_display_mode.setStyleSheet("""
            QPushButton {
                font-size:16pt; 
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
            ax.tick_params(labelsize=11)
            ax.set_facecolor('#f9f9f9')
        self.ax_histx, self.ax_histy, self.ax_histd = axes
        self.fig_hist.patch.set_facecolor('#ffffff')
        self.fig_hist.tight_layout()
        self.fig_hist.subplots_adjust(left=0.2, top=0.93, bottom=0.07, hspace=0.4)
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
        """Сброс всех накопленных данных и регенерация системы с текущими настройками"""
        # Регенерируем систему с текущими параметрами
        self._reinitialize_system(reset_hist=True)
        
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
        radius = config.constants.DEFAULT_RADIUS
        temp = float(self.edit_T.value()) / 10.0
        dt = config.constants.DEFAULT_DT
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
        self.ax_histx.set_ylabel("p(x)", labelpad=5, fontsize=13)
        self.ax_histx.set_title(f"{s['sim.hist.x']}", fontsize=14, fontweight='bold')
        self.ax_histx.tick_params(labelsize=11)

        if len(self.selected_particles_data['y']) > 0:
            self.ax_histy.hist(self.selected_particles_data['y'], bins=self.bins_count, density=True,
                               color="#f9ad6c", alpha=0.8)
        self.ax_histy.set_ylabel("p(y)", labelpad=5, fontsize=13)
        self.ax_histy.set_title(f"{s['sim.hist.y']}", fontsize=14, fontweight='bold')
        self.ax_histy.tick_params(labelsize=11)

        # --- THEORETICAL p(x) ---
        # v = self.system.vessel
        # vk = self._vessel_kind_to_key(v.kind)

        # xs = np.linspace(self.ax_histx.get_xlim()[0], self.ax_histx.get_xlim()[1], 400)

        # if vk == "circle":
        #     cx, cy, R = v.circle
        #     mask = np.abs(xs - cx) <= R
        #     px = np.zeros_like(xs)
        #     px[mask] = (2 / (np.pi * R * R)) * np.sqrt(R * R - (xs[mask] - cx)**2)
        #     self.ax_histx.plot(xs, px, linewidth=2, alpha=0.9)
        # elif vk == "rect":
        #     xmin, ymin, xmax, ymax = v.rect
        #     L = xmax - xmin
        #     px = np.ones_like(xs) * (1 / L)
        #     self.ax_histx.plot(xs, px, linewidth=2, alpha=0.9)


        # Разные режимы отображения для третьего графика
        if self.distance_display_mode == 0:  # Обычное расстояние
            if len(self.selected_particles_data['distances']) > 0:
                self.ax_histd.hist(self.selected_particles_data['distances'], bins=self.bins_count, density=True,
                                   color=config.constants.HISTOGRAM_COLORS['distance'], alpha=0.85)
            
            # --- THEORETICAL p(r) for uniform distribution in a vessel ---
            v = self.system.vessel
            vk = self._vessel_kind_to_key(v.kind)
            
            rs = np.linspace(self.ax_histd.get_xlim()[0], self.ax_histd.get_xlim()[1], 400)
            
            if vk == "circle":
                # For a circle of radius R, p(r) = 2*r/R^2 for 0 <= r <= 2*R
                cx, cy, R = v.circle
                pr = np.zeros_like(rs)
                mask = (rs >= 0) & (rs <= 2*R)
                pr[mask] = (2 * rs[mask]) / (R * R)
                self.ax_histd.plot(rs, pr, linewidth=2, alpha=0.9, color='red', label='Теория')
            elif vk == "rect":
                # For a rectangle, triangular distribution
                xmin, ymin, xmax, ymax = v.rect
                L = xmax - xmin
                H = ymax - ymin
                # Maximum possible distance is sqrt(L^2 + H^2)
                max_dist = np.sqrt(L*L + H*H)
                pr = np.zeros_like(rs)
                mask = (rs >= 0) & (rs <= max_dist)
                # Triangular distribution: p(r) = 4*r/max_dist^2 for r <= max_dist/2, 
                # and p(r) = 4*(max_dist-r)/max_dist^2 for r > max_dist/2
                mid = max_dist / 2.0
                for i in np.where(mask)[0]:
                    r = rs[i]
                    if r <= mid:
                        pr[i] = (4.0 * r) / (max_dist * max_dist)
                    else:
                        pr[i] = (4.0 * (max_dist - r)) / (max_dist * max_dist)
                self.ax_histd.plot(rs, pr, linewidth=2, alpha=0.9, color='red', label='Теория')
            
            self.ax_histd.set_ylabel("p(r)", labelpad=5, fontsize=13)
            self.ax_histd.set_xlabel("r", fontsize=13)
            self.ax_histd.set_title(f"{s['sim.hist.r']}", fontsize=14, fontweight='bold')

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

            self.ax_histd.set_ylabel("p(Δ)", labelpad=5, fontsize=13)
            self.ax_histd.set_xlabel("Δ", fontsize=13)
            self.ax_histd.set_title(s.get("sim.display_mode.projections", "Проекции расстояния"), fontsize=14, fontweight='bold')

            # Добавляем легенду только если есть данные
            if has_dx or has_dy:
                self.ax_histd.legend()

        else:  # Расстояние со знаком
            if len(self.selected_particles_data['signed_distance']) > 0:
                self.ax_histd.hist(self.selected_particles_data['signed_distance'], bins=self.bins_count, density=True,
                                   color="#8d6e63", alpha=0.85)
            self.ax_histd.set_ylabel("p(rₛ)", labelpad=5, fontsize=13)
            self.ax_histd.set_xlabel("rₛ", fontsize=13)
            self.ax_histd.set_title(s.get("sim.display_mode.signed", "Расстояние со знаком"), fontsize=14, fontweight='bold')
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
                                        radius=self.system.visual_radius, facecolor=config.constants.PARTICLE_COLOR,
                                        edgecolor="black", linewidth=0.5, alpha=0.8)
                self.ax_anim.add_patch(circle)
                self.particle_circles.append(circle)

        self.canvas_anim.draw_idle()

    def _init_simulation(self, N=config.constants.DEFAULT_N, radius=config.constants.DEFAULT_RADIUS, temp=config.constants.DEFAULT_TEMP, dt=config.constants.DEFAULT_DT,
                         vessel_kind="Прямоугольник", poly=None,
                         potential_params=None, mass=config.constants.DEFAULT_MASS, enable_collisions=True, interaction_step=1):
        if potential_params is None:
            potential_params = PotentialParams()
        if vessel_kind in ("rect", "Прямоугольник"):
            vessel = Vessel(kind="Прямоугольник", rect=config.constants.VESSEL_RECT_BOUNDS)
        elif vessel_kind in ("circle", "Круг"):
            vessel = Vessel(kind="Круг", circle=config.constants.VESSEL_CIRCLE_CENTER_AND_RADIUS)
        elif vessel_kind in ("poly", "Многоугольник"):
            if poly is None:
                poly = np.array(config.constants.VESSEL_POLYGON_DEFAULT)
            vessel = Vessel(kind="Многоугольник", poly=poly)
        else:
            vessel = Vessel(kind="Прямоугольник", rect=config.constants.VESSEL_RECT_BOUNDS)
        self.system = System(vessel=vessel, N=N, radius=radius, visual_radius=radius, temp=temp, dt=dt,
                             params=potential_params, mass=mass, enable_collisions=enable_collisions,
                             interaction_step=interaction_step)
        self.system.seed()

        if hasattr(self, 'particle_circles'):
            for c in self.particle_circles: c.remove()
        self.particle_circles = []
        for i in range(self.system.n()):
            circle = patches.Circle((self.system.pos[i, 0], self.system.pos[i, 1]),
                                    radius=self.system.visual_radius, facecolor=config.constants.PARTICLE_COLOR,
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
                    circle.set_facecolor(config.constants.SELECTED_PARTICLE_COLOR)  # Красный для выбранных
                    circle.set_edgecolor("darkred")
                    circle.set_linewidth(2)
                else:
                    circle.set_facecolor(config.constants.PARTICLE_COLOR)  # Синий для остальных
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
        self.ax_histx.set_ylabel("p(x)", labelpad=5, fontsize=13)
        self.ax_histx.set_title(f"{s['sim.hist.x']}", fontsize=14, fontweight='bold')
        self.ax_histx.tick_params(labelsize=11)

        if len(self.selected_particles_data['y']) > 0:
            self.ax_histy.hist(self.selected_particles_data['y'], bins=self.bins_count, density=True,
                               color="#f9ad6c", alpha=0.8)
        self.ax_histy.set_ylabel("p(y)", labelpad=5, fontsize=13)
        self.ax_histy.set_title(f"{s['sim.hist.y']}", fontsize=14, fontweight='bold')
        self.ax_histy.tick_params(labelsize=11)



        # Разные режимы отображения для третьего графика
        if self.distance_display_mode == 0:  # Обычное расстояние
            if len(self.selected_particles_data['distances']) > 0:
                self.ax_histd.hist(self.selected_particles_data['distances'], bins=self.bins_count, density=True,
                                   color=config.constants.HISTOGRAM_COLORS['distance'], alpha=0.85)
            self.ax_histd.set_ylabel("p(r)", labelpad=5, fontsize=13)
            self.ax_histd.set_xlabel("r", fontsize=13)
            self.ax_histd.set_title(f"{s['sim.hist.r']}", fontsize=14, fontweight='bold')
            self.ax_histd.tick_params(labelsize=11)

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

            self.ax_histd.set_ylabel("p(Δ)", labelpad=5, fontsize=13)
            self.ax_histd.set_xlabel("Δ", fontsize=13)
            self.ax_histd.set_title(s.get("sim.display_mode.projections", "Проекции расстояния"), fontsize=14, fontweight='bold')
            self.ax_histd.tick_params(labelsize=11)

            # Добавляем легенду только если есть данные
            if has_dx or has_dy:
                self.ax_histd.legend()

        else:  # Расстояние со знаком
            if len(self.selected_particles_data['signed_distance']) > 0:
                self.ax_histd.hist(self.selected_particles_data['signed_distance'], bins=self.bins_count, density=True,
                                   color="#8d6e63", alpha=0.85)
            self.ax_histd.set_ylabel("p(rₛ)", labelpad=5, fontsize=13)
            self.ax_histd.set_xlabel("rₛ", fontsize=13)
            self.ax_histd.set_title(s.get("sim.display_mode.signed", "Расстояние со знаком"), fontsize=14, fontweight='bold')
            self.ax_histd.tick_params(labelsize=11)
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
        self.ax_histx.set_ylabel("p(x)", labelpad=5, fontsize=13)
        self.ax_histx.set_title(s["sim.hist.x"], fontsize=14, fontweight='bold')
        self.ax_histx.tick_params(labelsize=11)

        if len(self.accumulated_y) > 0:
            self.ax_histy.hist(self.accumulated_y, bins=self.bins_count, density=True, color="#f9ad6c", alpha=0.8)
        self.ax_histy.set_ylabel("p(y)", labelpad=5, fontsize=13)
        self.ax_histy.set_title(s["sim.hist.y"], fontsize=14, fontweight='bold')
        self.ax_histy.tick_params(labelsize=11)

        # Разные режимы отображения для третьего графика
        if self.distance_display_mode == 0:  # Обычное расстояние
            if len(self.accumulated_distances) > 0:
                self.ax_histd.hist(self.accumulated_distances, bins=self.bins_count, density=True, color=config.constants.HISTOGRAM_COLORS['distance'],
                                   alpha=0.85)
            
            self.ax_histd.set_ylabel("p(r)", labelpad=5, fontsize=13)
            self.ax_histd.set_xlabel("r", fontsize=13)
            self.ax_histd.set_title(s["sim.hist.r"], fontsize=14, fontweight='bold')
            self.ax_histd.tick_params(labelsize=11)

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

            self.ax_histd.set_ylabel("p(Δ)", labelpad=5, fontsize=13)
            self.ax_histd.set_xlabel("Δ", fontsize=13)
            self.ax_histd.set_title(s.get("sim.display_mode.projections", "Проекции расстояния"), fontsize=14, fontweight='bold')
            self.ax_histd.tick_params(labelsize=11)

            # Добавляем легенду только если есть данные
            if has_dx or has_dy:
                self.ax_histd.legend()

        else:  # Расстояние со знаком
            if len(self.accumulated_signed) > 0:
                self.ax_histd.hist(self.accumulated_signed, bins=self.bins_count, density=True,
                                   color="#8d6e63", alpha=0.85)
            self.ax_histd.set_ylabel("p(rₛ)", labelpad=5, fontsize=13)
            self.ax_histd.set_xlabel("rₛ", fontsize=13)
            self.ax_histd.set_title(s.get("sim.display_mode.signed", "Расстояние со знаком"), fontsize=14, fontweight='bold')
            self.ax_histd.tick_params(labelsize=11)
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
        self.ax_anim.set_title(s["sim.anim.title"], fontsize=18, fontweight='bold', pad=20)
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
