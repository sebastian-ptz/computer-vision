from PySide6.QtWidgets import (
    QWidget, QLabel, QHBoxLayout, QVBoxLayout,
    QLineEdit, QFormLayout, QMenuBar, QMenu, QRadioButton, QSizePolicy, QCheckBox
)
from PySide6.QtGui import QPixmap, QImage, QIntValidator, QAction
from PySide6.QtCore import Qt
from manager import AppManager
import numpy as np
import cv2

# GUI Class
class ComputerVisionGUI(QWidget):
    def __init__(self, left_path=None, right_path=None, w=480, h=360):
        super().__init__()
        self.setWindowTitle("Computer Vision")
        self.left_path = left_path
        self.right_path = right_path    
        self.left_img = None
        self.right_img = None
        self.show_features = False
        self.display_mode = "original"

        self.setup_menu()
        self._setup_widgets(w, h)
        self._setup_layouts()

        self.manager = AppManager(self)
        self.load_images()

    # Setup
    def setup_menu(self):
        self.menu_bar = QMenuBar(self)
        file_menu = QMenu("File", self)
        load_action = QAction("Load Images", self)
        clear_action = QAction("Clear", self)
        file_menu.addAction(load_action)
        file_menu.addAction(clear_action)
        self.menu_bar.addMenu(file_menu)
        load_action.triggered.connect(self.load_images)
        clear_action.triggered.connect(self.clear)

    def _setup_widgets(self, width, height):
        self.left_widget = QLabel()
        self.right_widget = QLabel()
        self.left_widget.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.right_widget.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.left_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.max_corners_input = QLineEdit("500")
        self.max_corners_input.setFixedWidth(60)
        self.max_corners_input.setValidator(QIntValidator(1, 50000, self))

        self.max_matches_input = QLineEdit("200")
        self.max_matches_input.setFixedWidth(60)
        self.max_matches_input.setValidator(QIntValidator(1, 10000, self))
        self._last_max_matches = 200

        # Alpha input for transparency
        self.alpha_input = QLineEdit("0.5")
        self.alpha_input.setFixedWidth(60)
        from PySide6.QtGui import QDoubleValidator
        self.alpha_input.setValidator(QDoubleValidator(0.0, 1.0, 2, self))

        self.radio_trace = QRadioButton("Edges")
        self.radio_features = QRadioButton("Features")
        self.radio_matches = QRadioButton("Matches")

        self.checkbox_remove_extrema = QCheckBox("Remove Extrema")
        self.checkbox_remove_extrema.setChecked(False)
        self.checkbox_remove_extrema.stateChanged.connect(self._on_control_changed)

        # Connect signals
        self.radio_trace.toggled.connect(self._on_control_changed)
        self.radio_features.toggled.connect(self._on_control_changed)
        self.radio_matches.toggled.connect(self._on_control_changed)
        self.max_corners_input.returnPressed.connect(self._on_control_changed)
        self.max_matches_input.returnPressed.connect(self._on_control_changed)
        self.alpha_input.returnPressed.connect(self._on_control_changed)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.manager.handle_control_change()


    def _setup_layouts(self):
        imgs = QHBoxLayout()
        imgs.setSpacing(0)
        imgs.addWidget(self.left_widget, alignment=Qt.AlignRight)
        imgs.addWidget(self.right_widget, alignment=Qt.AlignLeft)


        input_row = QHBoxLayout()
        input_row.setAlignment(Qt.AlignHCenter)
        input_row.setSpacing(8)
        label_corners = QLabel("Corners:")
        label_corners.setFixedWidth(60)
        input_row.addWidget(label_corners)
        input_row.addWidget(self.max_corners_input)
        label_matches = QLabel("Matches:")
        label_matches.setFixedWidth(60)
        input_row.addWidget(label_matches)
        input_row.addWidget(self.max_matches_input)
        label_alpha = QLabel("Alpha:")
        label_alpha.setFixedWidth(50)
        input_row.addWidget(label_alpha)
        input_row.addWidget(self.alpha_input)

        radio_layout = QHBoxLayout()
        radio_layout.setAlignment(Qt.AlignHCenter)
        radio_layout.setSpacing(12)
        radio_layout.addWidget(self.radio_trace)
        radio_layout.addWidget(self.radio_features)
        radio_layout.addWidget(self.radio_matches)
        radio_layout.addWidget(self.checkbox_remove_extrema)

        layout = QVBoxLayout(self)
        layout.setMenuBar(self.menu_bar)
        layout.addLayout(imgs, stretch=1)
        layout.addLayout(input_row)
        layout.addLayout(radio_layout)
    def _get_alpha(self):
        try:
            val = float(self.alpha_input.text())
            return min(max(val, 0.0), 1.0)
        except Exception:
            return 0.5

    def _get_max_matches(self):
        try:
            val = int(self.max_matches_input.text())
            return val if val > 0 else 200
        except Exception:
            return 200

    # Events 
    def _on_control_changed(self, *args, **kwargs):
        self.manager.handle_control_change()

    # Publid Methods
    def load_images(self):
        if self.left_path:
            self.left_img = self._load_image(self.left_path)
            self._set_pixmap(self.left_widget, self.left_img)
        if self.right_path:
            self.right_img = self._load_image(self.right_path)
            self._set_pixmap(self.right_widget, self.right_img)
        self.manager.handle_control_change()

    def clear(self):
        self.left_img = self.right_img = None
        self.left_path = self.right_path = None
        self.show_features = False
        self.radio_features.setChecked(False)
        self.left_widget.clear()
        self.right_widget.clear()
        self.max_corners_input.setText("500")

    # Helpers
    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return img if img is not None else None

    def _set_pixmap(self, label, img):
        if img is None:
            label.setPixmap(QPixmap())
            return
        
        label_width, label_height = label.width(), label.height()
        img_height, img_width = img.shape[:2]

        scale = min(label_width / img_width, label_height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA 
                             if scale < 1.0 else cv2.INTER_LINEAR)
        
        # Create a transparent canvas and paste the resized image at the correct side
        canvas = np.zeros((label_height, label_width, 3), dtype=np.uint8)
        if label.alignment() & Qt.AlignRight:
            x0 = label_width - new_width
        else:
            x0 = 0
        y0 = (label_height - new_height) // 2
        canvas[y0:y0+new_height, x0:x0+new_width] = resized
        label.setPixmap(qpixmap(canvas))

    def _get_max_corners(self):
        try:
            val = int(self.max_corners_input.text())
            return val if val > 0 else 500
        except Exception:
            return 500

def qpixmap(img):
    if img is None:
        return QPixmap()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    qimg = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())