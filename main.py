#cmd + shift P -> Python: Select Interpreter -> computer_vision/bin/python
#python3 -m venv computer_vision
#pip install -r requirements.txt

import sys
from PySide6.QtWidgets import QApplication
from gui import ComputerVisionGUI

left_image_path = "Images/example002L.bmp"
right_image_path = "Images/example002R.bmp"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ComputerVisionGUI(left_path=left_image_path, right_path=right_image_path)
    win.show()
    sys.exit(app.exec())