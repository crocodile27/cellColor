import cellColor._opencv_config 
from qtpy.QtWidgets import QApplication
from .gui import MainWindow
import sys


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
