import sys
import PyQt5
from gui import *


app = PyQt5.QtWidgets.QApplication(sys.argv)

window = VideoWindow()
window.setWindowTitle("Gaze Detection V 0.1.1")
window.show()

# run the app
return_code = app.exec_()
print("stopped")
sys.exit(return_code)
