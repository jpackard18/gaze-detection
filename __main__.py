import sys
import PyQt5
from gui import *
from data_loader import load
from initwork import Initwork


app = PyQt5.QtWidgets.QApplication(sys.argv)

data = load('output_network.pkl')
network = Initwork(data['weights'], data['biases'])

window = VideoWindow(network)
window.setWindowTitle("Gaze Detection V 0.1.1")
window.show()

# run the app
return_code = app.exec_()
print("stopped")
sys.exit(return_code)
