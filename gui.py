from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


import numpy as np
import cv2

from camera import *
from eye_detection import detect_eyes, grab_eyes, grab_faces

KEEPASPECTRATIOPLEASE = 1

videoWindowSize = 0


class EyeDetectionWorker(QThread):

    def __init__(self, cap, imageLabelDisplay, window):
        QThread.__init__(self)
        self.cap = cap
        self.imageLabelDisplay = imageLabelDisplay
        self.window = window
        self.stopped = False

    def start(self):
        super(EyeDetectionWorker, self).start()
        self.stopped = False

    def stop(self):
        self.stopped = True

    # grabs an image and processes it
    def run(self):
        while not self.stopped:
            start_time = time.time()
            ret, frame = self.cap.read()
            result_img, eyes = detect_eyes(frame, draw_rects=True)
            #print(eyes)
            qImage = VideoWindow.convertMatToQImage(result_img)
            self.imageLabelDisplay.setPixmap(QPixmap.fromImage(qImage))
            self.imageLabelDisplay.show()
            time_delta = time.time() - start_time
            self.window.setFaceNum()
            print("Time taken for eye detection: " + str(time_delta))
        self.quit()



class VideoWindow(QMainWindow):

    def closeEvent(self, event):
        self.onCloseWindow()

    def onCloseWindow(self):
        self.worker.stop()
        self.cap.release()
        self.close()


    def __init__(self, parent=None):
        super(VideoWindow, self).__init__(parent)

        self.resize(1000, 600)

        global videoWindowSize

        videoWindowSize = self.height()

        self.cap = cv2.VideoCapture(0)
        # quit on alt+f4 or ctrl+w
        self.shortcut = QShortcut(QKeySequence.Close, self)
        self.shortcut.activated.connect(self.onCloseWindow)

        self.shortcut2 = QShortcut(QKeySequence("Shift+Left"), self)
        self.shortcut2.activated.connect(self.leftShift)
        self.shortcut3 = QShortcut(QKeySequence("Shift+Right"), self)
        self.shortcut3.activated.connect(self.rightShift)
        self.faceTester = QShortcut(QKeySequence("Space"), self)
        self.faceTester.activated.connect(self.setFaceNum)

        # Create a widget for window contents
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        # create the top status display row
        # create the main layout that contains the viewfinder and controls
        main_layout = QGridLayout()

        #Initialize all the widgets
        self.imageLabel = QLabel()
        self.startStopButton = QPushButton()
        self.startStopButton.setFixedWidth(videoWindowSize / 2)
        self.emptyBox = QLabel()
        self.giphy = QMovie("giphy.gif")
        self.emptyBox = QLabel()
        self.feedLabel = QLabel()
        font = QFont("century gothic", 20)
        self.feedLabel.setFont(font)
        self.boxLabel = QLabel()
        self.gazeNumber = QLabel()
        #self.cameraSelect = QComboBox()

        #Add all widgets to the grid layout and set their positions
        main_layout.addWidget(self.feedLabel, 0, 0, Qt.AlignRight)
        main_layout.addWidget(self.boxLabel, 0, 4, Qt.AlignLeft)
        main_layout.addWidget(self.imageLabel, 1, 0, 1, 2)
        main_layout.addWidget(self.emptyBox, 1, 3, 1, 2)
        main_layout.addWidget(self.gazeNumber, 2, 2, Qt.AlignCenter)
        main_layout.addWidget(self.startStopButton, 3, 2, 3, 1)
        #main_layout.addWidget(self.cameraSelect, 4, 2, Qt.AlignCenter)

        #Set the text for widgets
        self.startStopButton.setText("Stop")
        self.feedLabel.setText("Live Feed")
        self.boxLabel.setText("Nothing")
        self.gazeNumber.setText(self.setFaceNum())

        #Connect Buttons to their respective methods
        self.startStopButton.clicked.connect(self.stop)

        #Add the gif
        self.emptyBox.setMovie(self.giphy)
        self.giphy.start()

        # apply the main layout
        central_widget.setLayout(main_layout)

        #automatically capture stills
        self.worker = EyeDetectionWorker(self.cap, self.imageLabel, self)
        self.worker.start()

    # starts both the gif and the video feed
    def start(self):
        self.worker.start()
        self.startStopButton.clicked.connect(self.stop)
        self.startStopButton.setText("Stop")
        self.giphy.start()


    #pauses the gif and the video feed
    def stop(self):
        self.worker.stop()
        self.startStopButton.clicked.connect(self.start)
        self.startStopButton.setText("Start")
        self.giphy.stop()

    #gets the largest number image in the filed
    def get_img_num(self):
        num = 0
        for dirpath, dirnames, filenames in os.walk('EYES'):
            for filename in filenames:
                if not filename.endswith(".jpg"):
                    continue
                file_data = filename[:-4].split("_")
                temp_num = int(file_data[0])
                if temp_num > num:
                    num = temp_num
        return num + 1

    #captures an image and labels it based on looking/not
    def capture_img(self, is_looking):
        num = self.get_img_num()
        color_img = self.cap.read()[1]
        eyes = grab_eyes(color_img)
        if len(eyes) != 2:
            print("Unable to find two eyes in the frame >.(")
            return
        img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        if is_looking:
            file_string = 'EYES/{}_NA_NA_0V_0H.jpg'.format(num)
        else:
            file_string = 'EYES/{}_NA_NA_10V_15H.jpg'.format(num)
        cv2.imwrite(file_string, img)

    def leftShift(self):
        self.capture_img(False)

    def rightShift(self):
        self.capture_img(True)



    def setFaceNum(self):
        numFaces = len(grab_faces(self.cap.read()[1]))
        self.gazeNumber.setText("Number of Gazes: %d / %d" % (0, numFaces))
        return "Number of Gazes: %d / %d" % (0, numFaces)

    @staticmethod
    def convertQImageToMat(qImage):
        '''  Converts a QImage into an opencv MAT format  '''
        qImage = qImage.convertToFormat(4)
        width = qImage.width()
        height = qImage.height()
        
        ptr = qImage.bits()
        ptr.setsize(qImage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
        return arr


    @staticmethod
    def convertMatToQImage(cv_mat):
        '''  Converts an opencv MAT image to a QImage  '''

        global videoWindowSize

        #print("aa")
        #print(videoWindowSize)

        # convert to rgb
        cv_mat = cv2.cvtColor(cv_mat, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_mat.shape
        bytesPerLine = 3 * width
        qImage = QImage(cv_mat.data, width, height, bytesPerLine, QImage.Format_RGB888)
        qImage = qImage.scaled(videoWindowSize / 1.5, videoWindowSize / 1.5, KEEPASPECTRATIOPLEASE)
        return qImage

