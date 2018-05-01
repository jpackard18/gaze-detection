import cv2
from os.path import realpath, dirname, join


CASCADES_FOLDER = join(dirname(realpath(__file__)), 'cascades')
FACE_CASCADE = cv2.CascadeClassifier(join(CASCADES_FOLDER,
                                          'haarcascade_frontalface_default.xml'))
EYE_CASCADE = cv2.CascadeClassifier(join(CASCADES_FOLDER,
                                         'haarcascade_eye.xml'))

MAX_FACES = 2
MAX_EYES_PER_FACE = 2


class Eye:

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


def detect_eyes(img, draw_rects=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.2, 5)
    eyes_rect = []
    for (x, y, w, h) in faces[:MAX_FACES]:
        if draw_rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        if draw_rects:
            roi_color = img[y:y + h, x:x + w]
        eyes = EYE_CASCADE.detectMultiScale(roi_gray, 1.3, 4)
        for (ex, ey, ew, eh) in eyes[:MAX_EYES_PER_FACE]:
            eyes_rect.append(Eye(ex + x, ey + y, ew, eh))
            if draw_rects:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return img, eyes_rect


def grab_eyes(img):
    img, eyes_rect = detect_eyes(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("eye", img)
    if len(eyes_rect) < 2:
        return []
    eyes = []
    eyes_rect = sorted(eyes_rect, key=lambda rect: rect.x)
    for eye_rect in eyes_rect:
        x = eye_rect.x
        y = eye_rect.y
        w = eye_rect.width
        h = eye_rect.height
        # print("%d %d %d %d"%(x,y,w,h))
        eyes.append(cv2.resize(gray[y:y+h, x:x+w], (32, 32)))
    # cv2.imshow("Example", eyes[0])
    # cv2.imshow("Example 2", eyes[1])
    # cv2.waitKey(0)
    return eyes


def grab_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.2, 5)
    return faces


# img = cv2.imread("images/0001_2m_-15P_-10V_-5H.jpg")
if __name__ == "__main__":
    img = cv2.imread("images/0004_2m_-15P_-10V_-5H.jpg")
    grab_eyes(img)
