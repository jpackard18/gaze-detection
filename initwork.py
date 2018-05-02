import numpy as np
from network import sigmoid
import cv2


class Initwork:

    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def evaluate(self, img, faces):
        results = []
        for face_rect in faces:
            if len(face_rect) < 2:
                results.append(0)
                continue
            eyes = [cv2.resize(img[e.y: e.y + e.height, e.x:e.x + e.width], (32, 32)) for e in face_rect]
            # make black pixels grey in prior so we can tell them apart from the black ellipse mask
            eyes[0][eyes[0] == 0] = 1
            eyes[1][eyes[1] == 0] = 1
            # draw an ellipse that covers the pixels outside 15 pixels from center horizontally
            # and outside 11 pixels from center vertically
            # cv2.ellipse(ResultImage, (centerX,centerY), (width,height), startAngle, endAngle, angle, color, lineThickness)
            cv2.ellipse(eyes[0], (15, 15), (26, 21), 0, 0, 360, 0, 20)
            cv2.ellipse(eyes[1], (15, 15), (26, 21), 0, 0, 360, 0, 20)
            # combine the eyes horizontally from left eye to right eye
            flattened_eyes = np.append(eyes[0], eyes[1], axis=1).reshape(2048, 1)
            divided_eyes = np.true_divide(flattened_eyes, 255.0)
            result = self.feedforward(divided_eyes)[0][0]
            results.append(result)
        return results
