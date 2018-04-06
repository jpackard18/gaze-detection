import os
import argparse
import cv2
import numpy as np
from numpy import zeros
from eye_detection import grab_eyes
from array import vectorized_result
from data_loader import save
from datetime import datetime
from multiprocessing import Process, Queue


NUM_CPU_CORES = 1


class InputImage:

    def __init__(self, path, name):
        self.path = path
        self.name = name
        file_data = name[:-4].split("_")
        self.horizontal = int(file_data[4][:-1])
        self.vertical = int(file_data[3][:-1])

    def extract_eyes(self):
        img = cv2.resize(cv2.imread(self.path), (1440, 960))
        eyes = grab_eyes(img)
        if len(eyes) != 2:
            return None
        inv = []
        inv.append(255-eyes[0])
        inv.append(255-eyes[1])
        eyes[0][eyes[0] == 0] = 1
        eyes[1][eyes[1] == 0] = 1
        # draw an ellipse that covers the pixels outside 15 pixels from center horizontally
        # and outside 11 pixels from center vertically
        # cv2.ellipse(ResultImage, (centerX,centerY), (width,height), startAngle, endAngle, angle, color, lineThickness)
        cv2.ellipse(eyes[0], (15,15), (26,21), 0, 0, 360, 0, 20)
        cv2.ellipse(eyes[1], (15,15), (26,21), 0, 0, 360, 0, 20)
        cv2.imshow("Eye 1", eyes[0])
        cv2.imshow("Eye 2", eyes[1])
        cv2.waitKey()
        converted_eyes = zeros((2048, 1))
        for i in range(2):
            for x in range(eyes[i].shape[0]):
                for y in range(eyes[i].shape[1]):
                    converted_eyes[x*y + i*1024] = eyes[i][x][y] / 255.0
        return converted_eyes


def process_image_list(image_list, queue, worker_index):
    results = []
    # start with work index and skip the number of CPU cores
    for i in range(worker_index, len(image_list), NUM_CPU_CORES):
        if i % NUM_CPU_CORES == worker_index:
            inp = image_list[i]
            eyes = inp.extract_eyes()
            if eyes is None:
                print("Could not locate two eyes in the frame >.( \tfile: {}".format(inp.path))
                continue
            results.append(
                (eyes,
                 vectorized_result(inp.vertical, inp.horizontal))
            )
            print("Progress: {}%".format(round(i / len(image_list) * 100)))
    queue.put(results)


def create_training_data(gaze_set_path, output_file_path):
    training_data = []
    input_images = []
    for dirpath, dirnames, filenames in os.walk(gaze_set_path):
        for filename in filenames:
            if not filename.endswith(".jpg"):
                continue
            input_image = InputImage(os.path.join(dirpath, filename), filename)
            # Ignore horizontal gazes of +-15
            if abs(input_image.horizontal) == 15:
                continue
            input_images.append(input_image)
    print(len(input_images))
    processes = []
    queue = Queue(NUM_CPU_CORES)
    for core in range(NUM_CPU_CORES):
        # p = Process(target=process_image_list, args=(input_images[:4], queue, core))
        p = Process(target=process_image_list, args=(input_images, queue, core))
        p.start()
        processes.append(p)
    print("Done with appending processes")
    for i in range(NUM_CPU_CORES):
        training_data.extend(queue.get())
        print("Got queue")
    print("Size: {}".format(len(training_data)))
    print("Done, saving...")
    save(training_data, output_file_path)
    print("Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaze_path", help="Full path of the Columbia Gaze Data Set", default="/Users/lol/Downloads/Columbia Gaze Data Set")
    parser.add_argument("--output_file", help="Training Data Output File Path", default="training_{}.pkl".format(datetime.now().isoformat()))
    args = parser.parse_args()
    print(args.gaze_path)
    print(args.output_file)
    create_training_data(args.gaze_path, args.output_file)
