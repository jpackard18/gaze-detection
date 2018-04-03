import os
import argparse
import cv2
from numpy import zeros
from eye_detection import grab_eyes
from array import vectorized_result
from data_loader import save
from multiprocessing import Process, Queue


NUM_CPU_CORES = 12


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
        # cv2.imshow("Eye 1", eyes[0])
        # cv2.imshow("Eye 2", eyes[1])
        # cv2.waitKey()
        converted_eyes = zeros((2048, 1))
        for eye in eyes:
            for x in range(eye.shape[0]):
                for y in range(eye.shape[1]):
                    converted_eyes[x*y] = eye[x][y] / 255.0
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
    parser.add_argument("gaze_path", help="Full path of the Columbia Gaze Data Set")
    parser.add_argument("output_file", help="Training Data Output File Path")
    args = parser.parse_args()
    create_training_data(args.gaze_path, args.output_file)
