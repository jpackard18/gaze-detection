import os
import argparse
import numpy as np
import cv2
from eye_detection import grab_eyes
from output_vectorization import vectorized_result, vectorized_result_2
from data_loader import save
from datetime import datetime
from multiprocessing import Process, Queue

# enter in the number of physical cores here (it's generally enough)
NUM_CPU_CORES = 6


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
        # make black pixels grey in prior so we can tell them apart from the black ellipse mask
        eyes[0][eyes[0] == 0] = 1
        eyes[1][eyes[1] == 0] = 1
        # draw an ellipse that covers the pixels outside 15 pixels from center horizontally
        # and outside 11 pixels from center vertically
        # cv2.ellipse(ResultImage, (centerX,centerY), (width,height), startAngle, endAngle, angle, color, lineThickness)
        cv2.ellipse(eyes[0], (15,15), (26,21), 0, 0, 360, 0, 20)
        cv2.ellipse(eyes[1], (15,15), (26,21), 0, 0, 360, 0, 20)
        # combine the eyes horizontally from left eye to right eye
        converted_eyes = np.append(eyes[0], eyes[1], axis=1).reshape(2048, 1)
        return np.true_divide(converted_eyes, 255.0)

def process_image_list(image_list, queue, worker_index, is_v2=False):
    results = []
    # start with work index and skip the number of CPU cores
    for i in range(worker_index, len(image_list), NUM_CPU_CORES):
        if i % NUM_CPU_CORES == worker_index:
            inp = image_list[i]
            eyes = inp.extract_eyes()
            if eyes is None:
                print("Could not locate two eyes in the frame >.( \tfile: {}".format(inp.path))
                continue
            if is_v2:
                vr = vectorized_result_2(inp.vertical, inp.horizontal)
            else:
                vr = vectorized_result(inp.vertical, inp.horizontal)
            results.append((eyes, vr))
            print("Progress: {}%".format(round(i / len(image_list) * 100)))
    queue.put(results)


def create_training_data(gaze_set_path, output_file_path, is_v2=False):
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
    print("Found " + str(len(input_images)) + " images.")
    processes = []
    queue = Queue(NUM_CPU_CORES)
    for core in range(NUM_CPU_CORES):
        # process_image_list(input_images, queue, core, is_v2)
        p = Process(target=process_image_list, args=(input_images, queue, core, is_v2))
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
    parser.add_argument("--gaze_path", help="Full path of the Columbia Gaze Data Set",
                        default="/Users/lol/Downloads/Columbia Gaze Data Set")
    parser.add_argument("--output_file", help="Training Data Output File Path",
                        default="training_{}.pkl".format(datetime.now().strftime("%Y%m%d_%H%M")))
    parser.add_argument("--v2", help="v2 is the training data with only one output neuron. Do you want this? (yes/no)",
                        default="no")
    args = parser.parse_args()
    if args.v2 == "yes":
        args.output_file = args.output_file.rstrip('.pkl') + "-v2.pkl"
        is_v2 = True
    elif args.v2 == "no":
        is_v2 = False
    else:
        print("unknown input for --is_v2, setting to false...")
        is_v2 = False
    print(args.gaze_path)
    print(args.output_file)
    print(is_v2)
    create_training_data(args.gaze_path, args.output_file, is_v2)
