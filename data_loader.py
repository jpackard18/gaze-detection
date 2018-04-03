import pickle


# given a list of tuples of (eyes_image_nparray, resulting_vector), saves it on disk
def save(list_of_tuples, output_file_path):
    file = open(output_file_path, "wb")
    pickle.dump(list_of_tuples, file)


# returns a list of tuples of (eyes_image_nparray, resulting_vector) read from the disk
def load(file_path):
    file = open(file_path, "rb")
    list_of_tuples = pickle.load(file)
    return list_of_tuples
