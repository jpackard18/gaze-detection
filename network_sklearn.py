from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from data_loader import load

# never edit training data because I think pickle saves the changes to orginal file
training_data = load('training_data_v2.pkl')
training_data_edited = []
random.shuffle(training_data_edited)
count_not_gazing = 0
for image in training_data:
    if (image[1] == 1):
        training_data_edited.append(image)
    # we want the number of not gazing images to be the same as gazing (640)
    elif (image[1] == 0 and count_not_gazing < 640):
        training_data_edited.append(image)
        count_not_gazing += 1
random.shuffle(training_data_edited)
images=[]
expected=[]
for item in training_data_edited:
    images.append(item[0])
    expected.append(item[1])
images = np.array(images).reshape(len(images),len(images[0]))
expected = np.array(expected).reshape(len(expected),1)
print("Num pictures gazing:")
print(str(len(expected[expected==1])) + " / " + str(len(expected)))
print("Num pictures not gazing:")
print(str(len(expected[expected==0])) + " / " + str(len(expected)))
num_test = 300
# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma='auto')
# Learns to fit traning images to results (wether or not user is gazing)
classifier.fit(images[:-300], expected[:-300])
# Now predict the values of the testing images:
predicted = classifier.predict(images[-num_test:])

num_correct = 0
for i in range(len(predicted)):
    print(predicted[i])
    print(expected[i+len(expected) - num_test])
    num_correct += int(predicted[i] == expected[i+len(expected) - num_test])
print("Num correct: " + str(num_correct))

print(images[-1].reshape(32,64))
plt.imshow(images[-1].reshape(32,64), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
