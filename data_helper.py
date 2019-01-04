import numpy as np

"""
Helper functions for the train.py and test.py files.
"""

# Function used to normalize the data.
def normalize(image_data):
    return list(map(lambda x: x / 255.0, image_data))

# Function that forms the input/output pairs which are used to train the network.
def preprocess_data(data):
    correct_data = []
    for sample in data:
        output = np.zeros((10, 1))
        output[sample[1]] = 1
        input = np.array([sample[0]]).transpose()
        correct_data.append([input, output])
    return correct_data

# Combines the 'images' and 'labels' lists. Also normalises the image data.
def combine_data(images, labels):
    new_data = []
    for image, label in zip(images, labels):
        new_data.append([normalize(image), label])
    return np.asarray(new_data)
