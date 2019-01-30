import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpic
from network import Network
from data_helper import *
from mnist import MNIST
import sys, random

def prepare_testing_data(mndata):
    test_images, test_labels = mndata.load_testing()
    testing_raw_data = combine_data(test_images, test_labels)
    testing_data = preprocess_data(testing_raw_data)
    return testing_data

def get_vector_max(vector):
    max = 0
    for index in range(len(vector)):
        if vector[index][0] > vector[max][0]:
            max = index
    return (max, vector[max])

def test_network(net, test_data):
    correct = 0
    for sample in test_data:
        guess = net.get_guess(sample[0])[0]
        correct_output = get_vector_max(sample[1])[0]
        if guess == correct_output:
            correct = correct + 1
    print("{} of {} => {}%".format(correct, len(test_data), float(correct) / float(len(test_data)) * 100))

def visual_test(net, test_data, test_count):
    fig = plt.figure(figsize=(8, 8))
    random_indecis = [random.randrange(0, len(test_data)) for _ in range(test_count)]
    count = 1
    for index in random_indecis:
        guess = net.get_guess(test_data[index][0])[0]
        correct_output = get_vector_max(test_data[index][1])[0]
        print(guess, " : ", correct_output)
        fig.add_subplot(5, 5, count)
        count = count + 1
        plt.imshow(test_data[index][0].reshape(28, 28), cmap='gray', vmin=0, vmax=1.0)
    plt.show()

def real_image(net, path):
    img = mpic.imread(path)
    img_data = np.array([img]).reshape(28 * 28, 1)
    guess = net.feed_forward(img_data)[0][-1]
    print(format_result(guess))
    plt.imshow(img_data.reshape(28, 28), cmap="gray", vmin=0, vmax=1.0)
    plt.show()


# Function to show useful info about the network output vector.
def format_result(result_vector):
    result = ""
    for index in range(len(result_vector)):
        p = result_vector[index]
        result = result + "{} : {}%, ".format(index, round(p[0], 2) * 100)
    return result

def main():
    mndata = MNIST("./data")

    # Get testing_data to correct form.
    testing_data = prepare_testing_data(mndata)

    file_name = None
    test_file_path = None
    # Read network from file
    try:
        test_testing_data = (sys.argv[1].upper() == "True" or sys.argv[1].upper() == "T")
        file_name = sys.argv[2]
        if len(sys.argv) > 3:
            test_file_path = sys.argv[3]
    except IndexError:
        print("Invalid arguments\nUsage: python test.py [TEST_TEST_DATA] [NETWORK_FILENAME] [TEST_IMAGE_PATH]")
        return
    net = Network([28*28, 16, 10])
    arc = Network.load("/".join(['nets', file_name]))
    net.weights = arc[0]
    net.biases = arc[1]
    if test_testing_data:
        test_network(net, testing_data)
    if test_file_path != None:
        real_image(net, test_file_path)


if __name__ == "__main__":
    main()
