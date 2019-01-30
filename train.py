import numpy as np
import random, math
from data_helper import *
from mnist import MNIST
from matplotlib import pyplot as plt
from matplotlib import image as mpic
from network import Network

"""
Python script for training the network.
Networks are stored in the 'nets' folder.
python train.py [TRAINING_ITERATIONS] [TRAINING_RATE] [BATCH_SIZE] [FILENAME]
"""

# Prepares the training data
def prepare_training_data(mndata):
    # Load using mndata
    images, labels = mndata.load_training()

    # Combine images and labels to one dataset.
    training_raw_data = combine_data(images, labels)

    # From combined data, form the actual input and output vectors. Store them in an array.
    training_data = preprocess_data(training_raw_data)

    # Return the result
    return training_data

# Main function
#--------------
def main():
    import sys
    net = Network([28*28, 16, 10])

    max_counter = None
    tr = None
    batch_size = None
    file_name = False

    try:
        max_counter = int(sys.argv[1])
        tr = float(sys.argv[2])
        batch_size = int(sys.argv[3])
        file_name = sys.argv[4]
    except IndexError:
        print("Invalid arguments!\nUsage: python train.py [TRAINING_ITERATIONS] [TRAINING_RATE] [BATCH_SIZE] [FILE_NAME]")
        return

    mndata = MNIST("./data")
    training_data = prepare_training_data(mndata)
    print("Data loaded!. Starting training...")
    net.train(training_data, max_counter, tr, batch_size)
    Network.store("/".join(['nets', file_name]), (net.weights, net.biases))

if __name__ == "__main__":
    main()
