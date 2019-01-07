# MNIST_Network

This repository contains a Python3 implementation of a neural network. The network is used to identify handwritten digits from images.

## Usage

You can first train a network using the file 'train.py'.

`python(3) train.py [TRAINING_ITERATIONS] [TRAINING_RATE] [TRAINING_BATCH_SIZE] [FILENAME]`

Please make sure that the filename ends in `.npy` for the correct format.

After training the dataset it appears in a folder called `nets` with the filename you gave it.
You can then use the trained network by using the 'test.py' python program.

`python(3) test.py [FLAG_TO_USE_TEST_DATA] [NETWORK_FILENAME] [TEST_IMAGE_PATH]`

You can tell the program to run a test using the MNIST test data by having the first argument be 'T/(t)' or 'True(/true)'.
Network filename is just the name of the file eg. `myNetwork.npy`. The program automatically searches the `nets` folder for the correct network.

If you specify a `TEST_IMAGE_PATH` the network tries to guess the number that is (hopefully) contained in the image.
The image should be a 28x28 8-bit grayscale PNG image of a number. The number should be in white and the background in black.
Also make sure that the image is centered in the picture with atleast 4 pixel margins on every side.
The repository contains one test image: `test_image.png`. If you want to try your own please include it in the **root** directory of the project.
