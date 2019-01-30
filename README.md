# MNIST_Network

This repository contains a Python3 implementation of a neural network. The network is used to identify handwritten digits from images. The dataset is the famous MNIST dataset.

## Usage

You can first train a network using the file 'train.py'. Please note that the command takes a while to start training. It also outputs some progress info to the screen.

`python(3) train.py [TRAINING_ITERATIONS] [TRAINING_RATE] [TRAINING_BATCH_SIZE] [FILENAME]`

Example train command:

![Example](https://github.com/UrjalaCoder/MNIST_Network/blob/master/readme_pictures/example_train_command.PNG)

Please make sure that the filename ends in `.npy` for the correct format.

After training the dataset it appears in a folder called `nets` with the filename you gave it.
You can then use the trained network by using the 'test.py' python program.

`python(3) test.py [FLAG_TO_USE_TEST_DATA] [NETWORK_FILENAME] [TEST_IMAGE_PATH]`

Example test command:

![Example](https://github.com/UrjalaCoder/MNIST_Network/blob/master/readme_pictures/example_test_command.PNG)

You can tell the program to run a test using the MNIST test data by having the first argument be 'T/(t)' or 'True(/true)'.
Network filename is just the name of the file eg. `myNetwork.npy`. The program automatically searches the `nets` folder for the correct network.

If you specify a `TEST_IMAGE_PATH` the network tries to guess the number that is (hopefully) contained in the image.
The image should be a 28x28 8-bit grayscale PNG image of a number. The number should be in white and the background in black.
Also make sure that the image is centered in the picture with atleast 4 pixel margins on every side.
The repository contains one test image: `test_image.png`. If you want to try your own please include it in the **root** directory of the project.

Example test output:

![Example](https://github.com/UrjalaCoder/MNIST_Network/blob/master/readme_pictures/example_output_test.PNG)

Test image used:

![Example](https://github.com/UrjalaCoder/MNIST_Network/blob/master/test_image.png)


The first line is the percentage that the network got right from the testing dataset. It only appears when `[FLAG_TO_USE_TEST_DATA]` is set to `True`.
The second line is the network's guess what the image from `[TEST_IMAGE_PATH]` is. The higher the percentage the higher the confidence the network has.

## Repository files and folders

  * `network.py` This file contains the 'Network' class. This class contains the actual implementation of the neural network.
  * `test.py` This file is used to test the network. Usage information is in the **Usage** section.
  * `train.py` This is the file that is used to train the network. Usage information is in the **Usage** section.
  * `data_helper.py` This file should not be executed on its own. It's just used to contain some helpful functions that the other files utilize.
  * `test_image.png` Included test image. Image contains a picture of the digit (2).
  * `/data` This folder contains the training and testing data. It is the MNIST dataset.

## Dependencies

This python project requires these to work properly:
  * [python-mnist](https://github.com/sorki/python-mnist) Used to load the MNIST data. (Version 0.6)
  * [numpy](https://github.com/numpy/numpy) Used for matrix and vector calculations. (Version 1.15.4)
  * [matplotlib](https://github.com/matplotlib/matplotlib) Used to display images. (Version 3.0.2)

All of them can be installed using `pip install`.
