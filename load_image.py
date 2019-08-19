from PIL import Image
from definitions import EMNIST_TRAIN_PATH, EMNIST_TEST_PATH
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc

class EmnistLoader():
    def __init__(self):
        self.image_width, self.image_height = 28, 28
        self.scaled_width, self.scaled_height = 32, 32
        self.train_data_size = 88800
        self.test_data_size = 14800

    def rotate(self, image):
        image = image.reshape([self.image_width, self.image_height])
        image = np.fliplr(image)
        image = np.rot90(image)
        return image.reshape([self.image_width * self.image_height])

    def scale(self, images):
        scaled_images = []
        for image in images:
            temp_image = Image.fromarray(np.uint8(image))
            temp_image = temp_image.resize((self.scaled_width, self.scaled_height), Image.BICUBIC)
            scaled_images.append(np.array(temp_image, dtype=np.float32)/255)

        return np.array(scaled_images)

    def load(self):
        # Import training data from csv
        train = pd.read_csv(EMNIST_TRAIN_PATH, header=None, delimiter=',')
        test = pd.read_csv(EMNIST_TEST_PATH, header=None, delimiter=',')

        # Split training data into categories
        train_data = train.iloc[:, 1:]
        train_labels = train.iloc[:, 0]
        test_data = test.iloc[:, 1:]
        test_labels = test.iloc[:, 0]

        # Turn dataframes into numpy array
        train_data = train_data.values
        self.train_labels = train_labels.values
        test_data = test_data.values
        self.test_labels = test_labels.values
        del train, test

        # Rotate images
        train_data = np.apply_along_axis(self.rotate, 1, train_data)
        test_data = np.apply_along_axis(self.rotate, 1, test_data)

        # Append images to 2d array with correct dimensions
        unscaled_train = []
        for i in range(self.train_data_size):
            unscaled_train.append(train_data[i].reshape([self.image_width, self.image_height]))

        unscaled_test = []
        for i in range(self.test_data_size):
            unscaled_test.append(test_data[i].reshape([self.image_width, self.image_height]))

        # Scale the images squish them between 0 and 1
        self.train_data = self.scale(unscaled_train)
        self.test_data = self.scale(unscaled_test)
