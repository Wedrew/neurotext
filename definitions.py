import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EMNIST_TRAIN_PATH = os.path.join(ROOT_DIR, 'train-data', 'emnist', 'emnist-letters-train.csv')
EMNIST_TEST_PATH = os.path.join(ROOT_DIR, 'test-data', 'emnist', 'emnist-letters-test.csv')
SAVED_NETWORK_PATH = os.path.join(ROOT_DIR, 'saved_networks')
