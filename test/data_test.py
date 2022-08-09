import unittest
from utils import DataReader


class TestData(unittest.TestCase):
    def test_read_training(self):
        X, Y = DataReader.read_training('./data/toy_train.csv')
        self.assertEqual(X.shape[0], Y.shape[0])
