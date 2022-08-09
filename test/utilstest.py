import unittest
from utils import DataReader


class UtilsTest(unittest.TestCase):
    def test_read_training(self):
        X, Y = DataReader.read_training('../data/train.csv')
        self.assertEqual(X.shape[0], Y.shape[0])
        self.assertEqual(X.shape[0], 20800)

    def test_read_test(self):
        X, Y = DataReader.read_test('../data/test.csv', '../data/labels.csv')
        self.assertEqual(X.shape[0], 5200)
