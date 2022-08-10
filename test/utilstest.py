import unittest
from utils import DataReader, trim_text


class UtilsTest(unittest.TestCase):
    def test_read_training(self):
        X, Y = DataReader.read_training('../data/train.csv')
        self.assertEqual(X.shape[0], Y.shape[0])
        self.assertEqual(X.shape[0], 20800)

    def test_read_test(self):
        X, Y = DataReader.read_test('../data/test.csv', '../data/labels.csv')
        self.assertEqual(X.shape[0], 5200)

    def test_trim_test(self):
        X, Y = DataReader.read_training('../data/train.csv')
        trimmed = trim_text(X, 128)
        for i, row in trimmed.iterrows():
            self.assertIsNotNone(row.values[0])
            self.assertLessEqual(len(row.values[0]), len(X.iloc[i].values[0]))
