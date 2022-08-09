import unittest
from utils import DataReader
from data import TextDataset
from transformers import BertTokenizer


class TestData(unittest.TestCase):
    def test_read_training(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        X, Y = DataReader.read_training('../data/toy_train.csv')
        dataset = TextDataset(X, tokenizer, Y)
        for article in dataset.articles:
            self.assertEqual(article['input_ids'].shape[1], 128)
