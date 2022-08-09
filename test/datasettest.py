import unittest
from utils import DataReader
from dataset import TextDataset
from transformers import BertTokenizer


class TestData(unittest.TestCase):
    def test_read_training(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=128)
        X, Y = DataReader.read_training('../data/toy_train.csv')
        dataset = TextDataset(X, Y, tokenizer)
        for article in dataset.articles:
            self.assertEqual(article['input_ids'].shape[1], 128)
