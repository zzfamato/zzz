import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, df_articles, tokenizer, df_labels):
        super(TextDataset, self).__init__()
        self.articles = []
        self.labels = df_labels

        # self.articles = [tokenizer.build_inputs_with_special_tokens(tokenizer.
        #                                                             convert_tokens_to_ids(tokenizer.
        #                                                                                   tokenize(row.values[0])))
        #                  for _, row in df_articles.iterrows()]
        # tokenizer.encode_plus(max_length=512)
        self.articles = [tokenizer.encode_plus(row.values[0], max_length=128, return_tensors='pt',
                                               pad_to_max_length=True, truncation=True)
                         for _, row in df_articles.iterrows()]
        # print(self.articles)

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, i):
        return self.articles[i], torch.tensor(self.labels.iloc[i], dtype=torch.int)


def main():
    from transformers import BertTokenizer
    from utils import DataReader, analyze
    X, Y = DataReader.read_training('./data/toy_train.csv')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_loader = DataLoader(TextDataset(X, tokenizer, Y), batch_size=32, drop_last=False, num_workers=0)
    s = next(iter(train_loader))
    print(s)
    # analyze(X, Y)


if __name__ == '__main__':
    main()
