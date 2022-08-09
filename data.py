import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, df_articles, tokenizer, df_labels):
        super(TextDataset, self).__init__()
        self.articles = []
        self.labels = df_labels

        for _, row in df_articles.iterrows():
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(row.values[0]))
            self.articles.append(tokenizer.build_inputs_with_special_tokens(tokenized_text))

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, i):
        return torch.tensor(self.articles[i], dtype=torch.long), torch.tensor(self.labels.iloc[i], dtype=torch.int)


def main():
    from transformers import BertTokenizer
    from utils import DataReader, analyze
    X, Y = DataReader.read_training('./data/toy_train.csv')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=256)
    # train_loader = DataLoader(TextDataset(X, tokenizer, Y), batch_size=32, drop_last=False, num_workers=0)
    analyze(X, Y)


if __name__ == '__main__':
    main()
