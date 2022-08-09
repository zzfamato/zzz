import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    customized dataset class for text, articles are tokenized with a Bert tokenizer
    """
    def __init__(self, df_articles, df_labels, tokenizer):
        """
        self.articles is a list of BatchEncoding objects
        :param df_articles: an article dataframe
        :param df_labels: a label dataframe
        :param tokenizer: Bert tokenizer
        """
        super(TextDataset, self).__init__()
        self.labels = df_labels
        self.articles = [tokenizer.encode_plus(row.values[0], return_tensors='pt',
                                               truncation=True, padding='max_length')
                         for _, row in df_articles.iterrows()]

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, i):
        return self.articles[i], torch.tensor(self.labels.iloc[i], dtype=torch.long)
