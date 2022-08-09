import pandas as pd


def contains_label(text: str):
    """
    check if the given text contains label at the end, if yes it may be the end of a sample.
    :param text: a given line
    :return: a boolean value, if True is returned, the given text contains label information
    """
    text = text.strip()
    return text.endswith(",1") or text.endswith(",0")


class DataReader:
    def __init__(self):
        pass

    @staticmethod
    def read_training(train_src: str):
        """
        Read in training csv and corresponding label csv. Header is included.
        :param train_src: path to training file
        :return: an iterable Panda.DataFrame object
        """
        articles, labels = DataReader._transform_training(train_src)
        articles, labels = pd.DataFrame(articles, dtype=str), pd.DataFrame(labels, dtype=int)
        articles.info()
        labels.info()
        return articles, labels

    @staticmethod
    def read_test(test_src: str, label_src: str):
        """
        Read in test csv and ground true labels. Header is included.
        :param test_src: path to testing
        :param label_src: path to labels
        :return: an iterable Panda.DataFrame object
        """
        return DataReader._transform_test(test_src, label_src)

    @staticmethod
    def _transform_training(train_src):
        """
        pre-process and collate training samples from file.
        The ids are strictly increasing by 1, the end of a article is defined by the following rules:
        1, it contains a label at the end;
        2, the next line is a new sample or the end of the file.
        :param train_src: path to training csv file
        :return: a list of articles, a list of labels
        """
        articles, labels = [], []
        with open(train_src, 'r') as f:
            # skip the header
            next(f)
            line = f.readline()
            tid, text, label = None, None, None
            while line:
                # if a new article is found
                if tid is None:
                    if contains_label(line):
                        # this is a one-liner sample
                        _, rest = line.split(",", maxsplit=1)
                        text, _, label = rest.strip().rpartition(",")
                        articles.append(text)
                        labels.append(label)
                    else:
                        tid, text = line.split(",", maxsplit=1)
                        # print(tid, text)
                    line = f.readline()

                # a article is under processing
                else:
                    # if the current line could be the last line of a article
                    if contains_label(line):
                        # if a new tweet is found in the next line, we are sure this is the end of a article
                        next_line = f.readline()
                        if not next_line or next_line.startswith(str(int(tid)+1)+','):
                            more_text, _, label = line.rpartition(',')
                            text += more_text.strip()
                            line = next_line
                            articles.append(text)
                            labels.append(label)
                            tid = None
                            continue
                    # not the end of current article
                    text += line
                    line = f.readline()
        return articles, labels

    @staticmethod
    def _transform_test(test_src, label_src):
        """
        load test data from file
        :param test_src:
        :param label_src:
        :return: text dataframe and label dataframe
        """
        labels = pd.read_csv(label_src, header=0, encoding='utf-8', dtype=int)
        articles = pd.read_csv(test_src, header=0, encoding='utf-8', dtype=str)
        articles.info()
        labels.info()
        return articles, labels


def analyze(df_articles, df_labels):
    df = pd.concat([df_articles, df_labels], axis=1)
    df.columns = ['article', 'label']
    print("label distribution: \n", df.groupby("label").count())

