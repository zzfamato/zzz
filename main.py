import torch
from tqdm import tqdm
import random
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from dataset import TextDataset
from utils import DataReader
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification


def main():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="bert-base-uncased")
    parser.add_argument("--data_path", type=str, default="./data/train.csv")
    parser.add_argument("--test_path", type=str, default="./data/test.csv")
    parser.add_argument("--test_label_path", type=str, default="./data/labels.csv")
    parser.add_argument("--save_path", type=str, default="./out/bert4fn")
    parser.add_argument("--model_file", type=str, default="./out/bert4fn")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1029)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    # load a existing model and test
    if args.test:
        test(args)
        return
    # init
    tokenizer, model, criterion, optimizer = initialize(args)
    # load dataset
    print("loading data")
    train_loader, validation_loader = load_data(args, tokenizer)
    # start fine tuning
    print("start training")
    train(args, model, criterion, optimizer, train_loader)
    # save the model
    torch.save(model.state_dict(), args.save_path)
    # evaluate on validation set
    evaluate(model, validation_loader)


def initialize(args):
    """
    init components
    """
    setup_seed(args.seed)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model, model_max_length=args.max_seq_len)
    model = BertForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model = model.to(args.device)
    criterion = criterion.to(args.device)
    return tokenizer, model, criterion, optimizer


def setup_seed(seed):
    """
    initialize random seeds
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(args, tokenizer):
    """
    load training data or test data
    :return a test data loader for test or a tuple of (a train data loader, validation data loader)
    """
    if args.test:
        X, Y = DataReader.read_test(args.test_path, args.test_label_path)
        return DataLoader(TextDataset(X, Y, tokenizer))

    X, Y = DataReader.read_training(args.data_path)
    X_train, X_vali, Y_train, Y_vali = train_test_split(X, Y, test_size=0.2, random_state=args.seed)
    train_loader = DataLoader(TextDataset(X_train, Y_train, tokenizer), batch_size=args.batch_size,
                              drop_last=False, num_workers=0)
    validation_loader = DataLoader(TextDataset(X_vali, Y_vali, tokenizer), batch_size=1)
    return train_loader, validation_loader


def train(args, model, criterion, optimizer, train_loader):
    model.train()
    for epoch in range(args.num_epochs):
        all_loss = list()
        for batch in tqdm(train_loader):
            model.zero_grad()
            optimizer.zero_grad()
            x, y = batch
            output = model(x['input_ids'].squeeze(1).to(args.device),
                           attention_mask=x['attention_mask'].squeeze(1).to(args.device))
            loss = criterion(output.logits, y.squeeze(1).to(args.device))
            all_loss.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()

        print('Epoch#' + str(epoch + 1) + '\tLoss=' + str(np.mean(all_loss)))


def evaluate(model, test_loader):
    with torch.no_grad():
        preds, golds = [], []
        for _, batch in enumerate(test_loader):
            x, y = batch
            pred = model(x['input_ids'].squeeze(1), attention_mask=x['attention_mask'].squeeze(1))
            golds.extend(y.numpy().flatten())
            preds.extend([np.argmax(seq) for seq in pred.logits.detach().numpy()])

    print(metrics.classification_report(golds, preds, digits=3, output_dict=False))


def test(args):
    """
    load a trained model and test
    """
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model, model_max_length=args.max_seq_len)
    test_loader = load_data(args, tokenizer)
    model = BertForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=2)
    model.load_state_dict(args.model_file)
    evaluate(model, test_loader)


if __name__ == "__main__":
    main()
