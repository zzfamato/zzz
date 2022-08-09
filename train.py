import torch
import random
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from data import TextDataset
from utils import DataReader
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification


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


def main():
    parser = ArgumentParser()

    parser.add_argument("--pretrained_model", type=str, default="bert-base-uncased")
    parser.add_argument("--data_path", type=str, default="./data/toy_train.csv")
    parser.add_argument("--test_path", type=str, default="./data/test.csv")
    parser.add_argument("--test_label_path", type=str, default="./data/labels.csv")
    parser.add_argument("--save_path", type=str, default="./out/bertfn")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1029)

    args = parser.parse_args()

    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize model
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model, model_max_length=args.max_seq_len)
    model = BertForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model = model.to(device)
    criterion = criterion.to(device)
    # load in dataset
    X, Y = DataReader.read_training(args.data_path)
    # split out a validation set for development
    X_train, X_vali, Y_train, Y_vali = train_test_split(X, Y, test_size=0.2, random_state=args.seed)
    train_loader = DataLoader(TextDataset(X_train, Y_train, tokenizer), batch_size=32, drop_last=False, num_workers=0)

    model.train()
    for epoch in range(args.num_epochs):
        all_loss = list()
        for i, batch in enumerate(train_loader):
            model.zero_grad()
            optimizer.zero_grad()
            x, y = batch
            output = model(x['input_ids'].squeeze(1).to(device),
                           attention_mask=x['attention_mask'].squeeze(1).to(device))
            loss = criterion(output.logits, y.squeeze(1).to(device))
            all_loss.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()

        print('Epoch#' + str(epoch + 1) + '\tLoss=' + str(np.mean(all_loss)))
    # save the model
    state = model.state_dict()
    torch.save(state, args.save_path)
    evaluate(model, DataLoader(TextDataset(X_vali, Y_vali, tokenizer), batch_size=1))


def evaluate(model, test_loader):
    with torch.no_grad():
        preds, golds = [], []
        for i, batch in enumerate(test_loader):
            x, y = batch
            pred = model(x['input_ids'].squeeze(1), attention_mask=x['attention_mask'].squeeze(1))
            golds.extend(y.numpy().flatten())
            preds.extend([np.argmax(seq) for seq in pred.logits.detach().numpy()])

    print(metrics.classification_report(golds, preds, digits=3, output_dict=False))


if __name__ == "__main__":
    main()
