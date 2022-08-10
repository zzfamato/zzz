# Fake news!
***
Detect Fake news with BERT.
## Dataset overview

The training dataset has over 20,000 articles with fields for the article title, author, and text. The label field is the outcome variable where a 1 indicates the article is unreliable and a 0 indicates the article is reliable.
* train.csv contains 20800 articles. The ratio between positive and negative is balanced.
Some fields have missing values. The average article length is around 700.
* test.csv contains 5200 unlabeled articles.
* labels.csv contains ground true labels for test.
* toy_train_csv: a toy training data set

## Data preprocessing
An article is a long document consists of title, author, and text where text may contain multiple lines of sentences. To determine
the boundary of a article, rules belows are defined to detect article boundary:
* the last line of an article contains a label at the end;
* the next line is an new article with a "strictly increasing by 1" #id or the end of file.

Since the training dataset is balanced, and feature fields are text, the whole feature fields are concatenated to form a longer text.
The max article length is limited to 512 where the exceeding part is truncated and the missing part is filled with padding.

## System description
### files and folders
* /data: dataset
* /test: unit tests
* /out: store outputted models
* /main.py: fine-tune and test a model 
* /dataset.py: helper file to load dataset
* /utils.py: utils for loading data and dataset analysis

### to run
Start fine-tuning:
```
python3 main.py --train_path ./data/train.csv
```
Note: to know the parameters, see main.py.\
Test a existing model:
```
python3 main.py --test --model_file ./out/bertfn
```

## Evaluation
Since the training dataset is balanced, and type I error and type II error are equally important, accuracy and F1 score could be used for evaluation. However, when dealing with real world data, positive samples are rarer. F1 score is more preferable than accuracy.\
Belows are F1 scores of validation data and test data respectively.
| Attempt | #1 | #2 |
| :---: | :---: | :---: |
| Seconds | 301 | 283 |
## Why?

### why BERT?

### parameter tuning

Deliverables:
We would like you to build a pipeline for predicting the reliability of the articles in the dataset using any machine learning method of your choice. Please submit a folder containing your .py file(s). 
While producing a good result on the test data is ideal, the performance is not the main focus of this assignment. We are interested in your code quality, and your design process. We would like you to justify the design decisions you made such as:

The overall approach
The preprocessing steps
Evaluation procedure and metrics
Aspects that could be improved
etc.

Good luck!
