#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This example demonstrates the use of Conv1D for CNN text classification.
Original paper could be found at: https://arxiv.org/abs/1408.5882

This is the baseline model: CNN-rand.

The implementation is based on PyTorch.

We didn't implement cross validation,
but simply run `python cnn_mxnet.py` for multiple times,
the average accuracy is close to 78%.

It takes about 2 minutes for training 20 epochs on a GTX 970 GPU.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn import metrics

from data_helper.mr_loader import Corpus, read_vocab, process_text

import os
import time
from datetime import timedelta

base_dir = 'data/mr'
pos_file = os.path.join(base_dir, 'rt-polarity.pos.txt')
neg_file = os.path.join(base_dir, 'rt-polarity.neg.txt')
vocab_file = os.path.join(base_dir, 'rt-polarity.vocab.txt')

save_path = 'checkpoints'  # model save path
if not os.path.exists(save_path):
    os.mkdir(save_path)
model_file = os.path.join(save_path, 'mr_cnn_pytorch.pt')

use_cuda = torch.cuda.is_available()


class TCNNConfig(object):
    """
    CNN Parameters
    """
    embedding_dim = 128  # embedding vector size
    seq_length = 50  # maximum length of sequence
    vocab_size = 8000  # most common words

    num_filters = 100  # number of the convolution filters (feature maps)
    kernel_sizes = [3, 4, 5]   # three kind of kernels (windows)

    hidden_dim = 128  # hidden size of fully connected layer

    dropout_prob = 0.5  # how much probability to be dropped
    learning_rate = 1e-3  # learning rate
    batch_size = 50  # batch size for training
    num_epochs = 2  # total number of epochs

    num_classes = 2  # number of classes

    dev_split = 0.1  # percentage of dev data


class TextCNN(nn.Module):
    """
    CNN text classification model, based on the paper.
    """

    def __init__(self, config):
        super(TextCNN, self).__init__()

        V = config.vocab_size
        E = config.embedding_dim
        Nf = config.num_filters
        Ks = config.kernel_sizes
        C = config.num_classes
        Dr = config.dropout_prob

        self.embedding = nn.Embedding(V, E)  # embedding layer

        # three different convolutional layers
        self.convs = nn.ModuleList([nn.Conv1d(E, Nf, k) for k in Ks])
        self.dropout = nn.Dropout(Dr)  # a dropout layer
        self.fc1 = nn.Linear(3 * Nf, C)  # a dense layer for classification

    @staticmethod
    def conv_and_max_pool(x, conv):
        """Convolution and global max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])

    def forward(self, inputs):
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        embedded = self.embedding(inputs).permute(0, 2, 1)
        x = [self.conv_and_max_pool(embedded, k) for k in self.convs]  # convolution and global max pooling
        x = self.fc1(self.dropout(torch.cat(x, 1)))  # concatenation and dropout

        return x


def get_time_dif(start_time):
    """
    Return the time used since start_time.
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(data, model, loss):
    """
    Evaluation, return accuracy and loss
    """
    model.eval()  # set mode to evaluation to disable dropout
    data_loader = DataLoader(data, batch_size=50)

    data_len = len(data)
    total_loss = 0.0
    y_true, y_pred = [], []

    for data, label in data_loader:
        data, label = Variable(data, volatile=True), Variable(label, volatile=True)
        if use_cuda:
            data, label = data.cuda(), label.cuda()

        output = model(data)
        losses = loss(output, label)

        total_loss += losses.data[0]
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(label.data)

    acc = (np.array(y_true) == np.array(y_pred)).sum()
    return acc / data_len, total_loss / data_len


def train():
    """
    Train and evaluate the model with training and validation data.
    """
    print('Loading data...')
    start_time = time.time()
    config = TCNNConfig()
    corpus = Corpus(pos_file, neg_file, vocab_file, config.dev_split, config.seq_length, config.vocab_size)
    print(corpus)
    config.vocab_size = len(corpus.words)

    train_data = TensorDataset(torch.LongTensor(corpus.x_train), torch.LongTensor(corpus.y_train))
    test_data = TensorDataset(torch.LongTensor(corpus.x_test), torch.LongTensor(corpus.y_test))

    print('Configuring CNN model...')
    model = TextCNN(config)
    print(model)

    if use_cuda:
        model.cuda()

    # optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # set the mode to train
    print("Training and evaluating...")

    best_acc = 0.0
    for epoch in range(config.num_epochs):
        # load the training data in batch
        model.train()
        train_loader = DataLoader(train_data, batch_size=config.batch_size)
        for x_batch, y_batch in train_loader:
            inputs, targets = Variable(x_batch), Variable(y_batch)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)  # forward computation
            loss = criterion(outputs, targets)

            # backward propagation and update parameters
            loss.backward()
            optimizer.step()

        # evaluate on both training and test dataset
        train_acc, train_loss = evaluate(train_data, model, criterion)
        test_acc, test_loss = evaluate(test_data, model, criterion)

        if test_acc > best_acc:
            # store the best result
            best_acc = test_acc
            improved_str = '*'
            torch.save(model.state_dict(), model_file)
        else:
            improved_str = ''

        time_dif = get_time_dif(start_time)
        msg = "Epoch {0:3}, Train_loss: {1:>7.2}, Train_acc {2:>6.2%}, " \
              + "Test_loss: {3:>6.2}, Test_acc {4:>6.2%}, Time: {5} {6}"
        print(msg.format(epoch + 1, train_loss, train_acc, test_loss, test_acc, time_dif, improved_str))

    test(model, test_data)


def test(model, test_data):
    """
    Test the model on test dataset.
    """
    print("Testing...")
    start_time = time.time()
    test_loader = DataLoader(test_data, batch_size=50)

    # restore the best parameters
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    y_true, y_pred = [], []
    for data, label in test_loader:
        data, label = Variable(data, volatile=True), Variable(label, volatile=True)
        if use_cuda:
            data, label = data.cuda(), label.cuda()

        output = model(data)
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(label.data)

    test_acc = metrics.accuracy_score(y_true, y_pred)
    test_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    print("Test accuracy: {0:>7.2%}, F1-Score: {1:>7.2%}".format(test_acc, test_f1))

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_true, y_pred, target_names=['POS', 'NEG']))

    print('Confusion Matrix...')
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)

    print("Time usage:", get_time_dif(start_time))


def predict(text):
    # load config and vocabulary
    config = TCNNConfig()
    _, word_to_id = read_vocab(vocab_file)
    labels = ['POS', 'NEG']

    # load model
    model = TextCNN(config)
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    # process text
    text = process_text(text, word_to_id, config.seq_length)
    text = Variable(torch.LongTensor([text]), volatile=True)

    if use_cuda:
        model.cuda()
        text = text.cuda()

    # predict
    model.eval()  # very important
    output = model(text)
    pred = torch.max(output, dim=1)[1]

    return labels[pred.data[0]]


if __name__ == '__main__':
    train()
    predict('this film is awesome')
    predict('this film is so bad')
