#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This example demonstrates the use of Conv1D for CNN text classification.
Original paper could be found at: https://arxiv.org/abs/1408.5882

This is the baseline model: CNN-rand.

The implementation is based on MXNET/Gluon API.

We didn't implement cross validation,
but simply run `python cnn_mxnet.py` for multiple times,
the average accuracy is close to 76%.

It tooks about 2 minutes for training 20 epochs on a GTX 970 GPU.
"""

import mxnet as mx
from mxnet import gluon, autograd, metric
from mxnet import ndarray as nd
from mxnet.gluon import nn
from mxnet.gluon.data import Dataset, DataLoader
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
model_file = os.path.join(save_path, 'mr_cnn.params')


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


class Config(object):
    """
    CNN parameters
    """
    embedding_dim = 128  # embedding vector size
    seq_length = 50  # maximum length of sequence
    vocab_size = 8000  # most common words

    num_filters = 100  # number of the convolution filters (feature maps)
    kernel_sizes = [3, 4, 5]   # three kinds of kernels (windows)

    dropout_prob = 0.5  # dropout rate
    learning_rate = 1e-3  # learning rate
    batch_size = 50  # batch size for training
    num_epochs = 3  # total number of epochs

    num_classes = 2  # number of classes

    dev_split = 0.1  # percentage of dev data


class TCNNConfig(object):
    """
    CNN parameters
    """
    embedding_dim = 128  # embedding vector size
    seq_length = 50  # maximum length of sequence
    vocab_size = 8000  # most common words

    num_filters = 100  # number of the convolution filters (feature maps)
    kernel_sizes = [3, 4, 5]   # three kinds of kernels (windows)

    dropout_prob = 0.5  # dropout rate
    learning_rate = 1e-3  # learning rate
    batch_size = 50  # batch size for training
    num_epochs = 20  # total number of epochs

    num_classes = 2  # number of classes

    dev_split = 0.1  # percentage of dev data


class Conv_Max_Pooling(nn.Block):
    """
    Integration of Conv1D and GlobalMaxPool1D layers
    """
    def __init__(self, channels, kernel_size, **kwargs):
        super(Conv_Max_Pooling, self).__init__(**kwargs)

        with self.name_scope():
            self.conv = nn.Conv1D(channels, kernel_size)
            self.pooling = nn.GlobalMaxPool1D()

    def forward(self, x):
        output = self.pooling(self.conv(x))
        return nd.relu(output).flatten()


class TextCNN(nn.Block):
    """
    CNN text classification model, based on the paper.
    """
    def __init__(self, config, **kwargs):
        super(TextCNN, self).__init__(**kwargs)

        V = config.vocab_size
        E = config.embedding_dim
        Nf = config.num_filters
        Ks = config.kernel_sizes
        C = config.num_classes
        Dr = config.dropout_prob

        with self.name_scope():
            self.embedding = nn.Embedding(V, E)  # embedding layer

            # three different convolutional layers
            self.conv1 = Conv_Max_Pooling(Nf, Ks[0])
            self.conv2 = Conv_Max_Pooling(Nf, Ks[1])
            self.conv3 = Conv_Max_Pooling(Nf, Ks[2])
            self.dropout = nn.Dropout(Dr)  # a dropout layer
            self.fc1 = nn.Dense(C)         # a dense layer for classification

    def forward(self, x):
        x = self.embedding(x).transpose((0, 2, 1))   # Conv1D takes in NCW as input
        o1, o2, o3 = self.conv1(x), self.conv2(x), self.conv3(x)
        outputs = self.fc1(self.dropout(nd.concat(o1, o2, o3)))

        return outputs


def get_time_dif(start_time):
    """
    Return the time used since start_time.
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


class MRDataset(Dataset):
    """
    An implementation of the Abstracted gluon.data.Dataset, used for loading data in batch
    """
    def __init__(self, x, y):
        super(MRDataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index].astype(np.float32), self.y[index].astype(np.float32)

    def __len__(self):
        return len(self.x)


def evaluate(data_iterator, data_len, net, loss, ctx):
    """
    Evaluation, return accuracy and loss
    """
    total_loss = 0.0, 0
    acc = metric.Accuracy()

    for data, label in data_iterator:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)

        with autograd.record(train_mode=False): # set the training_mode to False
            output = net(data)
            losses = loss(output, label)

        total_loss += nd.sum(losses).asscalar()
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    print('data_len', data_len)
    print('acc', acc.get()[1])
    print('total_loss', total_loss / data_len)
    return acc.get()[1], total_loss / data_len


def train():
    """
    Train and evaluate the model with training and test data.
    """
    print("Loading data...")
    start_time = time.time()
    config = Config()
    corpus = Corpus(pos_file, neg_file, vocab_file, config.dev_split, config.seq_length, config.vocab_size)
    print(corpus)
    config.vocab_size = len(corpus.words)

    print("Configuring CNN model...")
    ctx = try_gpu()
    model = TextCNN(config)
    model.collect_params().initialize(ctx=ctx)
    print("Initializing weights on", ctx)
    print(model)

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': config.learning_rate})

    batch_size = config.batch_size
    train_loader = DataLoader(MRDataset(corpus.x_train, corpus.y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MRDataset(corpus.x_test, corpus.y_test), batch_size=batch_size, shuffle=False)

    print("Training and evaluating...")
    best_acc = 0.0
    for epoch in range(config.num_epochs):
        for data, label in train_loader:
            data, label = data.as_in_context(ctx), label.as_in_context(ctx)

            with autograd.record(train_mode=True):   # set the model in training mode
                output = model(data)
                losses = loss(output, label)

            # backward propagation and update parameters
            losses.backward()
            trainer.step(len(data))

        # evaluate on both training and test dataset
        train_acc, train_loss = evaluate(train_loader, len(corpus.x_train), model, loss, ctx)
        test_acc, test_loss = evaluate(test_loader, len(corpus.x_test), model, loss, ctx)

        if test_acc > best_acc:
            # store the best result
            best_acc = test_acc
            improved_str = '*'
            model.save_params(model_file)
        else:
            improved_str = ''

        time_dif = get_time_dif(start_time)
        msg = "Epoch {0:3}, Train_loss: {1:>7.2}, Train_acc {2:>6.2%}, " \
              + "Test_loss: {3:>6.2}, Test_acc {4:>6.2%}, Time: {5} {6}"
        print(type(train_loss))
        print(train_loss)
        print(type(train_acc))
        print(type(test_loss))
        print(test_loss)
        print(type(test_acc))

        print(msg.format(epoch + 1, train_loss, train_acc, test_loss, test_acc, time_dif, improved_str))

    test(model, test_loader, ctx)


def test(model, test_loader, ctx):
    """
    Test the model on test dataset.
    """
    print("Testing...")
    start_time = time.time()
    model.load_params(model_file, ctx=ctx)

    y_pred, y_true = [], []
    for data, label in test_loader:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        with autograd.record(train_mode=False): # set the training_mode to False
            output = model(data)
        pred = nd.argmax(output, axis=1).asnumpy().tolist()
        y_pred.append(pred)
        y_true.append(label.asnumpy().tolist())

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_true, y_pred, target_names=['POS', 'NEG']))

    print('Confusion Matrix...')
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)

    print("Time usage:", get_time_dif(start_time))


def predict(text):
    # load config and vocabulary
    config = Config()
    _, word_to_id = read_vocab(vocab_file)
    labels = ['POS', 'NEG']

    # load model
    ctx = try_gpu()
    model = TextCNN(config)
    model.load_params(model_file, ctx=ctx)

    # process text
    text = process_text(text, word_to_id, config.seq_length)
    text = nd.array([text]).as_in_context(ctx)

    output = model(text)
    pred = nd.argmax(output, axis=1).asscalar()

    return labels[int(pred)]


if __name__ == '__main__':
    train()
    predict('this film is awesome')
    predict('this film is so bad')