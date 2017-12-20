#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
MXNET version of CNN for sentence classification.
"""

from mxnet.gluon import nn
from mxnet import ndarray as nd


class Config(object):
    """
    CNN parameters
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
    num_epochs = 20  # total number of epochs

    print_per_batch = 100  # print out the intermediate status every n batches

    num_classes = 2  # number of classes

    dev_split = 0.1  # percentage of dev data


class CNN(nn.Block):
    def __init__(self, config, **kwargs):
        super(CNN, self).__init__(**kwargs)

        V = config.vocab_size
        E = config.embedding_dim
        Nf = config.num_filters
        Ks = config.kernel_sizes
        C = config.num_classes
        dropout = config.dropout_prob

        with self.name_scope():
            self.embedding = nn.Embedding(V, E)
            
            self.fc1 = nn.Dense(3 * Nf, C)

    @staticmethod
    def conv_max_pool(x, conv):
        return nd.relu(conv(x))

    def forward(self, inputs):
        embedded = nd.transpose(self.embedding(inputs), (0, 2, 1))
