#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import numpy as np
import re
import os


def open_file(filename, mode='r'):
    """
    Commonly used file reader and writer, change this to switch between python2 and python3.
    :param filename: filename
    :param mode: 'r' and 'w' for read and write respectively
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def build_vocab(data, vocab_dir, vocab_size=8000):
    """
    Build vocabulary file from training data.
    """
    print('Building vocabulary...')

    all_data = []  # group all data
    for content in data:
        all_data.extend(content.split())

    counter = Counter(all_data)  # count and get the most common words
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))

    words = ['<PAD>'] + list(words)  # add a padding with id 0 to pad the sentence to same length
    open_file(vocab_dir, 'w').write('\n'.join(words) + '\n')


def read_vocab(vocab_file):
    """
    Read vocabulary from file.
    """
    words = open_file(vocab_file).read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def process_text(text, word_to_id, max_length, clean=True):
    """tokenizing and padding"""
    if clean:  # if the data needs to be cleaned
        text = clean_str(text)
    text = text.split()

    text = [word_to_id[x] for x in text if x in word_to_id]
    if len(text) < max_length:
        text = [0] * (max_length - len(text)) + text
    return text[:max_length]


class Corpus(object):
    """
    Preprocessing training data.
    """

    def __init__(self, pos_file, neg_file, vocab_file, dev_split=0.1, max_length=50, vocab_size=8000):
        # loading data
        pos_examples = [clean_str(s.strip()) for s in open_file(pos_file)]
        neg_examples = [clean_str(s.strip()) for s in open_file(neg_file)]
        x_data = pos_examples + neg_examples
        y_data = [0.] * len(pos_examples) + [1.] * len(neg_examples)  # 0 for pos and 1 for neg

        if not os.path.exists(vocab_file):
            build_vocab(x_data, vocab_file, vocab_size)

        self.words, self.word_to_id = read_vocab(vocab_file)

        for i in range(len(x_data)):  # tokenizing and padding
            x_data[i] = process_text(x_data[i], self.word_to_id, max_length, clean=False)

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # shuffle
        indices = np.random.permutation(np.arange(len(x_data)))
        x_data = x_data[indices]
        y_data = y_data[indices]

        # train/dev split
        num_train = int((1 - dev_split) * len(x_data))
        self.x_train = x_data[:num_train]
        self.y_train = y_data[:num_train]
        self.x_test = x_data[num_train:]
        self.y_test = y_data[num_train:]

    def __str__(self):
        return 'Training: {}, Testing: {}, Vocabulary: {}'.format(len(self.x_train), len(self.x_test), len(self.words))
