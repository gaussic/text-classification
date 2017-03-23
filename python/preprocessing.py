from collections import Counter
import numpy as np
from keras.preprocessing import sequence

def hello():
    print('hello')

def read_data(filename, index=1):
    '''
    index = 1 : 题目
    index = 2 : 网址
    index = 3 : 内容
    '''
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = [line.split('\t')[0] for line in lines]
        contents = [line.split('\t')[index] for line in lines]
        return contents, labels
    
def get_words(data, max_features):
    text = ''
    for line in data:
        text += line
    text = text.replace('\u3000', '').replace('\t', '').replace('\n', '')
    
    counter = Counter(text[:]).most_common(max_features)
    words, _ = zip(*counter)
    
    word_to_id = dict((c, i) for i, c in enumerate(words))
    id_to_word = dict((i, c) for i, c in enumerate(words))
    return words, word_to_id, id_to_word

def get_classes(data):
    class_set = set(data)
    cls_to_id = dict((c, i) for i, c in enumerate(class_set))
    id_to_cls = dict((i, c) for i, c in enumerate(class_set))
    return class_set, cls_to_id, id_to_cls

def tokenize(data, label, word_to_id, cls_to_id, num_class):
    X = []
    y = []
    for i in range(len(data)):
        X.append([word_to_id[x] for x in data[i] if x in word_to_id])
        y_v = [0] * num_class
        y_v[cls_to_id[label[i]]] = 1
        y.append(y_v)
    return X, y

def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    从一个整数列表中提取  n-gram 集合。
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))

def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    增广输入列表中的每个序列，添加 n-gram 值
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

def build_ngram_tokens(data, max_features, ngram_range=2):
    print('Adding {}-gram features'.format(ngram_range))
    ngram_set = set()
    for input_list in data:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)
            
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}
    
    max_features = np.max(list(indice_token.keys())) + 1
    
    return token_indice, max_features
    

def pad_ngram_data(data, token_indice, maxlen, ngram_range=2):
    data = add_ngram(data, token_indice, ngram_range)
    return sequence.pad_sequences(data, maxlen=maxlen)
    


        
        
        
        
        
        
        
        
        