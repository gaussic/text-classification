
# coding: utf-8

# In[1]:

import preprocessing as ps
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D

from keras.preprocessing import sequence


# In[2]:

ngram_range = 1
max_features = 5000
batch_size = 64
embedding_dims = 128
epochs = 10
maxlen = 400
index = 3


# In[3]:

data_train, label_train = ps.read_data_maxlen('../train.txt', index, maxlen)
data_test, label_test = ps.read_data_maxlen('../test.txt', index, maxlen)
data_val, label_val = ps.read_data_maxlen('../val.txt', index, maxlen)


# In[4]:

words, word_to_id, id_to_word = ps.get_words(data_train + data_test + data_val, max_features)
class_set, cls_to_id, id_to_cls = ps.get_classes(label_val)

max_features = len(words)


# In[5]:

max_features


# In[6]:

X_train, y_train = ps.tokenize(data_train, label_train, word_to_id, cls_to_id, len(class_set))
X_test, y_test = ps.tokenize(data_test, label_test, word_to_id, cls_to_id, len(class_set))
X_val, y_val = ps.tokenize(data_val, label_val, word_to_id, cls_to_id, len(class_set))


# In[7]:

print(max(map(len, X_train)))
print(max(map(len, X_test)))
print(max(map(len, X_val)))


# In[8]:

if ngram_range > 1:
    token_indice, max_features = ps.build_ngram_tokens(X_train, max_features, ngram_range)
    X_train = ps.pad_ngram_data(X_train, token_indice, maxlen*2, ngram_range)
    X_test = ps.pad_ngram_data(X_test, token_indice, maxlen*2, ngram_range)
    X_val = ps.pad_ngram_data(X_val, token_indice, maxlen*2, ngram_range)
else:
    X_train = sequence.pad_sequences(X_train, maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen)
    X_val = sequence.pad_sequences(X_val, maxlen)


# In[9]:

y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)


# In[10]:

X_train, Y_train = ps.data_shuffle(X_train, y_train)
X_test, Y_test = ps.data_shuffle(X_test, y_test)
X_val, Y_val = ps.data_shuffle(X_val, y_val)


# In[11]:

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)
print(y_train.shape)
print(y_test.shape)
print(y_val.shape)


# In[12]:

# 构建模型
print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
# 先从一个高效的嵌入层开始，它将词汇表索引映射到 embedding_dim 维度的向量上
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))

# we add a GlobalAveragePooling1D, which will average the embeddings
# of all words in the document
# 添加一个 GlobalAveragePooling1D 层，它将平均整个序列的词嵌入
model.add(GlobalAveragePooling1D())

# We project onto a single unit output layer, and squash it with a sigmoid:
# 投影到一个单神经元输出层，然后使用 sigmoid 挤压。
model.add(Dense(len(class_set), activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()  # 概述


# In[ ]:

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_val, y_val))


# In[ ]:

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
print('Test score:', score)
print('Test accuracy:', acc)

