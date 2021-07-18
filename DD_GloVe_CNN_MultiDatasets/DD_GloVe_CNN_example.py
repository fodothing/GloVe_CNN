from numpy import asarray
from numpy import zeros
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
import time

from Text_Preprocess.TextPreprocess import *
##############################################################################
start = time.process_time()
##############################################################################
# 输入数据
docs = [['hello', 'baby', 'very', 'bad'],
        ['bad', 'day'],
        ['good', 'morning sir', 'thanks'],
        ['I', 'like', 'you', 'very', 'much'],
        ['I', 'can', 'do', 'more', 'things','for','you'],
        ['very','day','will','be','ok'],
        ['today','is','very','badly']
        ]
labels = [1, 1, 0, 0, 0, 0, 1]  # Negative polarity is class 0, and positive class 1.

labels = np.array(labels)
##############################################################################
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
print(f"vocab_size: {vocab_size}")
##############################################################################
# integer encode the documents

encoded_docs = t.texts_to_sequences(docs)
print(f"encoded_docs: {encoded_docs[:4]}")
##############################################################################
# pad documents to a max length of n words
# max_length = max([len(s.split()) for s in docs])
max_length = max([len(docs[i]) for i in range(len(docs))])
print(f"max_length: {max_length}")
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(f"padded_doc:\n {padded_docs[:4]}")
##############################################################################
# 划分train and test set
# note: random_state让每次划分的训练集和数据集相同
X_train, X_test, y_train, y_test = train_test_split(padded_docs, labels, test_size=0.3, random_state=2)  # , random_state=False
##############################################################################
path_glove = "D:\SoftPackage\pycharm\PyCharmProject\MachineLearning\Data\glove.6B.50d.txt"
# load the whole embedding into memory
embeddings_index = dict()
f = open(path_glove, mode='rt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s GloVe word vectors.' % len(embeddings_index))
##############################################################################
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 50))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
##############################################################################
# define model
model = Sequential()
e = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=max_length, trainable=False)

model.add(e)
model.add(Flatten())
# model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
##############################################################################
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
model.summary()
to_file = "D:\SoftPackage\pycharm\PyCharmProject\MachineLearning\Data\\Model_png\\example.png"
plot_model(model, to_file= to_file, show_shapes=True)
##############################################################################
# fit the model
model.fit(X_train, y_train, epochs=5, verbose=2)
##############################################################################
# 评价测试集
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('test set accuracy: %f' % (accuracy*100))

print(f"X_test: {X_test}")
print(f"y_test: {y_test}")

y_pred = model.predict_classes(X_test)  # 获取预测标签

print(f"y_pred:\n {y_pred}")


##############################################################################
end = time.process_time()
print(f"\nrunning time: {end-start}")
