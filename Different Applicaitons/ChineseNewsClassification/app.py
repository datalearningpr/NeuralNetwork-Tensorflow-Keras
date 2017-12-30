
import os 
import pickle
import jieba
import numpy as np

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Input, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split


#########################################################################
#   for news classification you should find the tutorial in the keras offical examples
#   but the example is based on English
#   for chinese NLP, it is a bit different
#   this example should show the difference
#########################################################################


# the chinese news data can be downloaded from http://thuctc.thunlp.org/message


# since the data is very large, we only use the first 1000 news of each 14 groups
def process_txt():
    y = []
    x = [] 
    for folder in os.listdir("THUCNews"):
        sub_folder = os.path.join("THUCNews", folder)
        if os.path.isdir(sub_folder):
            for i, txt in enumerate(os.listdir(sub_folder)):
                if i > 999:
                    break
                file_path = os.path.join(sub_folder, txt)
                with open(file_path, "r", encoding = "utf-8") as f:
                    text = f.read()
                    x.append(text)
                    y.append(folder)
    print("processed {} files".format(len(x)))
    return x, y


x, y = process_txt()

# this is the one of biggest difference between English and Chinese NLP
# English the token is easy to tell by the space
# there is no space in Chinese, so we need libraries to conduct the tokenization
x_cut = list(map(lambda x: list(jieba.cut(x)), x))

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(np.hstack(x_cut))
sequences = list(map(tokenizer.texts_to_sequences, x_cut))


# we need to merge the tokens to be a valid sequence
def merge_token(x):
    result = []
    for i in x:
        if i:
            result.append(i[0])
    return result
final_sequences = list(map(merge_token, sequences))


# one hot coding for the 14 different news labels
labels = list(np.unique(y))
def vec_y(y, labels = labels):
    result = len(labels) * [0]
    index = labels.index(y)
    result[index] = 1
    return result


num_words = 10000       # unique number of words used to vectorize the text
sequence_len = 1000     # to make each sentence to same length
embedding_dims = 50     # word embedding size


x_data = pad_sequences(final_sequences, maxlen=sequence_len)
y_data = np.vstack(map(vec_y, y))

# due to the fact that only test_file contains the 3 labels
# we use 30% of train data as test data
# we still treat as if there are 3 lables, no differnce to the output
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                    test_size=0.3, random_state=42)


#############################################################################
model = Sequential()
model.add(Embedding(num_words,
                    embedding_dims,
                    input_length=sequence_len))
model.add(GlobalAveragePooling1D())

model.add(Dense(14))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=(x_test, y_test))

model.save('Fast_text_model.h5')
#############################################################################

# model = load_model('Fast_text_model.h5')


# now you can put some new test data(in 1.text, 2.txt) to see how the model perform
with open("1.txt", "r", encoding = "utf-8") as f:
    text1 = f.read()

with open("2.txt", "r", encoding = "utf-8") as f:
    text2 = f.read()

# now try some new user input
test_text = [text1, text2]

test_x_cut = list(map(lambda x: list(jieba.cut(x)), test_text))

test_sequences = list(map(tokenizer.texts_to_sequences, test_text))
final_test_sequences = list(map(merge_token, test_sequences))

test_x = pad_sequences(final_test_sequences, maxlen=sequence_len)

pred = model.predict(np.asmatrix(test_x))
index = np.argmax(pred, axis=1)
output = np.array(labels)
print(output[index])






