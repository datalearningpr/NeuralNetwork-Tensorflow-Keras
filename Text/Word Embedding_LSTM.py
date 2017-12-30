
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
import pickle


# the dataset is using the http://help.sentiment140.com/for-students/ dataset
# however, the training dataset seems to only have 2 labels but test dataset has 3 labels
# but this will not be a big problem for practice purpose

train_file = 'training.1600000.processed.noemoticon.csv'
test_file = 'testdata.manual.2009.06.14.csv'

train_data = pd.read_csv(train_file, encoding = "ISO-8859-1", header = None)
test_data = pd.read_csv(test_file, encoding = "ISO-8859-1", header = None)


num_words = 10000       # unique number of words used to vectorize the text
sequence_len = 80       # to make each sentence to same length
embedding_dims = 128    # word embedding size



# tokenize the input text
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train_data.iloc[:,5])
sequences = tokenizer.texts_to_sequences(train_data.iloc[:,5])

word_index = tokenizer.word_index

# make the label one hot coding
def vec_y(y):
    if y == 0:
        return np.array([1, 0, 0])
    elif y == 2:
        return np.array([0, 1, 0])
    elif y == 4:
        return np.array([0, 0, 1])

# make all the text input have the same length so that it could be fed into word embedding
x_data = pad_sequences(sequences, maxlen=sequence_len)
y_data = np.vstack(train_data.iloc[:,0].apply(vec_y))

# due to the fact that only test_file contains the 3 labels
# we use 30% of train data as test data
# we still treat as if there are 3 lables, no differnce to the output
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                    test_size=0.3, random_state=42)



#############################################################################
# LSTM model with word embedding
#############################################################################

model = Sequential()
model.add(Embedding(num_words, embedding_dims, input_length=sequence_len))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=32,
          epochs=1,
validation_data=(x_test, y_test))

model.save('LSTM_text_model.h5')
#############################################################################

# now try some new user input
test_text = ["i am very happy",
             "you are very sad"]

test_sequences = tokenizer.texts_to_sequences(test_text)
test_x = pad_sequences(test_sequences, maxlen=sequence_len)

pred = model.predict(np.asmatrix(test_x))
index = np.argmax(pred, axis=1)
output = np.array(["negative", "neutral", "positive"])
print(output[index])
