

import tensorflow as tf
import numpy as np
import pandas as pd


# read data
data = pd.read_table("train.txt", delimiter=",")

# get the chinese character dict
word_dict = {}

def add_word(x, word_dict = word_dict):
    for i in x:
        if i in word_dict:
            word_dict[i] += 1
        else:
            word_dict[i] = 1
    

data.iloc[:, 1].apply(add_word)
# add space to the dict
vocabulary_list = [' '] + sorted(word_dict, key=word_dict.get, reverse=True)
# vocab will be the mapping from characters to numbers
vocab = dict([(x, y) for (y, x) in enumerate(vocabulary_list)])

# for the dataset we have, 4 should be more than enough
max_length = 4

def to_vector(x, max_length = 4, vocab = vocab):
    result = [0] * max_length

    for i, x in enumerate(x):
        result[i] = vocab.get(x, 0)

    return result

def vec_y(y):
    if y == 1:
        return np.array([1, 0])
    elif y == 0:
        return np.array([0, 1])

# make the data to be vectors 
x_data = np.vstack(data.iloc[:, 1].apply(to_vector))
y_data = np.vstack(data.iloc[:, 2].apply(vec_y))


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                    test_size=0.3, random_state=42)


# use the fast text model
# embedding + average pooling + output
num_words = len(vocab)      
sequence_len = 4     
embedding_dims = 50    

model = Sequential()
model.add(Embedding(num_words,
                    embedding_dims,
                    input_length=sequence_len))
model.add(GlobalAveragePooling1D())

model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=32,
          epochs=2,
		  validation_data=(x_test, y_test))

model.save('name2gender_model.h5')

##############################################################
# testing, training accuracy above 80%

test_case = [
	"德华",
	"艳芳",
	"富城",
	"菲",
	"克勤",
	"英"
]

gender = ["male", "female"]
test_data = np.array(list(map(to_vector, test_case)))
pred = model.predict(test_data)
result = list(np.array(gender * len(test_case))[np.argmax(pred, 1)])
for i in zip(test_case, result):
	print(i)

