
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
sequence_len = 100      # to make each sentence to same length
embedding_dims = 128    # word embedding size
filters = 64            # how many CONV
kernel_size = 5         # CONV size




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
# CNN LSTM model with word embedding
#############################################################################




class CNN_LSTM_comment():
    def __init__(self):

        self.embedding_size = embedding_dims
        self.filter_sizes = kernel_size
        self.num_filters = filters


        # create a simple NN to train the classification model
        with tf.name_scope("input"):
            self.x_input = tf.placeholder(tf.int32, [None, sequence_len], "x_input")
            self.y_input = tf.placeholder(tf.float32, [None, 3], "y_input")

        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([num_words, self.embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.x_input)
            # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)


        conv = tf.layers.conv1d(inputs = self.embedded_chars,
            filters = self.num_filters,
            kernel_size = self.filter_sizes,
            activation=tf.nn.relu,
            name = "conv")

        m = tf.layers.max_pooling1d(inputs = conv,
            pool_size = 4,
            strides = 4,
            name = "max_pool"
        )


        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(70)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, m, dtype = tf.float32)


        self.pred = tf.layers.dense(inputs=states[1], units=3, 
                                    name = "pred")

        

        # the accuracy is just for printing out
        with tf.name_scope("accuracy"):
            self.accu, self.accu_op = tf.metrics.accuracy(tf.argmax(self.y_input, 1), tf.argmax(tf.nn.softmax(self.pred), 1))

            predictions = tf.argmax(tf.nn.softmax(self.pred), 1)
            correct_predictions = tf.equal(predictions, tf.argmax(self.y_input, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))


        # since we are using the loss of softmax_cross_entropy
        # we do not need to add softmax activation function for the pred
        with tf.name_scope("loss"):
            self.loss = tf.losses.softmax_cross_entropy(self.y_input, self.pred)

        # use GradientDescentOptimizer
        with tf.name_scope("train_op"):
            self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)

        # the tf.metrics.accuracy requires the local_variables_initializer
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        self.saver = tf.train.Saver()


    def save_model(self):
        self.saver.save(self.sess, './CNN_LSTM_text_model.ckpt')

    def load_model(self):
        self.saver.restore(self.sess, './CNN_LSTM_text_model.ckpt')


model = CNN_LSTM_comment()

index = np.arange(len(x_train))
np.random.shuffle(index)
index_list = np.split(index, len(x_train) // 64)

for i, c in enumerate(index_list):
    
    x = x_train[c]
    y = y_train[c]

    model.sess.run(model.train_op, feed_dict={model.x_input: x, model.y_input: y})
    if i % 10 == 0:
        model.sess.run([model.accu_op, model.accu], feed_dict={model.x_input: x_test[:100], model.y_input: y_test[:100]})
        result = model.sess.run(model.accu, feed_dict={model.x_input: x_test[:100], model.y_input: y_test[:100]})
        result1 = model.sess.run(model.accuracy, feed_dict={model.x_input: x_test[:100], model.y_input: y_test[:100]})
        print(i, result, result1, model.sess.run(model.loss, feed_dict={model.x_input: x_test[:100], model.y_input: y_test[:100]}))
        model.save_model()


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
