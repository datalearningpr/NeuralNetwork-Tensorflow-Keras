

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# we screen row by row so that
input_size = 28     # input size is 28 columns
time_steps = 28     # we have 28 rows

# we are using the MNIST data from the tensorflow library
MNIST_data = input_data.read_data_sets('MNIST_data', one_hot=True)

# create RNN to train the classification model
with tf.name_scope("input"):
    x_input = tf.placeholder(tf.float32, [None, time_steps * input_size], "x_input")
    y_input = tf.placeholder(tf.float32, [None, 10], "y_input")
    x_reshape = tf.reshape(x_input, [-1, 28], "x_reshape")
    

# the model is CNN
with tf.name_scope("net"):

    l1 = tf.layers.dense(inputs = x_reshape,
        units = 64,
        activation=None,
        name = "l1"
    )
    
    l1_reshape = tf.reshape(l1, [-1, 28, 64], "l1_reshape")

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(64)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, l1_reshape, dtype = tf.float32)

    pred = tf.layers.dense(inputs=states[1], units=10, name = "pred")



# the accuracy is just for printing out
with tf.name_scope("accuracy"):
    accu, accu_op = tf.metrics.accuracy(tf.argmax(y_input, 1), tf.argmax(tf.nn.softmax(pred), 1))

# since we are using the loss of softmax_cross_entropy
# we do not need to add softmax activation function for the pred
with tf.name_scope("loss"):
    loss = tf.losses.softmax_cross_entropy(y_input, pred)

# use GradientDescentOptimizer
with tf.name_scope("train_op"):
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

# the tf.metrics.accuracy requires the local_variables_initializer
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


for i in range(1000):
    print(i)
    x, y = MNIST_data.train.next_batch(100)
    sess.run(train_op, feed_dict={x_input: x, y_input: y})

result = sess.run(accu_op, feed_dict={x_input: MNIST_data.test.images[:1000], y_input: MNIST_data.test.labels[:1000]})
print(result)




#################################################################################
# Keras version

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Reshape
from keras.optimizers import Adam


model = Sequential()
model.add(Reshape((-1, 28), input_shape=(784,)))
model.add(Dense(64))
model.add(LSTM(64))
model.add(Dense(10))
model.add(Activation('softmax')) 
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(MNIST_data.train.images, MNIST_data.train.labels,
                    batch_size=32,
                    epochs=2,
                    verbose=1,
                    validation_data=(MNIST_data.test.images[-1000:], MNIST_data.test.labels[-1000:]))


