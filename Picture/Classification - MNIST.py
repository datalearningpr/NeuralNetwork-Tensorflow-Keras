

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# we are using the MNIST data from the tensorflow library
MNIST_data = input_data.read_data_sets('MNIST_data', one_hot=True)

# create a simple NN to train the classification model
with tf.name_scope("input"):
    x_input = tf.placeholder(tf.float32, [None, 28 * 28], "x_input")
    y_input = tf.placeholder(tf.float32, [None, 10], "y_input")

# the model is just one layer
with tf.name_scope("net"):
    pred = tf.layers.dense(inputs = x_input,
        units = 10,
        activation=None,
        name = "pred"
    )

# the accuracy is just for printing out
with tf.name_scope("accuracy"):
    accu, accu_op = tf.metrics.accuracy(tf.argmax(y_input, 1), tf.argmax(tf.nn.softmax(pred), 1))

# since we are using the loss of softmax_cross_entropy
# we do not need to add softmax activation function for the pred
with tf.name_scope("loss"):
    loss = tf.losses.softmax_cross_entropy(y_input, pred)

# use GradientDescentOptimizer
with tf.name_scope("train_op"):
    train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# the tf.metrics.accuracy requires the local_variables_initializer
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


for i in range(3000):
    x, y = MNIST_data.train.next_batch(32)
    sess.run(train_op, feed_dict={x_input: x, y_input: y})
    if i % 100 == 0:
        sess.run([accu, accu_op], feed_dict={x_input: MNIST_data.test.images, y_input: MNIST_data.test.labels})
        result = sess.run(accu, feed_dict={x_input: MNIST_data.test.images, y_input: MNIST_data.test.labels})
        print(result)


#################################################################################
# Keras version

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


model = Sequential()
model.add(Dense(10, activation='softmax', input_shape=(784,)))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(MNIST_data.train.images, MNIST_data.train.labels,
                    batch_size=32,
                    epochs=2,
                    verbose=1,
                    validation_data=(MNIST_data.test.images[-1000:], MNIST_data.test.labels[-1000:]))


