

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


nb_conv = 5
nb_pool = 2

# we are using the MNIST data from the tensorflow library
MNIST_data = input_data.read_data_sets('MNIST_data', one_hot=True)

# create CNN to train the classification model
with tf.name_scope("input"):
    x_input = tf.placeholder(tf.float32, [None, 28 * 28], "x_input") / 255.0
    y_input = tf.placeholder(tf.float32, [None, 10], "y_input")
    x_reshape = tf.reshape(x_input, [-1, 28, 28, 1], "x_reshape")

# the model is CNN
with tf.name_scope("net"):

    # 2 layers of conv
    c1 = tf.layers.conv2d(inputs = x_reshape,
        filters = 32,
        kernel_size=(nb_conv, nb_conv),
        padding="same",
        activation=tf.nn.relu,
        name = "c1"
    )

    c2 = tf.layers.conv2d(inputs = c1,
        filters = 32,
        kernel_size=(nb_conv, nb_conv),
        padding="same",
        activation=tf.nn.relu,
        name = "c2"
    )

    # 1 layer of max pooling
    m1 = tf.layers.max_pooling2d(inputs = c2,
        pool_size = (nb_pool, nb_pool),
        strides = 2,
        name = "m1"
    )

    # flatten the pooling result
    f = tf.layers.flatten(inputs = c1,
        name = "f"
    )

    # 2 layers of fully connected NN
    l1 = tf.layers.dense(inputs = f,
        units = 256,
        activation=tf.nn.relu,
        name = "l1"
    )

    d1 = tf.nn.dropout(x = l1,
        keep_prob = 0.8,
        name = "d1"
    )

    l2 = tf.layers.dense(inputs = d1,
        units = 128,
        activation=tf.nn.relu,
        name = "l2"
    )

    d2 = tf.nn.dropout(x = l2,
        keep_prob = 0.8,
        name = "d2"
    )

    pred = tf.layers.dense(inputs=d2, units=10, name = "pred")



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
    x, y = MNIST_data.train.next_batch(64)
    sess.run(train_op, feed_dict={x_input: x, y_input: y})

result = sess.run(accu_op, feed_dict={x_input: MNIST_data.test.images[:1000], y_input: MNIST_data.test.labels[:1000]})
print(result)







#################################################################################
# Keras version

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam



model = Sequential()

model.add(Conv2D(32, nb_conv, nb_conv, input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax')) 


model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(np.reshape(MNIST_data.train.images, [-1, 28, 28, 1]), 
                    MNIST_data.train.labels,
                    batch_size=32,
                    epochs=1,
                    verbose=1,
                    validation_data=(np.reshape(MNIST_data.test.images[-1000:], [-1, 28, 28, 1]), MNIST_data.test.labels[-1000:]))
