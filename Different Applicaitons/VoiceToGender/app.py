
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split


# data loading and processing
# use 20% of data as testing
data = pd.read_csv("voice.csv")

X = data.ix[:, :-1]
Y = data.ix[:, -1].apply(lambda x: np.array([1, 0]) if x == "male" else np.array([0, 1]))

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2017)

x_train.reset_index(drop = True, inplace = True)
y_train.reset_index(drop = True, inplace = True)

input_size = x_train.shape[1]




# tensorflow version
def NN_tensorflow():
    # create a simple NN to train the classification model
    with tf.name_scope("input"):
        x_input = tf.placeholder(tf.float32, [None, input_size], "x_input")
        y_input = tf.placeholder(tf.float32, [None, 2],"y_input")

    # the model is just 2 layer
    with tf.name_scope("net"):

        l1 = tf.layers.dense(inputs = x_input,
            units = 128,
            activation=tf.nn.relu,
            name = "l1"
        )

        l2 = tf.layers.dense(inputs = l1,
            units = 64,
            activation=tf.nn.relu,
            name = "l2"
        )


        pred = tf.layers.dense(inputs = l2,
            units = 2,
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

    # use AdamOptimizer
    with tf.name_scope("train_op"):
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

    # the tf.metrics.accuracy requires the local_variables_initializer
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())


    batch_size = 32
    length = len(x_train) // batch_size

    for epoch in range(100):
        x_train.sample(frac=1).reset_index(drop=True, inplace=True)
        y_train.sample(frac=1).reset_index(drop=True, inplace=True)
        for i in range(length):
            x = x_train[i * batch_size: (i + 1) * batch_size].as_matrix()
            y = np.vstack(y_train[i * batch_size: (i + 1) * batch_size])

            sess.run(train_op, feed_dict={x_input: x, y_input: y})
            if i % 10 == 0:
                result, _ = sess.run([accu, accu_op], feed_dict={x_input: x_test.as_matrix(), y_input: np.vstack(y_test)})
                print(result)


# Keras version
def NN_Keras():

    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.utils import np_utils

    model = Sequential()
    model.add(Dense(128, input_shape=(input_size,)))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax')) 

    model.compile(loss='categorical_crossentropy',metrics=["accuracy"], optimizer='adam') 

    model.fit(np.asmatrix(x_train), np.vstack(y_train),
            batch_size=32, epochs=100,
            verbose=0)
    print(model.test_on_batch(np.asmatrix(x_test), np.vstack(y_test)))


# run it
NN_tensorflow()
NN_Keras()

# output is fine, very close to the output of https://github.com/primaryobjects/voice-gender
# the best is xgboost 99% on testset
# NN can get 96% better than rest of the methods