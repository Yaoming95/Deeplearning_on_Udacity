from __future__ import print_function

import tensorflow as tf
import numpy as np

from assignment.A2_1_reformat import get_data, open_file
from assignment.A2_2_gradientDescentTraining import accuracy
from assignment.A3_0_const import *


def ReLU_dropout_neural(x, weights, bias):
    layer = tf.add(tf.matmul(x, weights['w1']), bias['b1'])
    layer = tf.nn.relu(layer)
    layer = tf.nn.dropout(layer, keep_prob)
    layer = tf.matmul(layer, weights['w2']) + bias['b2']
    return layer

if __name__ == '__main__':
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = open_file()
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_data(train_dataset,
                                                                                                   train_labels,
                                                                                                   valid_dataset,
                                                                                                   valid_labels,
                                                                                                   test_dataset,
                                                                                                   test_labels)
    beta = 0.1
    num_steps = int(num_steps / 4)
    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
        tf_train_labels = tf.constant(train_labels[:train_subset])
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        weights = {
            'w1' : tf.Variable(tf.truncated_normal([image_size*image_size, hidden_nodes])),
            'w2' : tf.Variable(tf.truncated_normal([hidden_nodes, num_labels])),
        }
        bias = {
            'b1': tf.Variable(tf.zeros([hidden_nodes])),
            'b2': tf.Variable(tf.zeros([num_labels])),
        }

        logits = ReLU_dropout_neural(tf_train_dataset, weights, bias)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
        regularization = tf.add(tf.nn.l2_loss(weights['w1']) , tf.nn.l2_loss(weights['w2']))
        loss = tf.add(loss, beta*regularization)

        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(ReLU_dropout_neural(tf_valid_dataset, weights, bias))
        test_prediction = tf.nn.softmax(ReLU_dropout_neural(tf_test_dataset, weights, bias))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            for step in range(num_steps):
                _, l, predictions = session.run([optimizer, loss, train_prediction])
                if (step % 5 == 0):
                    print('Loss at step %d: %f' % (step, l))
                    print('Training accuracy: %.1f%%' % accuracy(
                        predictions, train_labels[:train_subset, :]))
                    # Calling .eval() on valid_prediction is basically like calling run(), but
                    # just to get that one numpy array. Note that it recomputes all its graph
                    # dependencies.
                    print('Validation accuracy: %.1f%%' % accuracy(
                        valid_prediction.eval(), valid_labels))
            print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
