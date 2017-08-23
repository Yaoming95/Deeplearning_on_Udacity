from __future__ import print_function

import tensorflow as tf

from assignment.A2_1_reformat import get_data, open_file
from assignment.A2_2_gradientDescentTraining import accuracy
from assignment.A3_0_const import *

if __name__ == '__main__':
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = open_file()
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_data(train_dataset,
                                                                                                   train_labels,
                                                                                                   valid_dataset,
                                                                                                   valid_labels,
                                                                                                   test_dataset,
                                                                                                   test_labels)
    # With gradient descent training, even this much data is prohibitive.
    # Subset the training data for faster turnaround.
    train_subset = 10000

    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
        tf_train_labels = tf.constant(train_labels[:train_subset])
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights = tf.Variable(
            tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
        regularization = tf.nn.l2_loss(weights)
        loss = tf.add(loss, beta*regularization)

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

        with tf.Session(graph=graph) as session:
            # This is a one-time operation which ensures the parameters get initialized as
            # we described in the graph: random weights for the matrix, zeros for the
            # biases.
            tf.global_variables_initializer().run()
            print('Initialized')
            for step in range(num_steps):
                # Run the computations. We tell .run() that we want to run the optimizer,
                # and get the loss value and the training predictions returned as numpy
                # arrays.
                _, l, predictions = session.run([optimizer, loss, train_prediction])
                if (step % 100 == 0):
                    print('Loss at step %d: %f' % (step, l))
                    print('Training accuracy: %.1f%%' % accuracy(
                        predictions, train_labels[:train_subset, :]))
                    # Calling .eval() on valid_prediction is basically like calling run(), but
                    # just to get that one numpy array. Note that it recomputes all its graph
                    # dependencies.
                    print('Validation accuracy: %.1f%%' % accuracy(
                        valid_prediction.eval(), valid_labels))
            print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))