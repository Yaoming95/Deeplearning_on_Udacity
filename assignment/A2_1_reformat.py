from __future__ import print_function

import numpy as np
from six.moves import cPickle as pickle

from assignment.A2_0_const import *


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def open_file():
    pickle_file = 'notMNIST.pickle'

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


def get_data(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


if __name__ == '__main__':
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = open_file()
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_data(train_dataset,
                                                                                                   train_labels,
                                                                                                   valid_dataset,
                                                                                                   valid_labels,
                                                                                                   test_dataset,
                                                                                                   test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
