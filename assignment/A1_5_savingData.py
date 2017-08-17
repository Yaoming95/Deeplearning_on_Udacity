from __future__ import print_function

import os

from assignment.A1_4_splitting import data_split
from six.moves import cPickle as pickle
from assignment.A1_P4_shuffling import randomize

if __name__ == '__main__':
    [valid_dataset, valid_labels, train_dataset, train_labels, test_dataset, test_labels] = data_split()
    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
    data_root = ""
    pickle_file = os.path.join(data_root, 'notMNIST.pickle')
    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise