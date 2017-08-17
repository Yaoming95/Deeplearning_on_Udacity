from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from assignment.util.LabelLetterEnum import LabelLetterEnum

from assignment.A1_4_splitting import data_split


def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels


def verify(dataset,labels = None):
    sample_idx = np.random.randint(len(dataset))  # pick a random image index
    sample_image = dataset[sample_idx, :, :]  # extract a 2D slice
    plt.figure()
    plt.imshow(sample_image)  # display it
    plt.show()
    if not (labels is None):
        print (LabelLetterEnum(labels[sample_idx]))


if __name__ == '__main__':
    train_size = 200000
    valid_size = 10000
    test_size = 10000

    [valid_dataset, valid_labels, train_dataset, train_labels, test_dataset, test_labels] = data_split()

    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

    [valid_dataset, valid_labels, train_dataset, train_labels, test_dataset, test_labels] = verify(train_dataset, train_labels)

