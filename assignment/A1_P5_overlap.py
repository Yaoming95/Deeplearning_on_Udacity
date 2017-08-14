from __future__ import print_function

import hashlib

import numpy as np

from A1_4_splitting import data_split


def get_overlap_index(dataset1, dataset2):
    hashset1 = np.array([hashlib.sha256(img).hexdigest() for img in dataset1])
    hashset2 = np.array([hashlib.sha256(img).hexdigest() for img in dataset2])
    overlap_index = []
    for index, hash1 in enumerate(hashset1):
        if len(np.where(hashset2 == hash1)[0]) > 0:
            overlap_index.append(index)
    return overlap_index

def data_sanitize(dataset1, label1, dataset2):
    overlap_index = get_overlap_index(dataset1, dataset2)
    for index in overlap_index:
        dataset1 = np.delete(dataset1, index, axis=0)
        label1 = np.delete(label1, index)
    return dataset1, label1


[valid_dataset, valid_labels, train_dataset, train_labels, test_dataset, test_labels] = data_split()
(train_dataset, train_labels) = data_sanitize(train_dataset, train_labels,test_dataset)
