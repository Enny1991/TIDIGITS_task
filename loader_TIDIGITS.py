import tensorflow as tf
import numpy as np
from libtidigits import *
from scipy.io import loadmat
from tensorflow.contrib.learn.python.learn.datasets import base


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_dataset(path, normalize=True):
    train_mfccs = loadmat(path + "tidigits_mfccs_train")['tidigits_mfccs_train']
    test_mfccs = loadmat(path + "tidigits_mfccs_test")['tidigits_mfccs_test']

    # Set up train set and test set
    X_train = train_mfccs['mfccs_third'][0][0][0]
    y_train = dense_to_one_hot(np.array([char_to_dig(dig) for dig in train_mfccs['dig'][0][0][0]]), 10)
    # y_train = np_utils.to_categorical(np.array([char_to_dig_binary(dig) for dig in train_mfccs['dig'][0][0][0]]))
    X_test = test_mfccs['mfccs_third'][0][0][0]
    y_test = dense_to_one_hot(np.array([char_to_dig(dig) for dig in test_mfccs['dig'][0][0][0]]), 10)
    # y_test = np_utils.to_categorical(np.array([char_to_dig_binary(dig) for dig in test_mfccs['dig'][0][0][0]]))
    if normalize:
        # Normalize
        mean, std = normalizer_params(X_train)
        X_train = test_normalizer(X_train, mean, std)
        X_test = test_normalizer(X_test, mean, std)

    max_len = 0
    x_train_len = []
    x_test_len = []
    for item in X_train:
        max_len = max(max_len, item.shape[1])
        x_train_len += [item.shape[1]]
    for item in X_test:
        max_len = max(max_len, item.shape[1])
        x_test_len += [item.shape[1]]

    X_train = pad_sequences(X_train, max_len=max_len, padding='pre')
    X_test = pad_sequences(X_test, max_len=max_len, padding='pre')
    x_train_len = np.array(x_train_len)
    x_test_len = np.array(x_test_len)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)
    print('Done preparing data.')

    train = DataSet(X_train, y_train, x_train_len)
    test = DataSet(X_test, y_test, x_test_len)
    return base.Datasets(train=train, validation=test, test=test)


class DataSet(object):

    def __init__(self,
                 mfcc,
                 labels,
                 lens):

        assert mfcc.shape[0] == labels.shape[0], (
            'mfcc.shape: %s labels.shape: %s' % (mfcc.shape, labels.shape))
        self._num_examples = mfcc.shape[0]

        idx = np.random.permutation(self.num_examples)
        self._mfcc = mfcc[idx]
        self._labels = labels[idx]
        self._lens = lens[idx]
        self._max_len = mfcc.shape[1]
        self._num_features = mfcc.shape[2]
        self._num_classes = labels.shape[1]
        self._index = 0

        # I have to create positive and negative masks
        pos_mask = np.zeros_like(mfcc)
        for i, l in enumerate(lens):
            pos_mask[i, :l, :] = 1
        neg_mask = np.abs(pos_mask - 1)
        self._pos_mask = pos_mask[idx]
        self._neg_mask = neg_mask[idx]


    @property
    def mfcc(self):
        return self._mfcc

    @property
    def num_features(self):
        return self._num_features

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def max_len(self):
        return self._max_len

    def next_batch(self, batch_size, n_features=None):
        values, labels, lens, pos_mask, neg_mask = self.mfcc[self._index:self._index + batch_size, :, :n_features if n_features is not None else self.num_features], \
                               self.labels[self._index: self._index + batch_size], \
                               self._lens[self._index: self._index + batch_size],  \
                               self._pos_mask[self._index: self._index + batch_size], \
                               self._neg_mask[self._index: self._index + batch_size]
        self._index += batch_size
        if self._index >= self._num_examples or self._num_examples - self._index < batch_size:
            self._index = 0
        return values, labels, lens, pos_mask, neg_mask

    def full_batch(self, n_features=None):
        return self.mfcc[:, :, :n_features if n_features is not None else self.num_features], self.labels, self._lens, self._pos_mask, self._neg_mask


