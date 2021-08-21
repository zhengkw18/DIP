# -*- coding: utf-8 -*-

import numpy as np


def load_test_data():
    test = np.load("./data/feature_test.npy")
    return test


"""
{0, 1, 2, 3, 4, 8, 9}
{0, 1, 2, 3, 4, 5, 6}
{0, 1, 2, 3, 4, 7, 9}
{0, 1, 2, 3, 4, 5, 7}
{0, 1, 3, 5, 6, 7, 8}
{0, 1, 3, 4, 5, 6, 8}
{0, 1, 3, 5, 6, 7, 8}
{0, 1, 3, 4, 7, 8, 9}
{0, 1, 2, 3, 5, 7, 9}
{1, 2, 3, 4, 5, 7, 9}
"""

train_val_split = [4, 6, 4, 5, 1, 1, 1, 1, 3, 9]


def load_train_data():
    Xy = np.load("./data/feature_train.npy")
    mask = np.array([int(train_val_split[int(Xy[i][-1])] == Xy[i][-2]) for i in range(Xy.shape[0])])
    Xy_train = Xy[np.where(mask == 0)]
    X_train, y_train = Xy_train[:, :-2], Xy_train[:, -1]
    Xy_val = Xy[np.where(mask == 1)]
    X_val, y_val = Xy_val[:, :-2], Xy_val[:, -1]
    return X_train, y_train, X_val, y_val


# Xy = np.load("./data/feature_train.npy")
# sets = [set() for _ in range(10)]
# for i in range(len(Xy)):
#     sets[int(Xy[i, -1])].add(int(Xy[i, -2]))
# for i in range(10):
#     print(sets[i])