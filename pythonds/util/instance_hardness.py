# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

"""This file contains the implementation of different Instance Hardness (IH) measures. Instance Hardness is a measure
of how hard a given sample is to be correctly classified by any classification algorithms. Samples associated with a
high degree of hardness are more likely to be misclassified.

In particular, the  k-Disagreeing Neighbors (kDN) is deemed to be strong correlated with the probability that a
sample is misclassified by different learning algorithms.

Reference
----------
Smith, M.R., Martinez, T. and Giraud-Carrier, C., 2014. An instance level analysis of data complexity.
Machine learning, 95(2), pp.225-256
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def k_disagreeing_neighbors(X_train, y_train, X_test, y_test, k):
    """Calculates the k-Disagreeing Neighbors (kDN) metric.

    This measure is strongly correlated with the probability that a sample is misclassified with

    Parameters
    ----------
    X_train : ndarray of shape = [n_samples, n_features] each row corresponding to a different instance
    from the training set.

    y_train : array of shape = [n_samples] containing the labels of each instance from the training set.

    X_test : ndarray of shape = [n_samples, n_features] each row corresponding to a different instance
    from the test set.

    y_test : array of shape = [n_samples] containing the labels of each instance from the test set.

    k : int, size of the neighborhood used to calculate the k-disagreeing neighbors

    Returns
    -------
    hardness : float, with the calculated IH of the corresponding sample

    Reference
    ----------
    Smith, M.R., Martinez, T. and Giraud-Carrier, C., 2014. An instance level analysis of data complexity.
    Machine learning, 95(2), pp.225-256
    """
    knn_c = KNeighborsClassifier(k, )
    knn_c.fit(X_train, y_train)
    k_disagreeing = []
    for idx, test in enumerate(X_test):
        _, [index_neighbors] = knn_c.kneighbors(test.reshape(1, -1))
        label_list = np.array([y_train[idx] for idx in index_neighbors])
        k_disagreeing.append(np.mean(label_list != y_test[idx]))

    hardness = np.array(k_disagreeing)

    return hardness
