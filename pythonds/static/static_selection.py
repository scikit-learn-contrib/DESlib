# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.base import ClassifierMixin


class StaticSelection(object, ClassifierMixin):
    """description of class"""

    def __init__(self, pool_classifiers, N, scheme='MV'):
        self.scheme = scheme.upper()
        assert scheme in ['MV', 'GA']
        assert 1 > N > 0
        self.perc_select = N
        self.pool = pool_classifiers
        self.n_classifiers = len(pool_classifiers) * N
        self.clf_indices = None

    def fit(self, X, y):
        performances = np.array(self.pool.n_estimators)

        for clf_idx, clf in enumerate(self.pool):
            performances[clf_idx] = clf.score(X, y)

        self.clf_indices = np.argsort(performances)[::-1][0:self.n_classifiers]

        return self

    def predict(self, X):
        return

    def predict_proba(self, X):
        return



