# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.base import ClassifierMixin

from pythonds.util.prob_functions import softmax


class SingleBest(ClassifierMixin):
    """Classification method that selects the classifier in the pool with highest
    score to be used for classification. Usually, the performance of the single best classifier
    is estimated based on the validation data.
    
    
    
    """

    def __init__(self, pool_classifiers):
        self.pool_classifiers = pool_classifiers
        self.n_classifiers = len(self.pool_classifiers)
        self.best_clf = None

    def fit(self, X, y):
        performances = np.zeros(self.n_classifiers)
        for idx, clf in enumerate(self.pool_classifiers):
            performances[idx] = clf.score(X, y)
        best = np.argmax(idx)
        return self.pool_classifiers[best]

    def predict(self, X):
        return self.best_clf.predict(X)

    def predict_proba(self, X):
        if "predict_proba" in dir(self.best_clf):
            return self.best_clf.predict_proba(X)
        elif "decision_function" in dir(self.best_clf):
            return softmax(self.best_clf.decision_function(X))
        else:
            raise ValueError("Base classifiers must be able to estimate probabilities.")


