# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.base import ClassifierMixin


class Oracle(ClassifierMixin):
    """ Classification method that selects the classifier in the pool with highest
    score to be used for classification. Usually, the performance of the single best classifier
    is estimated based on the validation data.



    """

    def __init__(self, pool_classifiers):
        self.pool_classifiers = pool_classifiers
        self.n_classifiers = len(self.pool_classifiers)

    def performance(self, X, y):
        predictions = np.zeros(y.size)
        for sample_index, x in enumerate(X):
            for clf in self.pool_classifiers:
                if clf.predict(x.reshape(1, -1))[0] == y[sample_index]:
                    predictions[sample_index] = 1

        return np.mean(predictions)

if __name__ == "__main__":

    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import Perceptron
    from pythonds.util.generate_pool import generate_pool
    iris = load_iris()
    model = DecisionTreeClassifier()
    model = Perceptron()
    pool = generate_pool(iris, model, 2)

    ranking = Oracle(pool)
    print(ranking.performance(iris.data, iris.target))
