# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from .base import BaseStaticEnsemble
from sklearn.utils.validation import check_X_y, check_array


class Oracle(BaseStaticEnsemble):
    """ Abstract method that always selects the base classifier that predicts
    the correct label if such classifier exists. This method is often used to
    measure the upper-limit performance that can be achieved by a dynamic
    classifier selection technique. It is used as a benchmark by several
    dynamic selection algorithms


        Parameters
    ----------
    pool_classifiers : list of classifiers (Default = None)
        The generated_pool of classifiers trained for the corresponding
        classification problem. Each base classifiers should support the method
        "predict". If None, then the pool of classifiers is a bagging
        classifier.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    References
    ----------
    Kuncheva, Ludmila I. "A theoretical study on six classifier fusion
    strategies." IEEE Transactions on Pattern Analysis & Machine Intelligence,
    (2002): 281-286.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
    selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """

    def __init__(self, pool_classifiers=None, random_state=None):
        super(Oracle, self).__init__(pool_classifiers=pool_classifiers,
                                     random_state=random_state)

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            Data used to fit the model.

        y : array of shape = [n_samples]
            class labels of each example in X.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        super(Oracle, self).fit(X, y)
        return self

    def predict(self, X, y):
        """Prepare the labels using the Oracle model.

         Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The data to be classified

        y : array of shape = [n_samples]
            Class labels of each sample in X.

        Returns
        -------
        predicted_labels : array of shape = [n_samples]
                           Predicted class for each sample in X.
        """

        X = check_array(X)
        predicted_labels = -np.ones(y.size, dtype=int)

        for sample_index, x in enumerate(X):

            for clf in self.pool_classifiers_:
                # If one base classifier predicts the correct answer, consider
                # as a correct prediction
                predicted = clf.predict(x.reshape(1, -1))[0]
                if predicted == y[sample_index]:
                    predicted = int(predicted)
                    predicted_labels[sample_index] = predicted
                    break
                predicted_labels[sample_index] = predicted

        return self.classes_.take(predicted_labels)

    def score(self, X, y):
        """Prepare the labels using the Oracle model.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The data to be classified

        y : array of shape = [n_samples]
            Class labels of each sample in X.

        Returns
        -------
        accuracy : float
                   Classification accuracy of the Oracle model.
        """
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y, self.predict(X, y))
        return accuracy
