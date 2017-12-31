# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np


class Oracle:
    """ Abstract method that always selects the base classifier that predicts the correct label if such classifier
    exists. This method is often used to measure the upper-limit performance that can be achieved by a dynamic
    classifier selection technique. It is used as a benchmark by several dynamic selection algorithms

    Parameters
    ----------
    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support methods "predict".

    References
    ----------
    Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms. John Wiley & Sons, 2004.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """

    def __init__(self, pool_classifiers):
        self.pool_classifiers = pool_classifiers
        self.n_classifiers = len(self.pool_classifiers)

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
        predicted_labels = -np.ones(y.size, dtype=int)

        for sample_index, x in enumerate(X):

            for clf in self.pool_classifiers:
                # If one base classifier predicts the correct answer, consider as a correct prediction
                predicted = clf.predict(x.reshape(1, -1))[0]
                if predicted == y[sample_index]:
                    predicted_labels[sample_index] = predicted
                    break

        return predicted_labels

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
        accuracy : Classification accuracy of the Oracle model.
        """
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y, self.predict(X, y))
        return accuracy


