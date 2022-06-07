# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.utils.validation import check_X_y, check_array

from deslib.static.base import BaseStaticEnsemble


class Oracle(BaseStaticEnsemble):
    """ Abstract method that always selects the base classifier that predicts
    the correct label if such classifier exists. This method is often used to
    measure the upper-limit performance that can be achieved by a dynamic
    classifier selection technique. It is used as a benchmark by several
    dynamic selection algorithms.

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

    n_jobs : int, default=-1
        The number of parallel jobs to run. None means 1 unless in
        a joblib.parallel_backend context. -1 means using all processors.
        Doesn’t affect fit method.

    References
    ----------
    Kuncheva, Ludmila I. "A theoretical study on six classifier fusion
    strategies." IEEE Transactions on Pattern Analysis & Machine Intelligence,
    (2002): 281-286.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
    selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """
    def __init__(self, pool_classifiers=None, random_state=None, n_jobs=-1):
        super(Oracle, self).__init__(pool_classifiers=pool_classifiers,
                                     random_state=random_state,
                                     n_jobs=n_jobs)

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Data used to fit the model.

        y : array of shape (n_samples)
            class labels of each example in X.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            dtype=np.float64,
            order="C",
            accept_large_sparse=False,
        )
        super(Oracle, self).fit(X, y)
        return self

    def predict(self, X, y):
        """Prepare the labels using the Oracle model.

         Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The data to be classified

        y : array of shape (n_samples)
            Class labels of each sample in X.

        Returns
        -------
        predicted_labels : array of shape (n_samples)
                           Predicted class for each sample in X.
        """
        X = check_array(X)
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        y = self.enc_.transform(y)
        preds = [clf.predict(X[:, self.estimator_features_[idx]])
                 for idx, clf in enumerate(self.pool_classifiers_)]
        preds = np.asarray(preds).T
        hit_miss = np.asarray(preds) == y.reshape(-1, 1)
        idx_sel_classifier = hit_miss.argmax(axis=1)
        predicted_labels = preds[np.arange(preds.shape[0]), idx_sel_classifier]

        return self.classes_.take(predicted_labels.astype(int))

    def predict_proba(self, X, y):
        """Estimates the posterior probabilities for each class for each sample
        in X.

        Note that as the Oracle is the ideal classifier selection, the
        classifier that estimate the highest probability for the correct class
        is the selected one.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The data to be classified.

        y : array of shape (n_samples)
            Class labels of each sample in X.

        Returns
        -------
        probas : array of shape (n_samples, n_classes)
            Posterior probabilities estimates for each class.

        """
        X = check_array(X)
        y = self.enc_.transform(y)

        probas = [clf.predict_proba(X[:, self.estimator_features_[idx]])
                  for idx, clf in enumerate(self.pool_classifiers_)]
        probas = np.array(probas).transpose((1, 0, 2))
        best_probas_ids = np.argmax(probas[np.arange(y.size), :, y], axis=1)
        return probas[np.arange(y.size), best_probas_ids, :]

    def score(self, X, y, sample_weights=None):
        """ Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The data to be classified.

        y : array of shape (n_samples)
            Class labels of each sample in X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        accuracy : float
                   Classification accuracy of the Oracle model.
        """
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y, self.predict(X, y),
                                  sample_weight=sample_weights)
        return accuracy
