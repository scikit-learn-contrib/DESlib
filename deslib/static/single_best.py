# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.metrics import check_scoring
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

from .base import BaseStaticEnsemble


class SingleBest(BaseStaticEnsemble):
    """Classification method that selects the classifier in the pool with
    highest score to be used for classification. Usually, the performance of
    the single best classifier is estimated based on the validation data.

    Parameters
    ----------
    pool_classifiers : list of classifiers (Default = None)
        The generated_pool of classifiers trained for the corresponding
        classification problem. Each base classifiers should support the method
        "predict". If None, then the pool of classifiers is a bagging
        classifier.

    scoring : string, callable (default = None)
        A single string or a callable to evaluate the predictions on the
        validation set.

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
    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection
    of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms.
    John Wiley & Sons, 2004.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
    selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.
    """

    def __init__(self, pool_classifiers=None, scoring=None,
                 random_state=None, n_jobs=-1):
        super(SingleBest, self).__init__(pool_classifiers=pool_classifiers,
                                         random_state=random_state,
                                         n_jobs=n_jobs)
        self.scoring = scoring

    def fit(self, X, y):
        """Fit the model by selecting the base classifier with the highest
        accuracy in the dataset. The single best classifier is kept in
        self.best_clf and its index is kept in self.best_clf_index.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Data used to fit the model.

        y : array of shape (n_samples)
            class labels of each example in X.

        """
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            dtype=np.float64,
            order="C",
            accept_large_sparse=False,
        )

        super(SingleBest, self).fit(X, y)

        if not self.base_already_encoded_:
            y_encoded = y
        else:
            y_encoded = self.enc_.transform(y)

        performances = self._estimate_performances(X, y_encoded)
        self.best_clf_index_ = np.argmax(performances)
        self.best_clf_ = self.pool_classifiers_[self.best_clf_index_]

        return self

    def _estimate_performances(self, X, y):
        performances = np.zeros(self.n_classifiers_)
        for idx, clf in enumerate(self.pool_classifiers_):
            scorer = check_scoring(clf, self.scoring)
            performances[idx] = scorer(clf,
                                       X[:, self.estimator_features_[idx]], y)
        return performances

    def predict(self, X):
        """Predict the label of each sample in X and returns the predicted
        label.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The data to be classified

        Returns
        -------
        predicted_labels : array of shape (n_samples)
                           Predicted class for each sample in X.
        """
        self._check_is_fitted()
        X = check_array(X)
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        predictions = self.best_clf_.predict(
            X[:, self.estimator_features_[self.best_clf_index_]])

        predictions = self._encode_base_labels(predictions)
        return self.classes_.take(predictions.astype(int))

    def predict_proba(self, X):
        """Estimates the posterior probabilities for each class for each sample
        in X. The returned probability estimates for all classes are ordered by
        the label of classes.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The data to be classified

        Returns
        -------
        predicted_proba : array of shape (n_samples, n_classes)
            Posterior probabilities estimates for each class.

        """
        self._check_is_fitted()
        if "predict_proba" not in dir(self.best_clf_):
            raise ValueError(
                "Base classifier must support the predict_proba function.")
        X = check_array(X)
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        predicted_proba = self.best_clf_.predict_proba(
            X[:, self.estimator_features_[self.best_clf_index_]])
        return predicted_proba

    def _check_is_fitted(self):
        """Verify if the estimator algorithm was fitted. Raises an error if it
        is not fitted.
        """
        check_is_fitted(self, "best_clf_")
