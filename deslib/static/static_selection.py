# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.metrics import check_scoring
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

from deslib.util.aggregation import majority_voting_rule
from deslib.util.aggregation import predict_proba_ensemble
from .base import BaseStaticEnsemble


class StaticSelection(BaseStaticEnsemble):
    """Ensemble model that selects N classifiers with the best performance in a
    dataset

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

    pct_classifiers : float (Default = 0.5)
        Percentage of base classifier that should be selected by the selection
        scheme.

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

    def __init__(self, pool_classifiers=None, pct_classifiers=0.5,
                 scoring=None, random_state=None, n_jobs=-1):

        super(StaticSelection, self).__init__(
            pool_classifiers=pool_classifiers,
            random_state=random_state,
            n_jobs=n_jobs)
        self.pct_classifiers = pct_classifiers
        self.scoring = scoring

    def fit(self, X, y):
        """Fit the static selection model by select an ensemble of classifier
        containing the base classifiers with highest accuracy in the given
        dataset.

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
        self._validate_parameters()

        super(StaticSelection, self).fit(X, y)

        self.n_classifiers_ensemble_ = int(
            self.n_classifiers_ * self.pct_classifiers)

        performances = np.zeros(self.n_classifiers_)

        if not self.base_already_encoded_:
            y_encoded = y
        else:
            y_encoded = self.enc_.transform(y)

        for clf_idx, clf in enumerate(self.pool_classifiers_):
            scorer = check_scoring(clf, self.scoring)
            performances[clf_idx] = scorer(clf, X[:, self.estimator_features_[
                                                         clf_idx]], y_encoded)

        self.clf_indices_ = np.argsort(performances)[::-1][
                            0:self.n_classifiers_ensemble_]
        self.ensemble_ = [self.pool_classifiers_[clf_idx] for clf_idx in
                          self.clf_indices_]

        return self

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

        votes = np.zeros((X.shape[0], self.n_classifiers_ensemble_))
        for idx, clf in enumerate(self.ensemble_):
            X_space = X[:, self.estimator_features_[self.clf_indices_[idx]]]
            votes[:, idx] = self._encode_base_labels(clf.predict(X_space))

        predicted_labels = majority_voting_rule(votes).astype(int)

        return self.classes_.take(predicted_labels)

    def predict_proba(self, X):
        """Estimates the posterior probabilities for sample in X.

         Parameters
         ----------
         X : array of shape (n_samples, n_features)
             The input data.

         Returns
         -------
         predicted_proba : array of shape (n_samples, n_classes)
                           Probabilities estimates for each sample in X.
         """
        self._check_is_fitted()
        X = check_array(X)
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        self._check_predict_proba()
        features = self.estimator_features_[self.clf_indices_]
        proba = predict_proba_ensemble(self.ensemble_, X, features)
        return proba

    def _validate_parameters(self):

        if not isinstance(self.pct_classifiers, float):
            raise TypeError('pct_classifiers should be a float.')
        if self.pct_classifiers > 1 or self.pct_classifiers < 0:
            raise ValueError(
                'The parameter pct_classifiers should be a number '
                'between 0 and 1.')

    def _check_is_fitted(self):
        """Verify if the estimator algorithm was fitted. Raises an error if it
        is not fitted.
        """
        check_is_fitted(self, "ensemble_")

    def _check_predict_proba(self):
        """ Checks if each base classifier in the ensemble (selected models)
        implements the predict_proba method.

        Raises
        -------
        ValueError
            If the base classifiers do not implements the predict_proba method.
        """
        for clf in self.ensemble_:
            if "predict_proba" not in dir(clf):
                raise ValueError(
                    "All base classifiers should output probability estimates")
