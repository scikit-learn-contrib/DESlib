# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from .base import BaseStaticEnsemble
from deslib.util.aggregation import majority_voting
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array


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

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.


    pct_classifiers : float (Default = 0.5)
        Percentage of base classifier that should be selected by the selection
        scheme.


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

    def __init__(self, pool_classifiers=None,
                 pct_classifiers=0.5,
                 random_state=None):
        super(StaticSelection, self).__init__(
            pool_classifiers=pool_classifiers, random_state=random_state)
        self.pct_classifiers = pct_classifiers

    def fit(self, X, y):
        """Fit the static selection model by select an ensemble of classifier
        containing the base classifiers with highest accuracy in the given
        dataset.

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
        self._validate_parameters()

        X, y = check_X_y(X, y)

        super(StaticSelection, self).fit(X, y)

        self.n_classifiers_ensemble_ = int(
            self.n_classifiers_ * self.pct_classifiers)

        performances = np.zeros(self.n_classifiers_)

        for clf_idx, clf in enumerate(self.pool_classifiers_):
            performances[clf_idx] = clf.score(X, self.y_enc_)

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
        X : array of shape = [n_samples, n_features]
            The data to be classified

        Returns
        -------
        predicted_labels : array of shape = [n_samples]
                           Predicted class for each sample in X.
        """
        X = check_array(X)
        self._check_is_fitted()
        predicted_labels = majority_voting(self.ensemble_, X).astype(int)

        return self.classes_.take(predicted_labels)

    def _check_is_fitted(self):
        """Verify if the estimator algorithm was fitted. Raises an error if it
        is not fitted.
        """
        check_is_fitted(self, "ensemble_")

    def _validate_parameters(self):

        if not isinstance(self.pct_classifiers, float):
            raise TypeError('pct_classifiers should be a float.')
        if self.pct_classifiers > 1 or self.pct_classifiers < 0:
            raise ValueError(
                'The parameter pct_classifiers should be a number '
                'between 0 and 1.')
