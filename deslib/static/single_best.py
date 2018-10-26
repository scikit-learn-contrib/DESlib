# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from .base import BaseStaticEnsemble
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array


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

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

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

    def __init__(self, pool_classifiers=None, random_state=None):
        super(SingleBest, self).__init__(pool_classifiers=pool_classifiers,
                                         random_state=random_state)

    def fit(self, X, y):
        """Fit the model by selecting the base classifier with the highest
        accuracy in the dataset. The single best classifier is kept in
        self.best_clf and its index is kept in self.best_clf_index.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            Data used to fit the model.

        y : array of shape = [n_samples]
            class labels of each example in X.

        """
        X, y = check_X_y(X, y)

        super(SingleBest, self).fit(X, y)

        performances = np.zeros(self.n_classifiers_)
        for idx, clf in enumerate(self.pool_classifiers_):
            performances[idx] = clf.score(X, self.y_enc_)
        self.best_clf_index_ = np.argmax(idx)
        self.best_clf_ = self.pool_classifiers_[self.best_clf_index_]

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
        predicted_labels = np.array(self.best_clf_.predict(X), dtype=int)
        return self.classes_.take(predicted_labels)

    def predict_proba(self, X):
        """Estimates the posterior probabilities for each class for each sample
        in X. The returned probability estimates for all classes are ordered by
        the label of classes.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The data to be classified

        Returns
        -------
        predicted_proba : array of shape = [n_samples, n_classes]
            Posterior probabilities estimates for each class.

        """
        self._check_is_fitted()

        if "predict_proba" not in dir(self.best_clf_):
            raise ValueError(
                "Base classifier must support the predict_proba function.")

        predicted_proba = self.best_clf_.predict_proba(X)
        return predicted_proba

    def _check_is_fitted(self):
        """Verify if the estimator algorithm was fitted. Raises an error if it
        is not fitted.
        """
        check_is_fitted(self, "best_clf_")
