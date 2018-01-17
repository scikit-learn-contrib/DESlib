# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError

from deslib.util.aggregation import majority_voting


class StaticSelection(ClassifierMixin):
    """Ensemble model that selects N classifiers with the best performance in a dataset

    Parameters
    ----------
    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support methods "predict".

    pct_classifiers : float (Default = 0.5)
                      percentage of base classifier that should be selected by the selection scheme.


    References
    ----------
    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms. John Wiley & Sons, 2004.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """

    def __init__(self, pool_classifiers, pct_classifiers=0.5):

        if not isinstance(pct_classifiers, float):
            raise TypeError('pct_classifiers should be a float.')

        if pct_classifiers > 1 or pct_classifiers < 0:
            raise ValueError('The parameter pct_classifiers should be a number between 0 and 1.')

        self.perc_select = pct_classifiers
        self.pool_classifiers = pool_classifiers
        self.n_classifiers_pool = len(pool_classifiers)
        self.n_classifiers_ensemble = int(len(pool_classifiers) * pct_classifiers)
        self.clf_indices = None

        self.n_features = None
        self.n_classes = None
        self.classes = None
        self.ensemble = None

    def fit(self, X, y):
        """Fit the static selection model by select an ensemble of classifier containing the base classifiers with
         highest accuracy in the given dataset.

         Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The data to be classified

        y : array of shape = [n_samples]
            Class labels of each sample in X.

        """
        self.classes = np.unique(y)
        self.n_classes = self.classes.size
        self.n_features = X.shape[1]

        performances = np.zeros(len(self.pool_classifiers))

        for clf_idx, clf in enumerate(self.pool_classifiers):
            performances[clf_idx] = clf.score(X, y)

        self.clf_indices = np.argsort(performances)[::-1][0:self.n_classifiers_ensemble]
        self.ensemble = [self.pool_classifiers[clf_idx] for clf_idx in self.clf_indices]

    def predict(self, X):
        """Predict the label of each sample in X and returns the predicted label.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The data to be classified

        Returns
        -------
        predicted_labels : array of shape = [n_samples]
                           Predicted class for each sample in X.
        """
        self._check_is_fitted()
        predicted_labels = majority_voting(self.ensemble, X)

        return predicted_labels

    def _check_is_fitted(self):
        """Verify if the estimator algorithm was fitted.
        Raises an error if it is not fitted.
        """
        if self.ensemble is None:
            raise NotFittedError('Estimator not fitted. Call "fit" before exploiting the model.')