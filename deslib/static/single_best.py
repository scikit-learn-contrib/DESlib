# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError


class SingleBest(ClassifierMixin):
    """Classification method that selects the classifier in the pool with highest
    score to be used for classification. Usually, the performance of the single best classifier
    is estimated based on the validation data.
    
    Parameters
    ----------
    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support methods "predict".

    References
    ----------
    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms. John Wiley & Sons, 2004.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """
    def __init__(self, pool_classifiers):
        self.pool_classifiers = pool_classifiers
        self.n_classifiers = len(self.pool_classifiers)

        self.best_clf = None
        self.best_clf_index = None
        self.n_features = None
        self.n_classes = None
        self.classes = None

    def fit(self, X, y):
        """Fit the model by selecting the base classifier with the highest accuracy in the dataset.
        The single best classifier is kept in self.best_clf and its index is kept in self.best_clf_index.

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

        performances = np.zeros(self.n_classifiers)
        for idx, clf in enumerate(self.pool_classifiers):
            performances[idx] = clf.score(X, y)
        self.best_clf_index = np.argmax(idx)
        self.best_clf = self.pool_classifiers[self.best_clf_index]

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
        predicted_labels = self.best_clf.predict(X)
        return predicted_labels

    def predict_proba(self, X):
        """Estimates the posterior probabilities for each class for each sample in X. The returned probability
         estimates for all classes are ordered by the label of classes.

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

        if "predict_proba" not in dir(self.best_clf):
            raise ValueError("Base classifier must support the predict_proba function.")

        predicted_proba = self.best_clf.predict_proba(X)
        return predicted_proba

    def _check_is_fitted(self):
        """Verify if the estimator algorithm was fitted. Raises an error if it is not fitted.
        """
        if self.best_clf is None:
                raise NotFittedError('Estimator not fitted. Call "fit" before exploiting the model.')

