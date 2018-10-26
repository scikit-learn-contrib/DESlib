import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator


class StackedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, pool_classifiers, meta_classifier):
        self.pool_classifiers = pool_classifiers
        self.n_classifiers = len(self.pool_classifiers)
        self.meta_classifier = meta_classifier

    def fit(self, X, y):
        base_preds = self._predict_base(X)
        self.meta_classifier.fit(base_preds, y)

    def predict(self, X):
        base_preds = self._predict_base(X)
        return self.meta_classifier.predict(base_preds)

    def _predict_base(self, X):
        predictions = np.zeros((X.shape[0], self.n_classifiers),
                               dtype=np.intp)

        for index, clf in enumerate(self.pool_classifiers):
            labels = clf.predict(X)
            predictions[:, index] = labels
        return predictions
