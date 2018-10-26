import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import BaseEnsemble
from sklearn.preprocessing import LabelEncoder


class StackedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, pool_classifiers, meta_classifier):
        self.pool_classifiers = pool_classifiers
        self.n_classifiers = len(self.pool_classifiers)
        self.meta_classifier = meta_classifier

        if isinstance(self.pool_classifiers, BaseEnsemble):
            self.base_already_encoded = True
        else:
            self.base_already_encoded = False

    def fit(self, X, y):
        # Check if base classifiers are not using LabelEncoder (the case for
        # scikit-learn's ensembles):

        y_ind = self._setup_label_encoder(y)
        base_preds = self._predict_proba_base(X)
        self.meta_classifier.fit(base_preds, y_ind)

    def predict(self, X):
        base_preds = self._predict_proba_base(X)
        return self.meta_classifier.predict(base_preds)

    def _predict_proba_base(self, X):
        probabilities = np.zeros(
            (X.shape[0], self.n_classifiers, self.n_classes_))

        for index, clf in enumerate(self.pool_classifiers):
            probabilities[:, index] = clf.predict_proba(X)
        return probabilities.reshape(X.shape[0],
                                     self.n_classifiers * self.n_classes_)

    def _setup_label_encoder(self, y):
        self.enc_ = LabelEncoder()
        y_ind = self.enc_.fit_transform(y)
        self.classes_ = self.enc_.classes_
        self.n_classes_ = self.classes_.size

        return y_ind

    def _encode_base_labels(self, y):
        if self.base_already_encoded:
            return y
        else:
            return self.enc_.transform(y)
