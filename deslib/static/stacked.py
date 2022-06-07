import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from deslib.static.base import BaseStaticEnsemble


class StackedClassifier(BaseStaticEnsemble):
    """ A Stacking classifier.

    Parameters
    ----------
    pool_classifiers : list of classifiers (Default = None)
        The generated_pool of classifiers trained for the corresponding
        classification problem. Each base classifiers should support the method
        "predict" and "predict_proba". If None, then the pool of classifiers
        is a bagging classifier.

    meta_classifier : object or None, optional (default=None)
        Classifier model used to aggregate the output of the base classifiers.
        If None, a :class:`LogisticRegression` classifier is used.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    passthrough : bool (default=False)
        When False, only the predictions of estimators will be used as
        training data for the meta-classifier. When True, the
        meta-classifier is trained on the predictions as well as the
        original training data.

    n_jobs : int, default=-1
        The number of parallel jobs to run. None means 1 unless in
        a joblib.parallel_backend context. -1 means using all processors.
        Doesn’t affect fit method.

    References
    ----------
    Wolpert, David H. "Stacked generalization." Neural networks 5,
    no. 2 (1992): 241-259.

    Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms.
    John Wiley & Sons, 2004.
    """

    def __init__(self, pool_classifiers=None, meta_classifier=None,
                 passthrough=False, random_state=None, n_jobs=-1):

        super(StackedClassifier, self).__init__(
            pool_classifiers=pool_classifiers,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.meta_classifier = meta_classifier
        self.passthrough = passthrough

    def fit(self, X, y):
        """Fit the model by training a meta-classifier on the outputs of the
        base classifiers

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
        super(StackedClassifier, self).fit(X, y)
        base_preds = self._predict_proba_base(X)
        X_meta = self._connect_input(X, base_preds)

        # Prepare the meta-classifier
        if self.meta_classifier is None:
            self.meta_classifier_ = LogisticRegression(
                solver='lbfgs',
                multi_class='auto',
                max_iter=1000,
                random_state=self.random_state_)

        else:
            self.meta_classifier_ = self.meta_classifier

        self.meta_classifier_.fit(X_meta, self.y_enc_)

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
        check_is_fitted(self, "meta_classifier_")
        X = check_array(X)
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        base_preds = self._predict_proba_base(X)
        X_meta = self._connect_input(X, base_preds)
        preds = self.meta_classifier_.predict(X_meta)
        return self.classes_.take(preds)

    def predict_proba(self, X):
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
        check_is_fitted(self, "meta_classifier_")
        X = check_array(X)
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        # Check if the meta-classifier can output probabilities
        if not hasattr(self.meta_classifier_, "predict_proba"):
            raise ValueError("Meta-classifier does not implement the"
                             " predict_proba method.")

        base_preds = self._predict_proba_base(X)
        X_meta = self._connect_input(X, base_preds)

        return self.meta_classifier_.predict_proba(X_meta)

    def _connect_input(self, X, base_preds):
        if self.passthrough:
            X_meta = np.hstack((base_preds, X))
        else:
            X_meta = base_preds
        return X_meta

    def _predict_proba_base(self, X):
        """ Get the predictions (probabilities) of each base classifier in the
        pool for all samples in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The test examples.

        Returns
        -------
        probabilities : array of shape = [n_samples, n_classifiers x n_classes]
                        Probabilities estimates of each base classifier for all
                        test samples.
        """
        # Check if base classifiers implement the predict proba method.
        self._check_predict_proba()

        probas = np.zeros(
            (X.shape[0], self.n_classifiers_, self.n_classes_))

        for index, clf in enumerate(self.pool_classifiers_):
            probas[:, index] = clf.predict_proba(
                X[:, self.estimator_features_[index]])

        probas = probas.reshape(X.shape[0],
                                self.n_classifiers_ * self.n_classes_)

        # remove first column as both features are collinear.
        if self.n_classes_ == 2:
            probas = probas[:, ::2]

        return probas

    def _check_predict_proba(self):
        """ Checks if each base classifier in the pool implements the
        predict_proba method.

        Raises
        ------
        ValueError
            If the base classifiers do not implements the predict_proba method.
        """
        for clf in self.pool_classifiers_:
            if "predict_proba" not in dir(clf):
                raise ValueError(
                    "All base classifiers should output probability estimates")
