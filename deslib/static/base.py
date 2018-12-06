# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

from abc import abstractmethod, ABCMeta
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaseEnsemble, BaggingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted, check_random_state


class BaseStaticEnsemble(BaseEstimator, ClassifierMixin):
    """Base class for static ensembles.

    All static ensemble techniques should inherit from this class.

    Warning: This class should not be instantiated directly, use derived
    classes instead.

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
    Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms.
    John Wiley & Sons, 2004.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
    selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, pool_classifiers=None, random_state=None):
        self.pool_classifiers = pool_classifiers
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the model according to the given training data.

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
        self.random_state_ = check_random_state(self.random_state)

        # Check if the pool of classifiers is None. If yes, use a
        # BaggingClassifier for the pool.
        if self.pool_classifiers is None:
            self.pool_classifiers_ = BaggingClassifier(
                random_state=self.random_state_)
            self.pool_classifiers_.fit(X, y)

        else:
            self.pool_classifiers_ = self.pool_classifiers

        self.n_classifiers_ = len(self.pool_classifiers_)

        # Check if base classifiers are not using LabelEncoder (the case for
        # scikit-learn's ensembles):
        if isinstance(self.pool_classifiers_, BaseEnsemble):
            self.base_already_encoded_ = True
        else:
            self.base_already_encoded_ = False

        self.y_enc_ = self._setup_label_encoder(y)

        self.n_classes_ = self.classes_.size
        self.n_features_ = X.shape[1]

        return self

    def _setup_label_encoder(self, y):
        """
        Setup the label encoder
        """
        self.enc_ = LabelEncoder()
        y_ind = self.enc_.fit_transform(y)
        self.classes_ = self.enc_.classes_

        return y_ind

    def _encode_base_labels(self, y):
        """
        Use label encoder to transform the labels in y into numeric values
        """
        if self.base_already_encoded_:
            return y
        else:
            return self.enc_.transform(y)
