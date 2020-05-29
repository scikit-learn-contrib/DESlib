import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_X_y
from sklearn.utils import check_array

from deslib.util import faiss_knn_wrapper
from deslib.util.prob_functions import softmax


class KNNE(BaseEstimator):
    """"
    Implementation of the K-Nearest Neighbors-Equality technique.

    This implementation fits a different KNN method for each class, and search
    on each class for the nearest examples.

    Parameters
    ----------

    n_neighbors : int, (default = 7)
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    algorithm : str = ['knn', 'faiss]', (default = 'knn')
        Whether to use scikit-learn or faiss for nearest neighbors estimation.

    References
    ----------
    Sierra, Basilio, Elena Lazkano, Itziar Irigoien, Ekaitz Jauregi,
    and Iñigo Mendialdua. "K nearest neighbor equality: giving equal chance
    to all existing classes."
    Information Sciences 181, no. 23 (2011): 5158-5168.

    Mendialdua, Iñigo, José María Martínez-Otzeta, I. Rodriguez-Rodriguez,
    T. Ruiz-Vazquez, and Basilio Sierra. "Dynamic selection of the best base
    classifier in one versus one." Knowledge-Based Systems 85 (2015): 298-306.

    Cruz, Rafael MO, Dayvid VR Oliveira, George DC Cavalcanti,
    and Robert Sabourin. "FIRE-DES++: Enhanced online pruning of base
    classifiers for dynamic ensemble selection."
    Pattern Recognition 85 (2019): 149-160.
    """

    def __init__(self, n_neighbors=7, knn_classifier='sklearn', **kwargs):

        self.n_neighbors = n_neighbors
        self.knn_classifier = knn_classifier
        self.kwargs = kwargs

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Data used to fit the model.

        y : array of shape (n_samples)
            class labels of each example in X.
        """
        X, y = check_X_y(X, y)

        self.knns_ = {}
        self.classes_indexes_ = {}
        self.fit_X_ = X
        self.fit_y_ = y
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size

        # Checking inputs
        self._check_n_neighbors(self.n_neighbors)
        self._set_knn_type()

        tmp = self._handle_n_neighbors(self.n_neighbors)
        self._mdc, self._mod, self._neighbors_per_class = tmp
        for class_ in self.classes_:
            self.classes_indexes_[class_] = np.argwhere(
                np.array(y) == class_).ravel()
            y_c = y[self.classes_indexes_[class_]]
            X_c = X[self.classes_indexes_[class_], :]
            knn = self.knn_type_(n_neighbors=self._neighbors_per_class,
                                 **self.kwargs)
            self.knns_[class_] = knn.fit(X_c, y_c)

        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.
        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        n_neighbors : int
            Number of neighbors to get (default is the value
            passed to the constructor).

        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        Returns
        -------
        dist : array
            Array representing the lengths to points, only present if
            return_distance=True

        ind : array
            Indices of the nearest points in the population matrix.
        """
        if n_neighbors is None:
            neighbors_per_class = self._neighbors_per_class
            mdc = self._mdc
            mod = self._mod
        else:
            mdc, mod, neighbors_per_class = self._handle_n_neighbors(
                n_neighbors)

        if X is None:
            X = self.fit_X_

        dists = []
        inds = []
        mod_dists = []
        mod_inds = []
        for class_, knn in self.knns_.items():
            dist_c, ind_c = knn.kneighbors(X, neighbors_per_class)
            real_ind_c = self.classes_indexes_[class_].take(ind_c)
            dists.append(dist_c[:, 0:mdc])
            inds.append(real_ind_c[:, 0:mdc])
            if mod > 0:
                mod_dists.append(dist_c[:, -1].reshape(-1, 1))
                mod_inds.append(real_ind_c[:, -1].reshape(-1, 1))

        dists, inds = self._organize_neighbors(dists, inds)

        if mod > 0:
            mod_dists, mod_inds = self._organize_neighbors(mod_dists, mod_inds)
            dists = np.hstack((dists, mod_dists[:, 0:mod]))
            inds = np.hstack((inds, mod_inds[:, 0:mod]))

        if return_distance:
            return dists, inds
        else:
            return inds

    def predict(self, X):
        """Predict the class label for each sample in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        preds : array, shape (n_samples,)
                Class labels for samples in X.
        """
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        proba : array of shape (n_samples, n_classes), or a list of n_outputs
            of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        X = check_array(X, accept_sparse='csr')

        dists, inds = self.kneighbors(X, return_distance=True)
        classes = self.fit_y_[inds]
        dists_array = np.empty((X.shape[0], self.n_classes_))
        # TODO: check if a more efficient implementation can be done
        for c in self.classes_:
            dists_array[:, c] = np.ma.MaskedArray(dists, classes != c).mean(
                axis=1)
        probas = softmax(1. / dists_array)
        return probas

    def _set_knn_type(self):

        if self.knn_classifier is None or self.knn_classifier in ['knn',
                                                                  'sklearn']:
            self.knn_type_ = KNeighborsClassifier

        elif self.knn_classifier == 'faiss':
            if not faiss_knn_wrapper.is_available():
                raise ImportError(
                    'Using knn_classifier="faiss" requires that the FAISS '
                    'library be installed.Please check the Installation '
                    'Guide.')
            self.knn_type_ = faiss_knn_wrapper.FaissKNNClassifier

        elif callable(self.knn_classifier):
            self.knn_type_ = self.knn_classifier
        else:
            raise ValueError('"knn_classifier" should be one of the following '
                             '["knn", "faiss", None] or an estimator class.')

    def _organize_neighbors(self, dists, inds):
        inds = np.concatenate(inds, axis=1)
        dists = np.concatenate(dists, axis=1)
        b = dists.argsort(axis=1)
        a = np.tile(np.arange(b.shape[0]).reshape(-1, 1), (1, b.shape[1]))
        dists, inds = dists[a, b], inds[a, b]
        return dists, inds

    def _check_n_neighbors(self, n_neighbors):
        if n_neighbors is None:
            raise ValueError('"n_neighbors" is required for the KNN-E model.')

        if n_neighbors < self.n_classes_:
            raise ValueError('"n_neighbors" must be equals or higher than '
                             'the number of classes. Got {}.'
                             .format(n_neighbors))

        if not np.issubdtype(type(n_neighbors), np.integer):
            raise TypeError(
                "n_neighbors does not take {} value, "
                "enter integer value".format(type(n_neighbors)))

    def _handle_n_neighbors(self, n_neighbors):
        mdc = int(n_neighbors / self.n_classes_)
        mod = n_neighbors % self.n_classes_
        if mod > 0:
            warnings.warn('"n_neighbors" is not a multiple of "n_classes". Got'
                          '{} and {}.One or more classes will have one less'
                          ' instance.'.format(n_neighbors,
                                              self.n_classes_))
            n_per_class = mdc + 1
        else:
            n_per_class = mdc
        return mdc, mod, n_per_class
