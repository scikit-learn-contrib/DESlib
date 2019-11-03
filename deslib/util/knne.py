import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_X_y
from sklearn.utils import check_array
from deslib.util.prob_functions import softmax


class KNNE(object):
    """"
    Implementation of the K-Nearest Neighbors-Equality technique.

    This implementation fits a different KNN method for each class, and search
    on each class for the nearest examples.

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

    def __init__(self,
                 n_neighbors=7,
                 knn_classifier='sklearn',
                 **kwargs):

        self.n_neighbors = n_neighbors
        self.knn_classifier = knn_classifier
        self.kwargs = kwargs

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            Data used to fit the model.

        y : array of shape = [n_samples]
            class labels of each example in X.
        """
        X, y = check_X_y(X, y)

        self.knns_ = {}
        self.classes_indexes_ = {}
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.n_classes = self.classes_.size

        if self.n_neighbors is not None:
            n_samples = {}
            n_classes = len(set(y))
            idxs = np.bincount(y).argsort()
            self.mdc = int(self.n_neighbors / n_classes)
            self.mod = self.n_neighbors % n_classes

            # for class_ in np.arange(n_classes)[idxs]:
            #     n_samples[class_] = mdc + (1 if mod > 0 else 0)
            #     mod = mod - 1
        else:
            raise ValueError('Either n_neighbors or n_neighbors_per_class'
                             ' needs to be informed!')

        for class_ in self.classes_:
            self.classes_indexes_[class_] = np.argwhere(
                np.array(y) == class_).ravel()
            y_c = y[self.classes_indexes_[class_]]
            X_c = X[self.classes_indexes_[class_], :]
            knn = KNeighborsClassifier(n_neighbors=self.mdc+1,
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
        if X is None:
            X = self.X_

        dists = []
        inds = []

        for class_, knn in self.knns_.items():
            dist_c, ind_c = knn.kneighbors(X)
            real_ind_c = self.classes_indexes_[class_].take(ind_c)
            dists.append(dist_c)
            inds.append(real_ind_c)

        inds = np.concatenate(inds, axis=1)
        dists = np.concatenate(dists, axis=1)

        b = dists.argsort(axis=1)
        a = np.tile(np.arange(b.shape[0]).reshape(-1, 1), (1, b.shape[1]))

        if return_distance:
            return  dists[a, b], inds[a, b]
        else:
            return inds[a, b]

    def predict(self, X):
        """Predict the class label for each sample in X.

        Parameters
        ----------

        X : array of shape = [n_samples, n_features]
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
        proba : array of shape = [n_samples, n_classes], or a list of n_outputs
            of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        X = check_array(X, accept_sparse='csr')

        inds, dists = self.kneighbors(X, self.n_neighbors,
                                      return_distance=True)
        classes = self.y_[inds]
        dists_array = np.empty((X.shape[0], self.n_classes))
        # TODO: check if a more efficient implementation can be done
        for c in self.classes_:
            dists_array[:, c] = np.ma.MaskedArray(dists, classes != c).mean(
                axis=1)
        probas = softmax(1./dists_array)
        return probas
