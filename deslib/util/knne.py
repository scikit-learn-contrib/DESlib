import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_X_y


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

    def __init__(self, n_neighbors=7, **kwargs):

        self.n_neighbors = n_neighbors
        self.kwargs = kwargs

    def fit(self, X, y):

        X, y = check_X_y(X, y)

        self.knns_ = {}
        self.classes_indexes_ = {}
        self.X_ = X
        self.y_ = y
        self.classes_ = set(y)

        if self.n_neighbors is not None:
            n_samples = {}
            n_classes = len(set(y))
            idxs = np.bincount(y).argsort()
            mdc = int(self.n_neighbors / n_classes)
            mod = self.n_neighbors % n_classes

            for class_ in np.arange(n_classes)[idxs]:
                n_samples[class_] = mdc + (1 if mod > 0 else 0)
                mod = mod - 1
        else:
            raise ValueError('either n_neighbors or n_neighbors_per_class')

        for class_ in set(y):
            self.classes_indexes_[class_] = np.argwhere(
                np.array(y) == class_).ravel()
            y_c = y[self.classes_indexes_[class_]]
            X_c = X[self.classes_indexes_[class_], :]
            knn = KNeighborsClassifier(n_neighbors=n_samples[class_],
                                       **self.kwargs)
            self.knns_[class_] = knn.fit(X_c, y_c)

        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):

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
            return dists[a, b], inds[a, b]
        else:
            return inds[a, b]

