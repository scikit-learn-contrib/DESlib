# coding=utf-8

# Author:  Le Thanh Nguyen-Meidine <nmlethanh91@gmail.com>
#
# License: BSD 3 clause

import numpy as np


def is_available():
    try:
        import faiss
        return True
    except ImportError:
        return False


class FaissKNNClassifier:
    """Faiss KNN wrapper.

    Parameters
    ----------

    n_neighbors : int (Default = 5)
                Number of neighbors used in the nearest neighbor search.

    n_jobs : int (Default = None)
             The number of jobs to run in parallel for both fit and predict.
              If -1, then the number of jobs is set to the number of cores.

    algorithm : str (Default = None)
                Algorithm used for nearest

    References
    ----------
    Johnson, Jeff, Matthijs Douze, and Hervé Jégou. "Billion-scale similarity
    search with gpus."
    arXiv preprint arXiv:1702.08734 (2017).
    """

    def __init__(self, n_neighbors=5, n_jobs=None, algorithm=None):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.y = None
        self.num_of_classes = None
        self.index = None

        import faiss
        self.faiss = faiss

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
        _, idx = self.kneighbors(X, self.n_neighbors)
        class_idx = self.y[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.num_of_classes), axis=1,
            arr=class_idx.astype(np.int64))
        preds = np.argmax(counts, axis=1)
        return preds

    def kneighbors(self, X, n_neighbors, return_distance=True):
        """Finds the K-neighbors of a point.

        Parameters
        ----------

        X : array of shape = [n_samples, n_features]
            The input data.

        n_neighbors : int
            Number of neighbors to get (default is the value passed to the
            constructor).

        Returns
        -------
        dists : list of shape = [n_samples, k]
            The distances between the query and each sample in the region of
            competence. The vector is ordered in an ascending fashion.

        idx : list of shape = [n_samples, k]
            Indices of the instances belonging to the region of competence of
            the given query sample.
        """
        X = np.atleast_2d(X).astype(np.float32)
        dist, idx = self.index.search(X, n_neighbors)
        return dist, idx

    def predict_proba(self, X):
        """Estimates the posterior probabilities for sample in X.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The input data.

        Returns
        -------
        preds_proba : array of shape = [n_samples, n_classes]
                          Probabilities estimates for each sample in X.
        """
        _, idx = self.kneighbors(X, self.n_neighbors)
        class_idx = self.y[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.num_of_classes), axis=1,
            arr=class_idx.astype(np.int64))
        preds = np.argmax(counts, axis=1)

        preds_proba = np.zeros((X.shape[0], self.num_of_classes))
        for i in range(preds.shape[0]):
            preds_proba[i] = counts[i] / self.n_neighbors

        return preds_proba

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            Data used to fit the model.

        y : array of shape = [n_samples]
            class labels of each example in X.
        """
        X = np.atleast_2d(X).astype(np.float32)
        X = np.ascontiguousarray(X)
        self.index = self.faiss.IndexFlatL2(X.shape[1])
        self.index.add(X)
        self.y = y
        self.num_of_classes = np.max(self.y) + 1
        return self
