# coding=utf-8

# Author:  Le Thanh Nguyen-Meidine <nmlethanh91@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.utils.validation import check_is_fitted


def is_available():
    try:
        import faiss
        return True
    except ImportError:
        return False


class FaissKNNClassifier:
    """ Scikit-learn wrapper interface for Faiss KNN.

    Parameters
    ----------
    n_neighbors : int (Default = 5)
                Number of neighbors used in the nearest neighbor search.

    n_jobs : int (Default = None)
             The number of jobs to run in parallel for both fit and predict.
              If -1, then the number of jobs is set to the number of cores.

    algorithm : {'brute', 'voronoi'} (Default = 'brute')

        Algorithm used to compute the nearest neighbors:

            - 'brute' will use the :class: `IndexFlatL2` class from faiss.
            - 'voronoi' will use :class:`IndexIVFFlat` class from faiss.
            - 'hierarchical' will use :class:`IndexHNSWFlat` class from faiss.

        Note that selecting 'voronoi' the system takes more time during
        training, however it can significantly improve the search time
        on inference. 'hierarchical' produce very fast and accurate indexes,
        however it has a higher memory requirement. It's recommended when
        you have a lots of RAM or the dataset is small.

        For more information see: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

    n_cells : int (Default = 100)
        Number of voronoi cells. Only used when algorithm=='voronoi'.

    n_probes : int (Default = 1)
        Number of cells that are visited to perform the search. Note that the
        search time roughly increases linearly with the number of probes.
        Only used when algorithm=='voronoi'.

    References
    ----------
    Johnson Jeff, Matthijs Douze, and Hervé Jégou. "Billion-scale similarity
    search with gpus." arXiv preprint arXiv:1702.08734 (2017).
    """

    def __init__(self,
                 n_neighbors=5,
                 n_jobs=None,
                 algorithm='brute',
                 n_cells=100,
                 n_probes=1):

        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.n_cells = n_cells
        self.n_probes = n_probes

        import faiss
        self.faiss = faiss

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
        idx = self.kneighbors(X, self.n_neighbors, return_distance=False)
        class_idx = self.y_[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes_), axis=1,
            arr=class_idx.astype(np.int16))
        preds = np.argmax(counts, axis=1)
        return preds

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.

        Parameters
        ----------

        X : array of shape (n_samples, n_features)
            The input data.

        n_neighbors : int
            Number of neighbors to get (default is the value passed to the
            constructor).

        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        Returns
        -------
        dists : list of shape = [n_samples, k]
            The distances between the query and each sample in the region of
            competence. The vector is ordered in an ascending fashion.

        idx : list of shape = [n_samples, k]
            Indices of the instances belonging to the region of competence of
            the given query sample.
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        elif n_neighbors <= 0:
            raise ValueError("Expected n_neighbors > 0."
                             " Got {}" .format(n_neighbors))
        else:
            if not np.issubdtype(type(n_neighbors), np.integer):
                raise TypeError(
                    "n_neighbors does not take {} value, "
                    "enter integer value" .format(type(n_neighbors)))

        check_is_fitted(self, 'index_')

        X = np.atleast_2d(X).astype(np.float32)
        dist, idx = self.index_.search(X, n_neighbors)
        if return_distance:
            return dist, idx
        else:
            return idx

    def predict_proba(self, X):
        """Estimates the posterior probabilities for sample in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        preds_proba : array of shape (n_samples, n_classes)
                          Probabilities estimates for each sample in X.
        """
        idx = self.kneighbors(X, self.n_neighbors, return_distance=False)
        class_idx = self.y_[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes_), axis=1,
            arr=class_idx.astype(np.int16))

        preds_proba = counts / self.n_neighbors

        return preds_proba

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Data used to fit the model.

        y : array of shape (n_samples)
            class labels of each example in X.
        """
        X = np.atleast_2d(X).astype(np.float32)
        X = np.ascontiguousarray(X)
        d = X.shape[1]  # dimensionality of the feature vector
        self._prepare_knn_algorithm(X, d)
        self.index_.add(X)
        self.y_ = y
        self.n_classes_ = np.unique(y).size
        return self

    def _prepare_knn_algorithm(self, X, d):
        if self.algorithm == 'brute':
            self.index_ = self.faiss.IndexFlatL2(d)
        elif self.algorithm == 'voronoi':
            quantizer = self.faiss.IndexFlatL2(d)
            self.index_ = self.faiss.IndexIVFFlat(quantizer, d, self.n_cells)
            self.index_.train(X)
            self.index_.nprobe = self.n_probes
        elif self.algorithm == 'hierarchical':
            self.index_ = self.faiss.IndexHNSWFlat(d, 32)
            self.index_.hnsw.efConstruction = 40
        else:
            raise ValueError("Invalid algorithm option."
                             " Expected ['brute', 'voronoi', 'hierarchical'], "
                             "got {}" .format(self.algorithm))
