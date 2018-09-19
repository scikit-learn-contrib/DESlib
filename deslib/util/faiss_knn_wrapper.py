import faiss
import numpy as np

class FaissKNNClassifier():
    def __init__(self, n_neighbors=5, n_jobs=None, algorithm=None):
        self.k = n_neighbors
        self.y = None
        self.num_of_classes = None
        self.index = None

    def predict(self, X):
        dist, idx = self.kneighbors(X, self.k)
        class_idx = self.y[idx]
        preds = np.amax(class_idx, axis=1)
        return preds

    def kneighbors(self, X, n_neighbors, return_distance=True):
        X = np.atleast_2d(X).astype(np.float32)
        dist, idx = self.index.search(X, n_neighbors)
        return dist, idx

    def predict_proba(self, X):
        dist, idx = self.kneighbors(X, self.k)
        class_idx = self.y[idx]
        preds = np.amax(class_idx, axis=1)

        #FIXME: can probably be improved for a vectorized version
        preds_proba = np.zeros(X.shape[0], self.num_of_classes)
        for i in range(preds):
            preds_proba[i] = np.bincount(class_idx[i, :]) / self.k

        return preds_proba

    def fit(self, X, y):
        X = np.atleast_2d(X).astype(np.float32)
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X)
        self.y = y
        self.num_of_classes = np.max(self.y) + 1