
import pytest
import sys
from sklearn.neighbors import KNeighborsClassifier
from deslib.tests.examples_test import *
from deslib.tests.test_des_integration import load_dataset

try:
    from deslib.util.faiss_knn_wrapper import FaissKNNClassifier
except ImportError:
    pass

@pytest.mark.skipif('faiss' not in sys.modules,
                    reason="requires the faiss library")
def test_faiss_predict():
    rng = np.random.RandomState(123456)
    _, X_test, X_train, y_dsel, y_test, y_train = load_dataset(None, rng)
    k = 7
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    f_knn_test = FaissKNNClassifier(n_neighbors=k)
    f_knn_test.fit(X_train, y_train)
    f_knn_preds = f_knn_test.predict(X_test)

    knn_test = KNeighborsClassifier(n_neighbors=k)
    knn_test.fit(X_train, y_train)
    knn_preds = knn_test.predict(X_test)

    assert ((f_knn_preds - knn_preds).sum() == 0)
