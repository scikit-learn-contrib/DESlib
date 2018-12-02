import numpy as np
from sklearn.linear_model import Perceptron
from deslib.dcs.ola import OLA
from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(OLA)


def test_estimate_competence_batch(example_estimate_competence):
    _, _, neighbors, distances, dsel_processed, _ = example_estimate_competence
    expected = np.array([[0.57142857,  0.71428571,  0.71428571],
                         [0.71428571,  0.85714286,  0.71428571],
                         [0.57142857, 0.71428571, 0.57142857]])

    ola_test = OLA()
    ola_test.DSEL_processed_ = dsel_processed

    ola_test.DFP_mask = np.ones((3, 3))
    query = np.array([[1, 1], [1, 1], [1, 1]])
    competences = ola_test.estimate_competence(query, neighbors,
                                               distances=distances)
    assert np.allclose(competences, expected)


# Test if the class is raising an error when the base classifiers do not
# implements the predict_proba method. In this case the test should not raise
# an error since this class does not require base classifiers that
# can estimate probabilities
def test_predict_proba(create_X_y):
    X, y = create_X_y
    clf1 = Perceptron()
    clf1.fit(X, y)
    OLA([clf1, clf1]).fit(X, y)
