import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.utils.estimator_checks import check_estimator

from deslib.des.des_p import DESP


def test_check_estimator():
    check_estimator(DESP)


# Test the estimate competence method receiving n samples as input
def test_estimate_competence_batch(example_estimate_competence,
                                   create_pool_classifiers):
    query = np.ones((3, 2))
    X, y, neighbors, distances, dsel_processed, _ = example_estimate_competence

    expected = np.array([[0.57142857, 0.4285714, 0.57142857],
                         [0.71428571, 0.2857142, 0.71428571],
                         [0.2857142, 0.71428571, 0.2857142]])

    des_p_test = DESP(create_pool_classifiers)
    des_p_test.fit(X, y)
    competences = des_p_test.estimate_competence(query, neighbors, distances)
    assert np.allclose(competences, expected, atol=0.01)


def test_select_two_classes():
    des_p_test = DESP()
    des_p_test.n_classes_ = 2
    expected = np.array([[True, False, True],
                         [True, False, True],
                         [False, True, False]])

    competences = np.array([[0.51, 0.0, 0.51],
                            [0.51, 0.0, 0.51],
                            [0.49, 1.0, 0.49]])

    selected = des_p_test.select(competences)

    assert np.array_equal(selected, expected)


# In this example, since the number of classes is 3, the competence level
# expected to be selected is > 0.33
def test_select_three_classes():
    des_p_test = DESP()
    des_p_test.n_classes_ = 3
    expected = np.array([[True, False, True],
                         [True, False, True],
                         [False, True, False]])

    competences = np.array([[0.34, 0.32, 1.0],
                            [0.50, 0.30, 1.01],
                            [0.25, 1.0, 0.25]])

    selected = des_p_test.select(competences)

    assert np.array_equal(selected, expected)


def test_select_none_competent():
    n_classifiers = 3
    des_p_test = DESP()
    des_p_test.n_classes_ = 2
    competences = np.ones(n_classifiers) * 0.49
    indices = des_p_test.select(competences)
    expected = np.array([[True, True, True]])
    assert np.array_equal(expected, indices)


# Test if the class is raising an error when the base classifiers do not
# implements the predict_proba method. In this case the test should not raise
# an error since this class does not require base classifiers that
# can estimate probabilities
def test_predict_proba(create_X_y):
    X, y = create_X_y
    clf1 = Perceptron()
    clf1.fit(X, y)
    DESP([clf1, clf1]).fit(X, y)
