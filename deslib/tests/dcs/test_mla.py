import pytest
import numpy as np
from sklearn.linear_model import Perceptron

from deslib.dcs.mla import MLA


from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(MLA)


# Should always be 1.0 since the supports for the correct class is always 1.
@pytest.mark.parametrize('index', [0, 1, 2])
def test_estimate_competence_all_ones(index, example_all_ones):
    _, y, neighbors, distances, dsel_processed, dsel_scores = example_all_ones
    query = np.atleast_2d([1, 1])

    mla_test = MLA()
    mla_test.n_classifiers_ = 3

    mla_test.DSEL_processed_ = dsel_processed
    mla_test.DSEL_scores = dsel_scores
    mla_test.DSEL_target_ = y
    mla_test.n_classes_ = 2

    neighbors = neighbors[index, :].reshape(1, -1)
    distances = distances[index, :].reshape(1, -1)

    expected = [1.0, 1.0, 1.0]

    predictions = np.array([[0, 1, 0]])

    competences = mla_test.estimate_competence(query,
                                               neighbors,
                                               distances=distances,
                                               predictions=predictions)

    assert np.isclose(competences, expected).all()


def test_estimate_competence_batch(example_estimate_competence):

    _, y, neighbors, _, dsel_processed, _ = example_estimate_competence

    query = np.array([[1, 1], [1, 1], [1, 1]])
    expected = np.array([[0.750,  0.666,  0.750],
                         [0.800,  1.000,  0.800],
                         [1.000,  0.600,  0.500]])

    mla_test = MLA()
    mla_test.n_classifiers_ = 3
    mla_test.DSEL_processed_ = dsel_processed
    distances = np.ones((3, 7))

    mla_test.DSEL_target_ = y
    mla_test.n_classes_ = 2
    predictions = np.array([[0, 1, 0]])

    competences = mla_test.estimate_competence(query,
                                               neighbors=neighbors,
                                               distances=distances,
                                               predictions=predictions)

    assert np.allclose(competences, expected, atol=0.01)


# in this test case, the target of the neighbors is always different than the
# predicted. So the estimation of competence should always be zero
@pytest.mark.parametrize('index', [0, 1, 2])
def test_estimate_competence_diff_target(index, example_estimate_competence):
    _, _, neighbors, distances, dsel_processed, _ = example_estimate_competence

    query = np.atleast_2d([1, 1])

    mla_test = MLA()
    mla_test.n_classifiers_ = 3

    mla_test.DSEL_processed_ = dsel_processed
    mla_test.DSEL_target_ = np.ones(15, dtype=int) * 3

    neighbors = neighbors[index, :].reshape(1, -1)
    distances = distances[index, :].reshape(1, -1)

    expected = [0.0, 0.0, 0.0]

    predictions = np.array([[0, 1, 0]])

    competences = mla_test.estimate_competence(query,
                                               neighbors,
                                               distances=distances,
                                               predictions=predictions)

    assert np.isclose(competences, expected).all()


# Testing example from kuncheva's book (combining pattern classifiers)
def test_estimate_competence_kuncheva_ex(example_kuncheva):
    query = np.atleast_2d([1, 1])
    example_kuncheva = example_kuncheva

    mla_test = MLA(k=example_kuncheva['k'])
    mla_test.n_classifiers_ = 2

    mla_test.DSEL_processed_ = np.repeat(example_kuncheva['dsel_processed'],
                                         2,
                                         axis=1)

    mla_test.dsel_scores_ = example_kuncheva['dsel_scores']
    mla_test.DSEL_target_ = example_kuncheva['y_dependent']
    mla_test.n_classes_ = example_kuncheva['n_classes']

    neighbors = example_kuncheva['neighbors'].reshape(1, -1)
    distances = example_kuncheva['distances'].reshape(1, -1)

    predictions = np.array([[1, 1]])
    competences = mla_test.estimate_competence(query,
                                               neighbors,
                                               distances=distances,
                                               predictions=predictions)

    assert np.allclose(competences, [0.95, 0.95], atol=0.01)


# Test if the class is raising an error when the base classifiers do not
# implements the predict_proba method. In this case the test should not raise
# an error since this class does not require base classifiers that
# can estimate probabilities
def test_predict_proba(create_X_y):
    X, y = create_X_y

    clf1 = Perceptron()
    clf1.fit(X, y)
    MLA([clf1, clf1]).fit(X, y)
