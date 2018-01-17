import pytest
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB

from deslib.des.meta_des import METADES
from deslib.tests.examples_test import *


def test_meta_classifier_not_predict_proba():
    with pytest.raises(ValueError):
        METADES(create_pool_classifiers(), Perceptron())


def test_compute_meta_features():
    query = np.atleast_2d([1, 2])
    pool = create_pool_classifiers()
    meta_test = METADES(pool)

    meta_test.processed_dsel = dsel_processed_ex1
    meta_test.dsel_scores = dsel_scores_ex1
    meta_test.DSEL_target = y_dsel_ex1
    meta_test.n_classes = 3

    neighbors = neighbors_ex1[0, :]
    neighbors_op = neighbors_ex1[2, 0:meta_test.Kp]

# Expected values for each meta feature based on the data of ex1.
    expected_f1 = [1, 0, 1, 1, 1, 0, 0]
    expected_f2 = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    expected_f3 = [4.0/7.0]
    expected_f4 = [0, 1, 1, 1, 0]
    expected_f5 = [0.5]

    meta_features = meta_test.compute_meta_features(query, neighbors, neighbors_op, pool[0], 0)
    expected = expected_f1 + expected_f2 + expected_f3 + expected_f4 + expected_f5
    assert meta_features == expected


def test_estimate_competence():

    query = np.atleast_2d([1, 1])
    meta_test = METADES(create_pool_classifiers())

    # Set the state of the system which is set by the fit method.
    meta_test.processed_dsel = dsel_processed_ex1
    meta_test.dsel_scores = dsel_scores_ex1
    meta_test.DSEL_target = y_dsel_ex1
    meta_test.n_classes = 3

    meta_test.meta_classifier = GaussianNB()
    meta_test.neighbors = neighbors_ex1
    meta_test.distances = distances_ex1

    meta_test._get_similar_out_profiles = MagicMock(return_value=[0, neighbors_ex1[2, 0:meta_test.Kp]])

    meta_test.meta_classifier.predict_proba = MagicMock(return_value=np.array([[0.0, 0.8]]))
    meta_test.DFP_mask = np.array([1, 0, 1])

    competences = meta_test.estimate_competence(query)
    assert np.allclose(competences, [0.8, 0.0, 0.8])


def test_select():
    meta_test = METADES(create_pool_classifiers())
    competences = np.array([0.8, 0.6, 0.7, 0.2, 0.3, 0.4, 0.6, 0.1, 1.0, 0.98])
    indices = meta_test.select(competences)
    assert set(indices) == {0, 1, 2, 6, 8, 9}


def test_select_no_competent_classifiers():
    meta_test = METADES(create_pool_classifiers())
    competences = np.zeros(meta_test.n_classifiers)
    indices = meta_test.select(competences)
    assert indices == list(range(meta_test.n_classifiers))


def test_meta_classifier_is_none():
    with pytest.warns(Warning):
        METADES(create_pool_classifiers(), meta_classifier=None)


def test_sample_selection():
    meta_test = METADES(create_pool_all_agree(0, 10) + create_pool_all_agree(1, 5))
    meta_test.processed_dsel = dsel_processed_ex1
    value = meta_test._sample_selection_agreement(0)
    assert value == 0.2


@pytest.mark.parametrize('Hc', ['a', None, 0.2, -1])
def test_parameter_Hc(Hc):
    with pytest.raises((ValueError, TypeError)):
        METADES(create_pool_classifiers(), Hc=Hc)


@pytest.mark.parametrize('gamma', ['a', None, 0, -1, 0.45])
def test_parameter_gamma(gamma):
    with pytest.raises((ValueError, TypeError)):
        METADES(create_pool_classifiers(), gamma=gamma)
