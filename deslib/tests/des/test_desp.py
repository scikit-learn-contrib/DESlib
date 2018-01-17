import pytest

from deslib.des.des_p import DESP
from deslib.tests.examples_test import *


@pytest.mark.parametrize('index, expected', [(0, [0.57142857, 0.4285714, 0.57142857]),
                                             (1, [0.71428571, 0.2857142, 0.71428571]),
                                             (2, [0.2857142, 0.71428571, 0.2857142])])
def test_estimate_competence(index, expected):
    query = np.atleast_2d([1, 1])

    des_p_test = DESP(create_pool_classifiers())
    des_p_test.fit(X_dsel_ex1, y_dsel_ex1)
    des_p_test.DFP_mask = np.ones(des_p_test.n_classifiers)
    des_p_test.neighbors = neighbors_ex1[index, :]
    des_p_test.distances = distances_ex1[index, :]
    competences = des_p_test.estimate_competence(query)
    assert np.isclose(competences, expected, atol=0.01).all()


@pytest.mark.parametrize('index, expected', [(0, [0, 2]),
                                             (1, [0, 2]),
                                             (2, [1])])
def test_select_two_classes(index, expected):
    query = np.atleast_2d([1, 1])

    des_p_test = DESP(create_pool_classifiers())
    des_p_test.fit(X_dsel_ex1, y_dsel_ex1)

    des_p_test.DFP_mask = np.ones(des_p_test.n_classifiers)
    des_p_test.neighbors = neighbors_ex1[index, :]
    des_p_test.distances = distances_ex1[index, :]

    competences = des_p_test.estimate_competence(query)
    selected = des_p_test.select(competences)

    assert selected == expected


# In this example, since the number of classes is 3, the competence level expected to be selected is > 0.33
@pytest.mark.parametrize('index, expected', [(0, [0, 1, 2]),
                                             (1, [0, 2]),
                                             (2, [1])])
def test_select_three_classes(index, expected):
    query = np.atleast_2d([1, 1])

    des_p_test = DESP(create_pool_classifiers())
    des_p_test.fit(X_dsel_ex1, y_dsel_ex1)

    des_p_test.n_classes = 3
    des_p_test.DFP_mask = np.ones(des_p_test.n_classifiers)
    des_p_test.neighbors = neighbors_ex1[index, :]
    des_p_test.distances = distances_ex1[index, :]

    competences = des_p_test.estimate_competence(query)
    selected = des_p_test.select(competences)

    assert selected == expected


# In this example, since the number of classes is 3, the competence level expected to be selected is > 0.1. All base
# Classifiers should be selected in this example
@pytest.mark.parametrize('index', [0, 1, 2])
def test_select_ten_classes(index,):
    query = np.atleast_2d([1, 1])

    des_p_test = DESP(create_pool_classifiers())
    des_p_test.fit(X_dsel_ex1, y_dsel_ex1)

    des_p_test.n_classes = 10
    des_p_test.DFP_mask = np.ones(des_p_test.n_classifiers)
    des_p_test.neighbors = neighbors_ex1[index, :]
    des_p_test.distances = distances_ex1[index, :]

    competences = des_p_test.estimate_competence(query)
    selected = des_p_test.select(competences)

    assert selected == list(range(des_p_test.n_classifiers))


def test_select_none_competent():
    des_p_test = DESP(create_pool_classifiers())
    des_p_test.n_classes = 2
    competences = np.ones(des_p_test.n_classifiers) * 0.49
    indices = des_p_test.select(competences)
    assert indices == list(range(des_p_test.n_classifiers))