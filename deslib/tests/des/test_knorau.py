import pytest

from deslib.des.knora_u import KNORAU
from deslib.tests.examples_test import *


@pytest.mark.parametrize('index, expected', [(0, [4.0, 3.0, 4.0]),
                                             (1, [5.0, 2.0, 5.0]),
                                             (2, [2.0, 5.0, 2.0])])
def test_estimate_competence(index, expected):
    query = np.atleast_2d([1, 1])

    knora_u_test = KNORAU(create_pool_classifiers())
    knora_u_test.fit(X_dsel_ex1, y_dsel_ex1)
    knora_u_test.DFP_mask = np.ones(knora_u_test .n_classifiers)
    knora_u_test.neighbors = neighbors_ex1[index, :]
    knora_u_test.distances = distances_ex1[index, :]
    competences = knora_u_test.estimate_competence(query)
    assert np.isclose(competences, expected, atol=0.01).all()


@pytest.mark.parametrize('index, expected', [(0, 0),
                                             (1, 0),
                                             (2, 1)])
def test_classify(index, expected):
    query = np.atleast_2d([1, 1])

    knora_u_test = KNORAU(create_pool_classifiers())
    knora_u_test.fit(X_dsel_ex1, y_dsel_ex1)
    knora_u_test.DFP_mask = np.ones(knora_u_test .n_classifiers)
    knora_u_test.neighbors = neighbors_ex1[index, :]
    knora_u_test.distances = distances_ex1[index, :]
    prediction = knora_u_test.classify_instance(query)

    assert prediction == expected


@pytest.mark.parametrize('index, expected', [(0, 0),
                                             (1, 0),
                                             (2, 0)])
def test_classify2(index, expected):
    query = np.atleast_2d([1, 1])

    knora_u_test = KNORAU(create_pool_classifiers()+create_pool_all_agree(0, 1))
    knora_u_test.fit(X_dsel_ex1, y_dsel_ex1)
    knora_u_test.DFP_mask = np.ones(knora_u_test .n_classifiers)
    knora_u_test.neighbors = neighbors_ex1[index, :]
    knora_u_test.distances = distances_ex1[index, :]
    prediction = knora_u_test.classify_instance(query)

    assert prediction == expected


def test_weights_zero():
    query = np.atleast_2d([1, 1])

    knora_u_test = KNORAU(create_pool_classifiers())
    knora_u_test.estimate_competence = MagicMock(return_value=np.zeros(3))

    result = knora_u_test.select(query)
    assert np.array_equal(result, np.array([0, 1, 0]))




