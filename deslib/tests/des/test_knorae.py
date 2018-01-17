import pytest

from deslib.des.knora_e import KNORAE
from deslib.tests.examples_test import *


@pytest.mark.parametrize('index, expected', [(0, [1.0, 0.0, 1.0]),
                                             (1, [2.0, 0.0, 2.0]),
                                             (2, [0.0, 3.0, 0.0])])
def test_estimate_competence(index, expected):
    query = np.atleast_2d([1, 1])

    knora_e_test = KNORAE(create_pool_classifiers())
    knora_e_test.fit(X_dsel_ex1, y_dsel_ex1)
    knora_e_test.DFP_mask = np.ones(knora_e_test .n_classifiers)
    knora_e_test.neighbors = neighbors_ex1[index, :]
    knora_e_test.distances = distances_ex1[index, :]
    competences = knora_e_test.estimate_competence(query)
    assert np.isclose(competences, expected, atol=0.01).all()


@pytest.mark.parametrize('index, expected', [(0, [0, 2]),
                                             (1, [0, 2]),
                                             (2, [1])])
def test_select(index, expected):
    query = np.atleast_2d([1, 1])

    knora_e_test = KNORAE(create_pool_classifiers())
    knora_e_test.fit(X_dsel_ex1, y_dsel_ex1)
    knora_e_test.DFP_mask = np.ones(knora_e_test .n_classifiers)
    knora_e_test.neighbors = neighbors_ex1[index, :]
    knora_e_test.distances = distances_ex1[index, :]
    competences = knora_e_test.estimate_competence(query)
    selected = knora_e_test.select(competences)

    assert selected == expected


# No classifier here is selected, since the always predict class 2 where there are only samples labeled as class 0 and 1
# in the region of competence
def test_select_none_competent():
    query = np.atleast_2d([1, 1])

    knora_e_test = KNORAE(create_pool_all_agree(2, 100))
    knora_e_test.fit(X_dsel_ex1, y_dsel_ex1)

    knora_e_test.neighbors = neighbors_ex1[0, :]
    knora_e_test.distances = distances_ex1[0, :]
    knora_e_test.DFP_mask = np.ones(knora_e_test.n_classifiers)

    competences = knora_e_test.estimate_competence(query)
    indices = knora_e_test.select(competences)

    assert indices == list(range(knora_e_test.n_classifiers))


