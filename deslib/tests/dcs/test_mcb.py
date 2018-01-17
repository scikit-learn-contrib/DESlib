import pytest

from deslib.dcs.mcb import MCB
from deslib.tests.examples_test import *

# ex1 the similarity will always be 100%
bks_dsel_ex1 = np.hstack((np.hstack((np.zeros((15, 1)), np.ones((15, 1)))), np.zeros((15, 1))))

# Change a bit to check if the filtering by similarity is working as intended.
bks_dsel_ex2 = np.hstack((np.hstack((np.zeros((15, 1)), np.ones((15, 1)))), np.zeros((15, 1))))
bks_dsel_ex2[1, :] = 2

bks_dsel_ex3 = bks_dsel_ex1 + 1


@pytest.mark.parametrize('similarity_threshold', [2.0, -1.0, -0.5])
def test_similarity_threshold(similarity_threshold):

    with pytest.raises(ValueError):
        MCB(create_pool_classifiers(), similarity_threshold=similarity_threshold)


@pytest.mark.parametrize('similarity_threshold', [None, 'a'])
def test_similarity_threshold_type(similarity_threshold):

    with pytest.raises(TypeError):
        MCB(create_pool_classifiers(), similarity_threshold=similarity_threshold)


def test_bks_transform():
    query = np.array([0, -1])

    pool_classifiers = create_pool_classifiers()
    mcb_test = MCB(pool_classifiers)
    value = mcb_test._BKS_transform(query.reshape(1, -1))
    expected = np.array([0, 1, 0])
    assert np.equal(value, expected).all()

    pool_classifiers2 = create_pool_all_agree(1, 10)
    mcb_test2 = MCB(pool_classifiers2)
    bks_result = mcb_test2._BKS_transform(query)
    expected = np.array([1] * 10)
    assert np.equal(bks_result, expected).all()


@pytest.mark.parametrize('index, expected', [(0, [0.57142857,  0.71428571,  0.71428571]),
                                             (1, [0.71428571,  0.85714286,  0.71428571]),
                                             (2, [0.57142857,  0.71428571,  0.57142857])])
def test_estimate_competence(index, expected):
    query = np.array([1, 1])

    mcb_test = MCB(create_pool_classifiers())
    mcb_test.processed_dsel = dsel_processed_ex1
    mcb_test.neighbors = neighbors_ex1[index, :]
    mcb_test.distances = distances_ex1[index, :]
    mcb_test.DFP_mask = [1, 1, 1]
    mcb_test.BKS_dsel = bks_dsel_ex1
    competences = mcb_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, expected).all()

# This second test case uses a different KS matrix to filter out some neighbors.


@pytest.mark.parametrize('index, expected', [(0, [0.66666666,  0.83333333,  0.66666666]),
                                             (1, [0.83333333,  1.0,  0.66666666])])
def test_estimate_competence2(index, expected):
    query = np.array([1, 1])

    mcb_test = MCB(create_pool_classifiers())
    mcb_test.processed_dsel = dsel_processed_ex1
    mcb_test.neighbors = neighbors_ex1[index, :]
    mcb_test.distances = distances_ex1[index, :]
    mcb_test.DFP_mask = [1, 1, 1]
    # Only changing the pre-processed BKS to see if the filter works.
    mcb_test.BKS_dsel = bks_dsel_ex2
    competences = mcb_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, expected).all()


# This third test uses an totally wrong bks matrix, so that the technique is obligated to use the whole
#  region of competence

@pytest.mark.parametrize('index, expected', [(0, [0.57142857,  0.71428571,  0.71428571]),
                                             (1, [0.71428571,  0.85714286,  0.71428571]),
                                             (2, [0.57142857,  0.71428571,  0.57142857])])
def test_estimate_competence3(index, expected):
    query = np.array([1, 1])

    mcb_test = MCB(create_pool_classifiers())
    mcb_test.processed_dsel = dsel_processed_ex1
    mcb_test.neighbors = neighbors_ex1[index, :]
    mcb_test.distances = distances_ex1[index, :]
    mcb_test.DFP_mask = [1, 1, 1]
    # Only changing the pre-processed BKS to see if the filter works.
    mcb_test.BKS_dsel = bks_dsel_ex3
    competences = mcb_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, expected).all()