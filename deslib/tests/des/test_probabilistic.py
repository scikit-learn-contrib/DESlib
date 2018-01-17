from deslib.des.probabilistic import Probabilistic, RRC, DESKL, Logarithmic, Exponential, MinimumDifference
from deslib.tests.examples_test import *


# Being all zeros, no base classifier is deemed competent, so the system selects all of them
def test_select_all_ones():
    competences = np.ones(100)
    probabilistic_test = Probabilistic(create_pool_all_agree(1, 100))
    probabilistic_test.n_classes = 2
    indices = probabilistic_test.select(competences)
    assert indices == list(range(100))


# Being all zeros, no base classifier is deemed competent, so the system selects all of them
def test_select_all_zeros():
    competences = np.zeros(100)
    probabilistic_test = Probabilistic(create_pool_all_agree(1, 100))
    probabilistic_test.n_classes = 2
    indices = probabilistic_test.select(competences)
    assert indices == list(range(100))


# Being all zeros, no base classifier is deemed competent, so the system selects all of them
def test_select_random_classifier():
    competences = np.random.rand(100)
    expected = np.where(competences > 0.25)[0]
    probabilistic_test = Probabilistic(create_pool_all_agree(1, 100))
    probabilistic_test.n_classes = 4
    indices = probabilistic_test.select(competences)
    assert np.array_equal(indices, expected)


# Being all zeros, no base classifier is deemed competent, so the system selects all of them
def test_select_threshold():
    competences = np.random.rand(100)
    expected = np.where(competences > 0.5)[0]

    probabilistic_test = Probabilistic(create_pool_all_agree(1, 100))
    probabilistic_test.selection_threshold = 0.5
    indices = probabilistic_test.select(competences)
    assert np.array_equal(indices, expected)


def test_potential_function_zeros():
    dists = np.zeros(10)
    value = Probabilistic.potential_func(dists)
    assert np.array_equal(value, np.ones(10))


def test_potential_function():
    dists = np.array([1.0, 0.5, 2, 0.33])
    value = Probabilistic.potential_func(dists)
    assert np.allclose(value, [0.3679, 0.7788, 0.0183, 0.8968], atol=0.001)


def test_estimate_competence():

    query = np.atleast_2d([1, 1])
    probabilistic_test = Probabilistic(create_pool_classifiers())
    probabilistic_test.distances = [0.5, 1.0, 2.0]
    probabilistic_test.neighbors = [0, 1, 2]
    probabilistic_test.DFP_mask = np.ones(probabilistic_test.n_classifiers)

    probabilistic_test.C_src = np.array([[0.5, 0.2, 0.8],
                                         [1.0, 1.0, 1.0],
                                         [1.0, 0.6, 0.3]])

    competence = probabilistic_test.estimate_competence(query)
    assert np.allclose(competence, [0.665, 0.458, 0.855], atol=0.01)


# Test the estimate competence function when the competence source is equal to zero. The competence should also be zero.
def test_estimate_competence_zeros():
    query = np.atleast_2d([1, 1])
    probabilistic_test = Probabilistic(create_pool_classifiers())
    probabilistic_test.distances = distances_ex1[0, 0:3]
    probabilistic_test.neighbors = [0, 2, 1]
    probabilistic_test.DFP_mask = np.zeros(probabilistic_test.n_classifiers)
    probabilistic_test.C_src = np.zeros((3, 3))
    competence = probabilistic_test.estimate_competence(query)
    assert np.sum(competence) == 0.0


# Test the estimate competence function when the competence source is equal to one. The competence should also be ones.
def test_estimate_competence_ones():
    query = np.atleast_2d([1, 1])
    probabilistic_test = Probabilistic(create_pool_classifiers())
    probabilistic_test.distances = distances_ex1[0, 0:3]
    probabilistic_test.neighbors = [0, 2, 1]
    probabilistic_test.DFP_mask = np.ones(probabilistic_test.n_classifiers)
    probabilistic_test.C_src = np.ones((3, 3))
    competence = probabilistic_test.estimate_competence(query)
    assert (competence == 1.0).all()


""" Test the source_competence using the rrc method. Here we consider the same values from the example
applied in the test_prob_functions.py to assert if the source_competence function call the ccprmod correctly
and fill the competence source (C_src) with the correct results.

The scores used are: [[0.3, 0.6, 0.1], [1.0 / 3, 1.0 / 3, 1.0 / 3], [0.5, 0.2, 0.3], [0.5, 0.2, 0.3]]
The correct labels are: [1, 0, 0, 1]
The expected value should be: an np.array (4,1) with the values = [[0.7849], [0.3328], [0.6428], [0.1194]]
"""


def test_source_competence_rrc():
    rrc_test = RRC([create_base_classifier(return_value=1, return_prob=1.0)])
    rrc_test.dsel_scores = np.array([[0.3, 0.6, 0.1], [1.0 / 3, 1.0 / 3, 1.0 / 3], [0.5, 0.2, 0.3], [0.5, 0.2, 0.3]])
    rrc_test.DSEL_target = [1, 0, 0, 1]
    rrc_test.n_classes = 3
    rrc_test.n_samples = 4
    C_src = rrc_test.source_competence()
    expected = np.array([[0.7849], [0.3328], [0.6428], [0.1194]])
    assert np.allclose(C_src, expected, atol=0.01)


""" Test the source_competence using the kullback leibler divergence method. Here we consider the same values 
applied in the test_prob_functions.py to assert if the source_competence function call the ccprmod correctly
and fill the competence source (C_src) with the correct results.

The scores used are: [[0.3, 0.6, 0.1], [1.0 / 3, 1.0 / 3, 1.0 / 3], [0.5, 0.2, 0.3], [0.5, 0.2, 0.3]]
The correct labels are: [1, 0, 0, 1]
The expected value should be: an np.array (4,1) with the values = [[0.7849], [0.3328], [0.6428], [0.1194]]
"""


def test_source_competence_kl():
    entropy_test = DESKL([create_base_classifier(return_value=1, return_prob=1.0)])
    entropy_test.dsel_scores = np.array([[0.33, 0.33, 0.33], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    entropy_test.processed_dsel = np.array([[False], [True], [False]])
    entropy_test.n_classes = 3
    entropy_test.n_samples = 3
    C_src = entropy_test.source_competence()
    expected = np.array([[0.0], [1.0], [-1.0]])
    assert np.allclose(C_src, expected, atol=0.01)


""" Test the source_competence using the kullback leibler divergence method. Here we consider the same values 
applied in the test_prob_functions.py to assert if the source_competence function call the ccprmod correctly
and fill the competence source (C_src) with the correct results.

The scores used are: [[0.3, 0.6, 0.1], [1.0 / 3, 1.0 / 3, 1.0 / 3], [0.5, 0.2, 0.3], [0.5, 0.2, 0.3]]
The correct labels are: [1, 0, 0, 1]
The expected value should be: an np.array (4,1) with the values = [[0.7849], [0.3328], [0.6428], [0.1194]]
"""


def test_source_competence_minimum_difference():
    md_test = MinimumDifference([create_base_classifier(return_value=1, return_prob=1.0)])
    md_test .dsel_scores = np.array([[0.3, 0.6, 0.1], [1.0 / 3, 1.0 / 3, 1.0 / 3], [0.5, 0.2, 0.3], [0.5, 0.2, 0.3]])
    md_test .DSEL_target = [1, 0, 0, 1]
    md_test .n_classes = 3
    md_test .n_samples = 4
    C_src = md_test .source_competence()
    expected = np.array([[0.3], [0.0], [0.2], [-0.3]])
    assert np.allclose(C_src, expected, atol=0.01)


""" Test the source_competence using the logarithmic method. Here we consider the same values 
applied in the test_prob_functions.py to assert if the source_competence function call the ccprmod correctly
and fill the competence source (C_src) with the correct results.

The scores used are: [[0.67, 0.33, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
The correct labels are: [1, 1, 1], so the supports for the correct class are: [0.33, 0.0, 1.0]
The expected value should be: an np.array (3,1) with the values = [[0.0], [-1.0], [1.0]]]
"""


def test_source_competence_logarithmic():
    log_test = Logarithmic([create_base_classifier(return_value=1, return_prob=1.0)])
    log_test.dsel_scores = np.array([[0.67, 0.33, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    log_test.DSEL_target = [1, 1, 1]
    log_test.n_classes = 3
    log_test.n_samples = 3
    C_src = log_test.source_competence()
    expected = np.array([[0.0], [-1.0], [1.0]])
    assert np.allclose(C_src, expected, atol=0.01)


""" Test the source_competence using the exponential method. Here we consider the same values 
applied in the test_prob_functions.py to assert if the source_competence function call the ccprmod correctly
and fill the competence source (C_src) with the correct results.

Only two classes are considered in this example.
The scores used are: [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]].
The correct labels are: [1, 1, 1], so the supports for the correct class are: [0.5, 0.0, 1.0].
The expected value should be: an np.array (3,1) with the values = [[0.0], [-1.0], [1.0]]].
"""


def test_source_competence_exponential():
    exp_test = Exponential([create_base_classifier(return_value=1, return_prob=1.0)])
    exp_test.dsel_scores = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    exp_test.DSEL_target = [1, 1, 1]
    exp_test.n_classes = 2
    exp_test.n_samples = 3
    C_src = exp_test.source_competence()
    expected = np.array([[0.0], [-1.0], [1.0]])
    assert np.allclose(C_src, expected, atol=0.01)

