import numpy as np
import pytest

from deslib.util.prob_functions import (ccprmod,
                                        log_func,
                                        min_difference,
                                        softmax,
                                        exponential_func,
                                        entropy_func)


# Example from the original paper "A probabilistic model of classifier
# competence for dynamic ensemble selection"
@pytest.mark.parametrize("supports, idx_correct_label, expected",
                         [([[0.3, 0.6, 0.1], [1.0 / 3, 1.0 / 3, 1.0 / 3]],
                           [1, 0], [0.784953394056843, 0.332872292262951]),
                          ([[0.5, 0.2, 0.3], [0.5, 0.2, 0.3]], [0, 1],
                           [0.6428, 0.1194])])
def test_ccprmod_return_value(supports, idx_correct_label, expected):
        value = ccprmod(supports, idx_correct_label)
        assert np.isclose(value, expected, atol=0.001).all()


@pytest.mark.parametrize('B', [0, -1, None, 0.55])
def test_valid_ccprmod_beta(B):
    supports = [0.3, 0.6, 0.1]
    idx_correct_label = [1]

    with pytest.raises((ValueError, TypeError)):
        ccprmod(supports, idx_correct_label, B)


def test_ccprmod_zero_support():
    supports = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.8, 0.2]]
    idx_correct_label = [0, 2, 0]
    assert np.isclose(ccprmod(supports, idx_correct_label), 0, atol=0.01).all()


def test_ccprmod_one_support():
    supports = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    idx_correct_label = [2, 1]
    assert np.isclose(ccprmod(supports, idx_correct_label), 1, atol=0.01).all()


def test_softmax_sum_to_one():
    test = np.random.rand(10)
    assert np.allclose(np.sum(softmax(test)), 1.0, atol=0.001)


@pytest.mark.parametrize('vector, expected',
                         [([0, 1, -0.5, 0.5],
                           [0.1674, 0.4551, 0.1015, 0.2760]),
                          ([3.20, 5, 7.856, 9.65],
                           [0.0013, 0.0081, 0.1412, 0.8493])])
def test_softmax(vector, expected):
    assert np.isclose(softmax(vector), expected, atol=0.001).all()


@pytest.mark.parametrize('supports_correct, expected',
                         [(np.array([0.5]), 0),
                          (np.array([0.0]), -1.0),
                          (np.array([1.0]), 1.0)])
def test_log_func_two_classes(supports_correct, expected):
    n_classes = 2
    result = log_func(n_classes, supports_correct)
    assert np.isclose(result, expected)


@pytest.mark.parametrize('supports_correct, expected',
                         [(np.array([0.33]), 0),
                          (np.array([0.0]), -1.0),
                          (np.array([1.0]), 1.0)])
def test_log_func_multi_class(supports_correct, expected):
    n_classes = 3
    result = log_func(n_classes, supports_correct)
    assert np.isclose(result, expected, atol=0.01)


# Example from the paper "A probabilistic model of classifier competence for
# dynamic ensemble selection"
@pytest.mark.parametrize("supports, idx_correct_label, expected",
                         [(np.array([[0.3, 0.6, 0.1], [0.33, 0.33, 0.33]]),
                          [1, 0], [0.3, 0.0]),
                          (np.array([[0.5, 0.2, 0.3], [0.5, 0.2, 0.3]]),
                          [0, 1], [0.2, -0.3])])
def test_min_difference(supports, idx_correct_label, expected):
    result = min_difference(supports, idx_correct_label)
    assert np.isclose(result, expected, atol=0.01).all()


@pytest.mark.parametrize('supports_correct, expected',
                         [(np.array([0.33]), -0.01),
                          (np.array([0.0]), -1.0),
                          (np.array([1.0]), 1.0)])
def test_exponential_func_multi_class(supports_correct, expected):
    n_classes = 3
    result = exponential_func(n_classes, supports_correct)
    assert np.isclose(result, expected, atol=0.01).all()


def test_exponential_func_multi_class_batch():
    supports_correct = np.array([0.33, 0.0, 1.0])
    expected = [-0.01, -1.0, 1.0]
    n_classes = 3
    result = exponential_func(n_classes, supports_correct)
    assert np.allclose(result, expected, atol=0.01)


@pytest.mark.parametrize('supports_correct, expected',
                         [(np.array([0.5]), 0.00),
                          (np.array([0.0]), -1.0),
                          (np.array([1.0]), 1.0)])
def test_exponential_func_two_class(supports_correct, expected):
    n_classes = 2
    result = exponential_func(n_classes, supports_correct)
    assert np.isclose(result, expected, atol=0.01).all()


def test_exponential_func():
    n_classes = 2
    result1 = exponential_func(n_classes, np.array([0.2]))
    assert result1 < 0.0

    result2 = exponential_func(n_classes, np.array([0.8]))
    assert result2 > 0.0

    assert result2 > result1

    result3 = exponential_func(n_classes, np.array([1.0]))
    result4 = exponential_func(n_classes, np.array([0.9]))
    assert result3 > result4 > result2 > result1


def test_entropy_func_three_classes():
    n_classes = 3
    supports = np.array([[0.33, 0.33, 0.33], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    is_correct = np.array([0, 1, 0])
    expected = [0.0, 1.0, -1.0]
    result = entropy_func(n_classes, supports, is_correct)
    assert np.isclose(result, expected, atol=0.01).all()


def test_entropy_func_parameter_shape():
    with pytest.raises(ValueError):
        entropy_func(2, np.array([0, 1]), np.array([0]))
