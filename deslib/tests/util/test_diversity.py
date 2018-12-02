import sys

import numpy as np
import pytest

from deslib.util.diversity import (_process_predictions,
                                   double_fault,
                                   Q_statistic,
                                   ratio_errors,
                                   agreement_measure,
                                   disagreement_measure,
                                   correlation_coefficient)


@pytest.fixture
def example_diversity(create_X_y):

    y_pred_classifier1 = np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    y_pred_classifier2 = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1])

    y_real = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 1])

    y_ex1 = create_X_y[1]
    return y_pred_classifier1, y_pred_classifier2, y_real, y_ex1


@pytest.fixture
def example_diversity_ones_zeros(create_X_y):

    y = create_X_y[1]
    y_pred_ones = np.ones(15)
    y_pred_zeros = np.zeros(15)
    return y, y_pred_ones, y_pred_zeros


def test_process_predictions_ones_zeros(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    N00, N10, N01, N11 = _process_predictions(y, y_pred_ones, y_pred_zeros)
    assert N00 == 0.0 and N11 == 0.0 and N01 == 9.0/15.0 and N10 == 6.0/15.0


def test_process_predictions_zeros_ones(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    N00, N10, N01, N11 = _process_predictions(y, y_pred_zeros, y_pred_ones)
    assert N00 == 0.0 and N11 == 0.0 and N01 == 6.0/15.0 and N10 == 9.0/15.0


def test_process_predictions_zeros(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    N00, N10, N01, N11 = _process_predictions(y, y_pred_zeros, y_pred_zeros)
    assert N00 == 6.0/15.0 and N11 == 9.0/15.0 and N01 == 0.0 and N10 == 0.0


def test_process_predictions_ones(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    N00, N10, N01, N11 = _process_predictions(y, y_pred_ones, y_pred_ones)
    assert N00 == 9.0/15.0 and N11 == 6.0/15.0 and N01 == 0.0 and N10 == 0.0


def test_process_predictions_diff_sizes(example_diversity_ones_zeros):
    y_pred1 = np.ones(10)
    y_pred2 = np.ones(15)
    y, _, _ = example_diversity_ones_zeros

    with pytest.raises(ValueError):
        _ = _process_predictions(y, y_pred1, y_pred2)


def test_double_fault_ones_zeros(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    df = double_fault(y, y_pred_ones, y_pred_zeros)
    assert df == 0.0


def test_double_fault_order(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    df1 = double_fault(y, y_pred_ones, y_pred_zeros)
    df2 = double_fault(y, y_pred_zeros, y_pred_ones)
    assert df1 == df2


def test_double_fault_zeros(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    df = double_fault(y, y_pred_zeros, y_pred_zeros)
    assert df == 6.0/15.0


def test_double_fault_ones(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    df = double_fault(y, y_pred_ones, y_pred_ones)
    assert df == 9.0/15.0


def test_double_fault():
    labels = np.array([0, 0, 0, 0, 1, 1, 1])
    pred1 = np.array([1, 0, 1, 0, 0, 0, 0])
    pred2 = np.array([1, 0, 0, 0, 1, 0, 0])

    actual = double_fault(labels, pred1, pred2)

    assert actual == 3. / 7  # three common errors out of 7 predictions


def test_q_statistic_ones_zeros(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    Q = Q_statistic(y, y_pred_ones, y_pred_zeros)
    assert Q == -1.0


def test_q_statistic_ones(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    Q = Q_statistic(y, y_pred_ones, y_pred_ones)
    assert Q == 1.0


def test_q_statistic_zeros(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    Q = Q_statistic(y, y_pred_zeros, y_pred_zeros)
    assert Q == 1.0


def test_ratio_errors_zeros(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    ratio = ratio_errors(y, y_pred_zeros, y_pred_zeros)
    assert ratio == 0.0


def test_ratio_errors_ones(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    ratio = ratio_errors(y, y_pred_ones, y_pred_ones)
    assert ratio == 0.0


def test_ratio_ones_zeros(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    ratio = ratio_errors(y, y_pred_ones, y_pred_zeros)
    assert ratio == sys.float_info.max


def test_ratio_order(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    ratio1 = ratio_errors(y, y_pred_ones, y_pred_zeros)
    ratio2 = ratio_errors(y, y_pred_zeros, y_pred_ones)
    assert ratio1 == ratio2


def test_ratio_errors_diff_classifiers(example_diversity):
    y_pred_classifier1, y_pred_classifier2, y_real, y_ex1 = example_diversity
    ratio = ratio_errors(y_real, y_pred_classifier1, y_pred_classifier2)
    assert np.isclose(ratio, 1.66, atol=0.01)


def test_agreement(example_diversity):
    y_pred_classifier1, y_pred_classifier2, y_real, y_ex1 = example_diversity
    agreement = agreement_measure(y_real,
                                  y_pred_classifier1,
                                  y_pred_classifier2)
    assert np.isclose(agreement, 0.5)


def test_disagreement(example_diversity):
    y_pred_classifier1, y_pred_classifier2, y_real, y_ex1 = example_diversity
    disagreement = disagreement_measure(y_real,
                                        y_pred_classifier1,
                                        y_pred_classifier2)
    assert np.isclose(disagreement, 0.5)


def test_coefficient_correlation(example_diversity):
    y_pred_classifier1, y_pred_classifier2, y_real, y_ex1 = example_diversity
    coefficient = correlation_coefficient(y_real,
                                          y_pred_classifier1,
                                          y_pred_classifier2)
    assert np.isclose(coefficient, 0.0)
