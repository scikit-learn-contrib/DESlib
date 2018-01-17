import sys

import numpy as np
import pytest

from deslib.tests.examples_test import y_dsel_ex1
from deslib.util.diversity import _process_predictions, double_fault, Q_statistic, ratio_errors

y_pred_ones = np.ones(15)
y_pred_zeros = np.zeros(15)

y_pred_classifier1 = np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
y_pred_classifier2 = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1])

y_real = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 1])


def test_process_predictions_ones_zeros():
    N00, N10, N01, N11 = _process_predictions(y_dsel_ex1, y_pred_ones, y_pred_zeros)
    assert N00 == 0.0 and N11 == 0.0 and N01 == 9.0/15.0 and N10 == 6.0/15.0


def test_process_predictions_zeros_ones():
    N00, N10, N01, N11 = _process_predictions(y_dsel_ex1, y_pred_zeros, y_pred_ones)
    assert N00 == 0.0 and N11 == 0.0 and N01 == 6.0/15.0 and N10 == 9.0/15.0


def test_process_predictions_zeros():
    N00, N10, N01, N11 = _process_predictions(y_dsel_ex1, y_pred_zeros, y_pred_zeros)
    assert N00 == 6.0/15.0 and N11 == 9.0/15.0 and N01 == 0.0 and N10 == 0.0


def test_process_predictions_ones():
    N00, N10, N01, N11 = _process_predictions(y_dsel_ex1, y_pred_ones, y_pred_ones)
    assert N00 == 9.0/15.0 and N11 == 6.0/15.0 and N01 == 0.0 and N10 == 0.0


def test_process_predictions_diff_sizes():
    y_pred1 = np.ones(10)
    y_pred2 = np.ones(15)
    with pytest.raises(ValueError):
        _ = _process_predictions(y_dsel_ex1, y_pred1, y_pred2)


def test_double_fault_ones_zeros():
    df = double_fault(y_dsel_ex1, y_pred_ones, y_pred_zeros)
    assert df == 0.0


def test_double_fault_order():
    df1 = double_fault(y_dsel_ex1, y_pred_ones, y_pred_zeros)
    df2 = double_fault(y_dsel_ex1, y_pred_zeros, y_pred_ones)
    assert df1 == df2


def test_double_fault_zeros():
    df = double_fault(y_dsel_ex1, y_pred_zeros, y_pred_zeros)
    assert df == 6.0/15.0


def test_double_fault_ones():
    df = double_fault(y_dsel_ex1, y_pred_ones, y_pred_ones)
    assert df == 9.0/15.0


def test_double_fault():
    labels = np.array([0, 0, 0, 0, 1, 1, 1])
    pred1 = np.array([1, 0, 1, 0, 0, 0, 0])
    pred2 = np.array([1, 0, 0, 0, 1, 0, 0])

    actual = double_fault(labels,
                          pred1,
                          pred2)

    assert actual == 3./7  # three common errors out of 7 predictions


def test_q_statistic_ones_zeros():
    Q = Q_statistic(y_dsel_ex1, y_pred_ones, y_pred_zeros)
    assert Q == -1.0


def test_q_statistic_ones():
    Q = Q_statistic(y_dsel_ex1, y_pred_ones, y_pred_ones)
    assert Q == 1.0


def test_q_statistic_zeros():
    Q = Q_statistic(y_dsel_ex1, y_pred_zeros, y_pred_zeros)
    assert Q == 1.0


def test_ratio_errors_zeros():
    ratio = ratio_errors(y_dsel_ex1, y_pred_zeros, y_pred_zeros)
    assert ratio == 0.0


def test_ratio_errors_ones():
    ratio = ratio_errors(y_dsel_ex1, y_pred_ones, y_pred_ones)
    assert ratio == 0.0


def test_ratio_ones_zeros():
    ratio = ratio_errors(y_dsel_ex1, y_pred_ones, y_pred_zeros)
    assert ratio == sys.float_info.max


def test_ratio_order():
    ratio1 = ratio_errors(y_dsel_ex1, y_pred_ones, y_pred_zeros)
    ratio2 = ratio_errors(y_dsel_ex1, y_pred_zeros, y_pred_ones)
    assert ratio1 == ratio2


def test_ratio_errors_diff_classifiers():
    ratio = ratio_errors(y_real, y_pred_classifier1, y_pred_classifier2)
    assert np.isclose(ratio, 1.66, atol=0.01)
