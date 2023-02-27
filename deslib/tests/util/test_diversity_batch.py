import numpy as np
import pytest

from deslib.util.diversity_batch import (_process_predictions,
                                         double_fault,
                                         Q_statistic,
                                         ratio_errors,
                                         agreement_measure,
                                         disagreement_measure,
                                         correlation_coefficient)


@pytest.fixture
def create_X_y():
    # ex1: The distribution of samples of a test example.
    X = np.array(
        [
            [-1, 1],
            [-0.75, 0.5],
            [-1.5, 1.5],
            [1, 1],
            [0.75, 0.5],
            [1.5, 1.5],
            [1, -1],
            [-0.5, 0.5],
            [0.5, 0.5],
            [0, -1],
            [0.75, -0.5],
            [0.0, 0.0],
            [-1, -1],
            [0, -0.5],
            [1, -1],
        ]
    )
    # Labels associated with the samples. This information is used
    # by techniques based on a posteriori information.
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    return X, y


@pytest.fixture
def example_diversity(create_X_y):
    y_pred_classifier1 = np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    y_pred_classifier2 = np.tile(np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1]),
                                 (5, 1))

    y_real = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 1])

    y_ex1 = create_X_y[1]
    return y_pred_classifier1, y_pred_classifier2, y_real, y_ex1


@pytest.fixture
def example_diversity_ones_zeros(create_X_y):
    y = create_X_y[1]
    y_pred_ones = np.ones(15)
    y_pred_zeros = np.zeros((5, 15))
    return y, y_pred_ones, y_pred_zeros


def test_process_predictions_ones_zeros(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    N00, N10, N01, N11 = _process_predictions(y, y_pred_ones, y_pred_zeros)
    assert (
            (N00 == np.full((5,), 0.0)).all() and
            (N11 == np.full((5,), 0.0)).all() and
            (N01 == np.full((5,), 9.0 / 15.0)).all() and
            (N10 == np.full((5,), 6.0 / 15.0)).all()
    )


def test_double_fault_ones_zeros(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    df = double_fault(y, y_pred_ones, y_pred_zeros)
    assert (df == np.full((5,), 0)).all()


def test_double_fault():
    labels = np.array([0, 0, 0, 0, 1, 1, 1])
    pred1 = np.array([1, 0, 1, 0, 0, 0, 0])
    pred2 = np.tile(np.array([1, 0, 0, 0, 1, 0, 0]), (5, 1))

    actual = double_fault(labels, pred1, pred2)

    assert (
            actual == np.full((5,), 3.0 / 7)
    ).all()  # three common errors out of 7 predictions


def test_q_statistic_ones_zeros(example_diversity_ones_zeros):
    y, y_pred_ones, y_pred_zeros = example_diversity_ones_zeros
    Q = Q_statistic(y, y_pred_ones, y_pred_zeros)
    assert (Q == np.full((5,), -1.0)).all()


def test_ratio_errors_diff_classifiers(example_diversity):
    y_pred_classifier1, y_pred_classifier2, y_real, y_ex1 = example_diversity
    ratio = ratio_errors(y_real, y_pred_classifier1, y_pred_classifier2)
    assert np.isclose(ratio, 1.66, atol=0.01).all()


def test_agreement(example_diversity):
    y_pred_classifier1, y_pred_classifier2, y_real, y_ex1 = example_diversity
    agreement = agreement_measure(y_real,
                                  y_pred_classifier1,
                                  y_pred_classifier2)
    assert np.isclose(agreement, 0.5).all()


def test_disagreement(example_diversity):
    y_pred_classifier1, y_pred_classifier2, y_real, y_ex1 = example_diversity
    disagreement = disagreement_measure(y_real,
                                        y_pred_classifier1,
                                        y_pred_classifier2)
    assert np.isclose(disagreement, 0.5).all()


def test_coefficient_correlation(example_diversity):
    y_pred_classifier1, y_pred_classifier2, y_real, y_ex1 = example_diversity
    coefficient = correlation_coefficient(
        y_real, y_pred_classifier1, y_pred_classifier2
    )
    assert np.isclose(coefficient, 0.0).all()
