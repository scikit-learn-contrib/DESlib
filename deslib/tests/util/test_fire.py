import numpy as np

from deslib.util.dfp import frienemy_pruning
from deslib.util.dfp import frienemy_pruning_preprocessed
from ..conftest import create_base_classifier


# Since no classifier crosses the region of competence,
# all of them must be selected
def test_frienemy_no_classifier_crosses(example_estimate_competence):
    _, y, neighbors = example_estimate_competence[0:3]
    n_classifiers = 3
    predictions = np.zeros((y.size, n_classifiers))
    mask = frienemy_pruning_preprocessed(neighbors, y, predictions)
    assert mask.all()


# In this example, all base classifier should be considered crossing the
# region of competence since they always predicts the correct label for
# the samples in DSEL.
def test_frienemy_all_classifiers_crosses(example_all_ones):
    X, y, neighbors, _, dsel_processed, _ = example_all_ones
    result = frienemy_pruning_preprocessed(neighbors, y, dsel_processed)
    assert result.all()


# Check if the batch processing is working by passing multiple
# samples at the same time.
def test_frienemy_not_all_classifiers_crosses(example_estimate_competence):
    expected = np.array([[1, 1, 0], [0, 1, 0], [1, 1, 1]])
    _, y, neighbors, _, dsel_processed, _ = example_estimate_competence

    # passing three samples to compute the DFP at the same time
    result = frienemy_pruning_preprocessed(neighbors[:, :3], y, dsel_processed)
    assert np.array_equal(result, expected)


# Test the case where the sample is located in a safe region
# (i.e., all neighbors comes from the same class)
def test_frienemy_safe_region(example_estimate_competence):
    X, y, _, _, dsel_processed, _ = example_estimate_competence
    neighbors = np.tile(np.array([0, 1, 2, 6, 7, 8, 14]), (10, 1))

    result = frienemy_pruning_preprocessed(neighbors, y, dsel_processed)
    assert result.all()


def test_frienemy_not_processed():
    X = np.random.rand(5, 2)
    y = np.array([0, 0, 0, 1, 1])
    X_query = np.random.rand(1, 2)
    clf1 = create_base_classifier(return_value=[0, 1, 0, 0, 1])
    clf2 = create_base_classifier(return_value=[1, 1, 1, 1, 1])
    clf3 = create_base_classifier(return_value=[0, 0, 0, 0, 0])
    clf4 = create_base_classifier(return_value=[0, 0, 0, 1, 1])
    pool = [clf1, clf2, clf3, clf4]
    dfp_mask = frienemy_pruning(X_query, X, y, pool, 5)
    assert np.array_equal(dfp_mask, np.array([[1, 0, 0, 1]]))
