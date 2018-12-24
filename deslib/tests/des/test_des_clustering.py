from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import Perceptron
from sklearn.utils.estimator_checks import check_estimator

from deslib.des.des_clustering import DESClustering
from deslib.util.diversity import Q_statistic, ratio_errors, \
    negative_double_fault


def test_check_estimator():
    check_estimator(DESClustering)


""" Considering a test scenario in which all samples from class 0 are indexed
in cluster n. 0 and classes_ 1 to cluster n. 1. For this example, the base
classifiers that always predicts 0 should me most accurate on the cluster 0,
while the base classifiers that predicts 1 for the cluster with index == 1.
"""

""" In this test scenario, each cluster contains samples from classes_ 1 and 2.
"""
return_cluster_index_ex2 = np.array(
    [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1])


def test_fit_homogeneous_clusters(create_pool_classifiers,
                                  example_estimate_competence):
    clustering_test = DESClustering(create_pool_classifiers * 2,
                                    clustering=KMeans(n_clusters=2),
                                    pct_accuracy=0.5, pct_diversity=0.33)

    X, y = example_estimate_competence[0:2]
    clustering_test.clustering.predict = MagicMock(return_value=y)

    clustering_test.fit(X, y)

    assert (clustering_test.accuracy_cluster_[0, 1] == 0.0 and
            clustering_test.accuracy_cluster_[0, [0, 2]].all() == 1.0)
    assert (clustering_test.accuracy_cluster_[1, 1] == 1.0 and
            clustering_test.accuracy_cluster_[1, [0, 2]].all() == 0.0)
    for idx in clustering_test.indices_[0, :]:
        assert idx in (0, 2, 3, 5)


def test_fit_heterogeneous_clusters(example_estimate_competence,
                                    create_pool_classifiers):
    clustering_test = DESClustering(create_pool_classifiers,
                                    clustering=KMeans(n_clusters=2),
                                    pct_accuracy=0.5,
                                    pct_diversity=0.33)
    X, y = example_estimate_competence[0:2]

    clustering_test.clustering.predict = MagicMock(
        return_value=return_cluster_index_ex2)
    clustering_test.fit(X, y)

    # Index selected should be of any classifier that predicts the label 0
    assert np.isclose(clustering_test.accuracy_cluster_[:, 1], [0.428, 0.375],
                      atol=0.01).all()
    assert np.isclose(clustering_test.accuracy_cluster_[:, 0], [0.572, 0.625],
                      atol=0.01).all()
    assert clustering_test.indices_[0, 0] == 0 or clustering_test.indices_[
        0, 0] == 2
    assert clustering_test.indices_[1, 0] == 0 or clustering_test.indices_[
        1, 0] == 2


def test_estimate_competence(create_pool_classifiers,
                             example_estimate_competence):
    query = np.atleast_2d([1, 1])
    clustering_test = DESClustering(create_pool_classifiers * 2,
                                    clustering=KMeans(n_clusters=2),
                                    pct_accuracy=0.5, pct_diversity=0.33)

    X, y = example_estimate_competence[0:2]

    # Keep the original predict method to change after
    clustering_test.clustering.predict = MagicMock(
        return_value=return_cluster_index_ex2)
    clustering_test.fit(X, y)

    clustering_test.clustering_.predict = MagicMock(return_value=0)
    competences = clustering_test.estimate_competence(query)

    assert np.array_equal(competences, clustering_test.accuracy_cluster_[0, :])

    clustering_test.clustering_.predict = MagicMock(return_value=1)
    competences = clustering_test.estimate_competence(query)
    assert np.array_equal(competences, clustering_test.accuracy_cluster_[1, :])


def test_fit_clusters_less_diverse(example_estimate_competence,
                                   create_pool_classifiers):
    clustering_test = DESClustering(create_pool_classifiers * 2,
                                    clustering=KMeans(n_clusters=2),
                                    pct_accuracy=0.5, pct_diversity=0.33,
                                    more_diverse=False)
    X, y = example_estimate_competence[0:2]

    clustering_test.clustering.predict = MagicMock(return_value=y)
    clustering_test.fit(X, y)

    assert (clustering_test.accuracy_cluster_[0, 1] == 0.0 and
            clustering_test.accuracy_cluster_[0, [0, 2]].all() == 1.0)
    assert (clustering_test.accuracy_cluster_[1, 1] == 1.0 and
            clustering_test.accuracy_cluster_[1, [0, 2]].all() == 0.0)
    for idx in clustering_test.indices_[0, :]:
        assert idx in (1, 3, 4, 5)


def test_select():
    query = np.atleast_2d([1, -1])
    clustering_test = DESClustering()

    clustering_test.clustering_ = KMeans()
    clustering_test.clustering_.predict = MagicMock(return_value=[0])
    clustering_test.indices_ = np.array([[0, 2], [1, 4]])
    assert np.array_equal(clustering_test.select(query), [[0, 2]])

    clustering_test.clustering_.predict = MagicMock(return_value=[1])
    assert np.array_equal(clustering_test.select(query), [[1, 4]])


# Since the majority of the base classifiers selected predicts class 0,
# the final decision of the ensemble should be 0.
def test_classify_instance(create_pool_classifiers):
    query = np.ones((1, 2))
    clustering_test = DESClustering(create_pool_classifiers * 4,
                                    clustering=KMeans(n_clusters=2))

    clustering_test.select = MagicMock(return_value=[0, 1, 2, 3, 5, 6, 7, 9])
    predictions = []
    for clf in clustering_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])

    predicted = clustering_test.classify_with_ds(query, np.array(predictions))
    assert predicted == 0


def test_input_diversity_parameter(create_X_y):
    X, y = create_X_y

    with pytest.raises(ValueError):
        des_clustering = DESClustering(metric='abc')
        des_clustering.fit(X, y)


def test_J_N_values(create_X_y):
    X, y = create_X_y

    with pytest.raises(ValueError):
        des_clustering = DESClustering(pct_accuracy=0.5, pct_diversity=0)
        des_clustering.fit(X, y)


def test_J_higher_than_N(create_X_y):
    X, y = create_X_y

    with pytest.raises(ValueError):
        des_clustering = DESClustering(pct_accuracy=0.3, pct_diversity=0.5)
        des_clustering.fit(X, y)


def test_diversity_metric_Q(create_X_y):
    X, y = create_X_y

    test = DESClustering(metric='Q')
    # Mocking this method to avoid preprocessing the cluster information
    # that is not required in this test.
    test._preprocess_clusters = MagicMock(return_value=1)
    test.fit(X, y)
    assert test.diversity_func_ == Q_statistic


def test_diversity_metric_DF(create_X_y):
    X, y = create_X_y

    test = DESClustering(metric='DF')
    # Mocking this method to avoid preprocessing the cluster
    # information that is not required in this test.
    test._preprocess_clusters = MagicMock(return_value=1)
    test.fit(X, y)
    assert test.diversity_func_ == negative_double_fault


def test_diversity_metric_ratio(create_X_y):
    X, y = create_X_y

    test = DESClustering(metric='ratio')
    # Mocking this method to avoid preprocessing the cluster
    # information that is not required in this test.
    test._preprocess_clusters = MagicMock(return_value=1)
    test.fit(X, y)
    assert test.diversity_func_ == ratio_errors


# Test if the class is raising an error when the base classifiers do not
# implements the predict_proba method. In this case the test should not raise
# an error since this class does not require base classifiers that
# can estimate probabilities
def test_predict_proba(example_estimate_competence):
    X, y = example_estimate_competence[0:2]

    clf1 = Perceptron()
    clf1.fit(X, y)
    DESClustering([clf1, clf1]).fit(X, y)


def test_classify_with_ds_single_sample():
    query = np.ones(2)
    predictions = np.array([0, 1, 0])

    des_clustering_test = DESClustering()
    des_clustering_test.select = MagicMock(return_value=np.array([[0, 2]]))
    result = des_clustering_test.classify_with_ds(query, predictions)
    assert np.allclose(result, 0)


def test_classify_with_ds_diff_sizes():
    query = np.ones((10, 2))
    predictions = np.ones((5, 3))

    des_clustering = DESClustering()

    with pytest.raises(ValueError):
        des_clustering.classify_with_ds(query, predictions)


def test_proba_with_ds_diff_sizes():
    query = np.ones((10, 2))
    predictions = np.ones((5, 3))
    probabilities = np.ones((5, 3, 2))

    des_clustering = DESClustering()

    with pytest.raises(ValueError):
        des_clustering.predict_proba_with_ds(query, predictions, probabilities)


def test_not_clustering_algorithm(create_X_y):
    X, y = create_X_y

    des_clustering = DESClustering(clustering=Perceptron())
    with pytest.raises(ValueError):
        des_clustering.fit(X, y)
