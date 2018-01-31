from unittest.mock import MagicMock

import numpy as np
import pytest

from deslib.des.des_clustering import DESClustering
from deslib.tests.examples_test import create_pool_classifiers, X_dsel_ex1, y_dsel_ex1
from deslib.util.diversity import Q_statistic, ratio_errors, negative_double_fault

""" Considering a test scenario in which all samples from class 0 are indexed in cluster n. 0 and classes 1 to cluster
n. 1. For this example, the base classifiers that always predicts 0 should me most accurate on the cluster 0, while
the base classifiers that predicts 1 for the cluster with index == 1.
"""
return_cluster_index_ex1 = y_dsel_ex1


""" In this test scenario, each cluster contains samples from classes 1 and 2.
"""
return_cluster_index_ex2 = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1])


def test_fit_homogeneous_clusters():

    clustering_test = DESClustering(create_pool_classifiers()*2, k=2, pct_accuracy=0.5, pct_diversity=0.33)
    clustering_test.roc_algorithm.fit_predict = MagicMock(return_value=return_cluster_index_ex1)

    clustering_test.DFP_mask = np.ones(clustering_test.n_classifiers)
    clustering_test.fit(X_dsel_ex1, y_dsel_ex1)

    assert clustering_test.accuracy_cluster[0, 1] == 0.0 and clustering_test.accuracy_cluster[0, [0, 2]].all() == 1.0
    assert clustering_test.accuracy_cluster[1, 1] == 1.0 and clustering_test.accuracy_cluster[1, [0, 2]].all() == 0.0
    for idx in clustering_test.indices[0, :]:
        assert idx in (0, 2, 3, 5)


def test_fit_heterogeneous_clusters():

    clustering_test = DESClustering(create_pool_classifiers(), k=2, pct_accuracy=0.5, pct_diversity=0.33)
    clustering_test.roc_algorithm.fit_predict = MagicMock(return_value=return_cluster_index_ex2)

    clustering_test.DFP_mask = np.ones(clustering_test.n_classifiers)
    clustering_test.fit(X_dsel_ex1, y_dsel_ex1)

    # Index selected should be of any classifier that predicts the class label 0
    assert np.isclose(clustering_test.accuracy_cluster[:, 1], [0.428, 0.375], atol=0.01).all()
    assert np.isclose(clustering_test.accuracy_cluster[:, 0], [0.572, 0.625], atol=0.01).all()
    assert clustering_test.indices[0, 0] == 0 or clustering_test.indices[0, 0] == 2
    assert clustering_test.indices[1, 0] == 0 or clustering_test.indices[1, 0] == 2


def test_estimate_competence():

    query = np.atleast_2d([1, 1])
    clustering_test = DESClustering(create_pool_classifiers(), k=2, pct_accuracy=0.5, pct_diversity=0.33)
    clustering_test.roc_algorithm.fit_predict = MagicMock(return_value=return_cluster_index_ex2)

    clustering_test.DFP_mask = np.ones(clustering_test.n_classifiers)
    clustering_test.fit(X_dsel_ex1, y_dsel_ex1)
    clustering_test.roc_algorithm.predict = MagicMock(return_value=0)
    competences = clustering_test.estimate_competence(query)

    assert np.array_equal(competences, clustering_test.accuracy_cluster[0, :])

    clustering_test.roc_algorithm.predict = MagicMock(return_value=1)
    competences = clustering_test.estimate_competence(query)
    assert np.array_equal(competences, clustering_test.accuracy_cluster[1, :])


def test_fit_clusters_less_diverse():

    clustering_test = DESClustering(create_pool_classifiers()*2, k=2, pct_accuracy=1.0,
                                    pct_diversity=0.60, more_diverse=False)

    clustering_test.roc_algorithm.fit_predict = MagicMock(return_value=return_cluster_index_ex1)

    clustering_test.DFP_mask = np.ones(clustering_test.n_classifiers)
    clustering_test.fit(X_dsel_ex1, y_dsel_ex1)

    assert clustering_test.accuracy_cluster[0, 1] == 0.0 and clustering_test.accuracy_cluster[0, [0, 2]].all() == 1.0
    assert clustering_test.accuracy_cluster[1, 1] == 1.0 and clustering_test.accuracy_cluster[1, [0, 2]].all() == 0.0
    for idx in clustering_test.indices[0, :]:
        assert idx in (1, 3, 4, 5)


def test_select():
    query = np.atleast_2d([1, -1])
    clustering_test = DESClustering(create_pool_classifiers() * 2, k=2)
    clustering_test.roc_algorithm.predict = MagicMock(return_value=[0])
    clustering_test.indices = np.array([[0, 2], [1, 4]])
    assert np.array_equal(clustering_test.select(query), [0, 2])

    clustering_test.roc_algorithm.predict = MagicMock(return_value=[1])
    assert np.array_equal(clustering_test.select(query), [1, 4])


# Since the majority of the base classifiers selected predicts class 0, the final decision of the ensemble should be 0.
def test_classify_instance():
    query = np.atleast_2d([1, -1])
    clustering_test = DESClustering(create_pool_classifiers() * 4, k=2)
    clustering_test.select = MagicMock(return_value=[0, 1, 2, 3, 5, 6, 7, 9])

    predicted = clustering_test.classify_instance(query)
    assert predicted == 0


def test_input_diversity_parameter():
    with pytest.raises(ValueError):
        DESClustering(create_pool_classifiers()*10, metric='abc')


def test_J_N_values():
    with pytest.raises(ValueError):
        DESClustering(create_pool_classifiers()*10, pct_accuracy=0.5, pct_diversity=0)


def test_J_higher_than_N():
    with pytest.raises(ValueError):
        DESClustering(create_pool_classifiers()*100, pct_accuracy=0.3, pct_diversity=0.5)


def test_diversity_metric_Q():
    test = DESClustering(create_pool_classifiers() * 10, metric='Q')
    assert test.diversity_func == Q_statistic


def test_diversity_metric_DF():
    test = DESClustering(create_pool_classifiers() * 10, metric='DF')
    assert test.diversity_func == negative_double_fault


def test_diversity_metric_ratio():
    test = DESClustering(create_pool_classifiers() * 10, metric='ratio')
    assert test.diversity_func == ratio_errors