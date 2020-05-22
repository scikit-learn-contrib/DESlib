"""Implementation of the Dynamic Frienemy Pruning (DFP) algorithm for online
pruning of base classifiers.

References
----------
Oliveira, D.V.R., Cavalcanti, G.D.C. and Sabourin, R., Online Pruning
of Base Classifiers for Dynamic Ensemble Selection,
Pattern Recognition, vol. 72, December 2017, pp 44-58.

Cruz, Rafael MO, Dayvid VR Oliveira, George DC Cavalcanti, and Robert Sabourin.
"FIRE-DES++: Enhanced online pruning of base classifiers for dynamic ensemble
selection." Pattern Recognition 85 (2019): 149-160.
"""

# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause


import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def frienemy_pruning(X_query, X_dsel, y_dsel, ensemble, k):
    """Implements the Online Pruning method (frienemy) which prunes base
    classifiers that do not cross the region of competence of a given instance.
    A classifier crosses the region of competence if it correctly
    classify at least one sample for each different class in the region.

    Parameters
    ----------
    X_query : array-like of shape (n_samples, n_features)
        Test set.
    X_dsel : array-like of shape (n_samples, n_features)
        Dynamic selection set.
    y_dsel : array-like of shape (n_samples,)
        The target values (Dynamic selection set).
    ensemble : list of shape = [n_classifiers]
        The ensemble of classifiers to be pruned.
    k : int
        Number of neighbors used to compute the regions of competence.

    Returns
    -------
    DFP_mask : array-like of shape = [n_samples, n_classifiers]
               Mask containing 1 for the selected base classifier and 0
               otherwise.

    """
    predictions = np.zeros((X_dsel.shape[0], len(ensemble)),
                           dtype=np.intp)
    for index, clf in enumerate(ensemble):
        predictions[:, index] = clf.predict(X_dsel)
    hit_miss = predictions == y_dsel[:, np.newaxis]
    competence_region = KNeighborsClassifier(n_neighbors=k).fit(X_dsel, y_dsel)
    neighbors = competence_region.kneighbors(X_query, return_distance=False)
    return frienemy_pruning_preprocessed(neighbors, y_dsel, hit_miss)


def frienemy_pruning_preprocessed(neighbors, y_val, hit_miss):
    """Implements the Online Pruning method (frienemy) which prunes base
    classifiers that do not cross the region of competence of a given instance.
    A classifier crosses the region of competence if it correctly
    classify at least one sample for each different class in the region.

    Notes
    -----
    This implementation assumes the regions of competence of each query example
    (neighbors) and the predictions for the dynamic selection data (hit_miss)
    were already pre-computed.

    Parameters
    ----------
    neighbors : array-like of shape (n_samples, n_neighbors)
        Indices of the k nearest neighbors.
    y_val : array-like of shape (n_samples,)
        The target values (class labels).
    hit_miss : array-like of shape (n_samples, n_classifiers)
        Matrix containing 1 when the base classifier made the correct
        prediction, 0 otherwise.

    Returns
    -------
    DFP_mask : array-like of shape = [n_samples, n_classifiers]
               Mask containing 1 for the selected base classifier and 0
               otherwise.
    """
    if neighbors.ndim < 2:
        neighbors = neighbors.reshape(1, -1)

    n_samples = neighbors.shape[0]
    n_classifiers = hit_miss.shape[1]
    dfp_mask = np.zeros((n_samples, n_classifiers))

    # TODO: vectorize this code?
    for sample_idx in range(n_samples):
        curr_neighbors = neighbors[sample_idx]
        neighbors_y = y_val[curr_neighbors]
        if len(set(neighbors_y)) > 1:
            # Indecision region. Check if the base classifier predict the
            # correct label for a sample belonging to each class.
            for clf_index in range(n_classifiers):
                [mask] = np.where(hit_miss[curr_neighbors, clf_index])
                if len(set(neighbors_y[mask])) > 1:
                    dfp_mask[sample_idx, clf_index] = 1.0
        else:
            # Safe region.
            dfp_mask[sample_idx, :] = 1.0
    # rows that all classifiers were pruned are set to 1.0
    dfp_mask[np.all(dfp_mask == 0, axis=1)] = 1.0
    return dfp_mask
