"""Implementation of the FIRE-DES algorithm."""

# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause


import numpy as np


def frienemy_pruning(neighbors, n_classifiers, y, predictions):
    """Implements the Online Pruning method (frienemy) to remove base
    classifiers that do not cross the region of competence. We consider
    that a classifier crosses the region of competence if it correctly
    classify at least one sample for each different class in the region.

    Parameters
    ----------
    neighbors :

    n_classifiers :

    y :

    predictions :

    Returns
    -------
    DFP_mask : array of shape = [n_samples, n_classifiers]
               Mask containing 1 for the selected base classifier and 0
               otherwise.
    References
    ----------
    Oliveira, D.V.R., Cavalcanti, G.D.C. and Sabourin, R., Online Pruning
    of Base Classifiers for Dynamic Ensemble Selection,
    Pattern Recognition, vol. 72, December 2017, pp 44-58.
    """
    # using a for loop for processing a batch of samples temporarily.
    # Change later to numpy processing
    if neighbors.ndim < 2:
        neighbors = np.atleast_2d(neighbors)
    n_samples, _ = neighbors.shape
    mask = np.zeros((n_samples, n_classifiers))

    # TODO: vectorize this part of the code
    for sample_idx in range(n_samples):
        # Check if query is in a indecision region
        neighbors_y = y[neighbors[sample_idx]]

        if len(set(neighbors_y)) > 1:
            # There are more than on class in the region of competence
            # (So it is an indecision region).
            # Check if the base classifier predict the correct label for
            # a sample belonging to each class.
            for clf_index in range(n_classifiers):
                clf_predictions = predictions[
                    neighbors[sample_idx], clf_index]
                correct_class_pred = [y[index] for
                                      count, index in
                                      enumerate(neighbors[sample_idx])
                                      if clf_predictions[count] == 1]
                # True means that it correctly classified
                # at least one neighbor for each class in
                # the region of competence
                if np.unique(correct_class_pred).size > 1:
                    mask[sample_idx, clf_index] = 1.0
            # Check if all classifiers were pruned
            if not np.count_nonzero(mask[sample_idx, :]):
                # Do not apply the pruning mechanism.
                mask[sample_idx, :] = 1.0

        else:
            # The sample is located in a safe region. All base classifiers
            # can predict the label
            mask[sample_idx, :] = 1.0

    return mask
