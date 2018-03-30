# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.dcs.base import DCS


class APosteriori(DCS):
    """A Posteriori Dynamic classifier selection.

    The A Posteriori method uses the probability of correct classification of a given base classifier :math:`c_{i}`
    for each neighbor :math:`x_{k}` with respect to a single class. Consider a classifier :math:`c_{i}` that assigns
    a test sample to class :math:`w_{l}`. Then, only the samples belonging to class :math:`w_{l}` are taken into
    account during the competence level estimates. Base classifiers with a higher probability of correct
    classification have a higher competence level. Moreover, the method also weights the influence of each
    neighbor :math:`x_{k}` according to its Euclidean distance to the query sample. The closest neighbors have a higher
    influence on the competence level estimate. In cases where no sample in the region of competence belongs to the
    predicted class, :math:`w_{l}`, the competence level estimate of the base classifier is equal to zero.

    A single classifier is selected only if its competence level is
    significantly higher than that of the other base classifiers in the pool (higher than a pre-defined threshold).
    Otherwise, all classifiers in the pool are combined using the majority voting rule. The selection methodology can
    be modified by modifying the hyper-parameter selection_method.

    Parameters
    ----------
    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support methods "predict" and "predict_proba".

    k : int (Default = 7)
        Number of neighbors used to estimate the competence of the base classifiers.

    DFP : Boolean (Default = False)
          Determines if the dynamic frienemy pruning is applied.

    with_IH : Boolean (Default = False)
              Whether the hardness level of the region of competence is used to decide
              between using the DS algorithm or the KNN for classification of a given query sample.

    safe_k : int (default = None)
             The size of the indecision region.

    IH_rate : float (default = 0.3)
              Hardness threshold. If the hardness level of the competence region is lower than
              the IH_rate the KNN classifier is used. Otherwise, the DS algorithm is used for classification.

    selection_method : String (Default = "best")
                       Determines which method is used to select the base classifier
                       after the competences are estimated.

    diff_thresh : float (Default = 0.1)
                  Threshold to measure the difference between the competence level of the base
                  classifiers for the random and diff selection schemes. If the difference is lower than the
                  threshold, their performance are considered equivalent.

    rng : numpy.random.RandomState instance
          Random number generator to assure reproducible results.

    References
    ----------
    G. Giacinto and F. Roli, Methods for Dynamic Classifier Selection
    10th Int. Conf. on Image Anal. and Proc., Venice, Italy (1999), 659-664.

    Ko, Albert HR, Robert Sabourin, and Alceu Souza Britto Jr. "From dynamic classifier selection to dynamic ensemble
    selection." Pattern Recognition 41.5 (2008): 1718-1731.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """
    def __init__(self, pool_classifiers, k=7, DFP=False, with_IH=False, safe_k=None, IH_rate=0.30,
                 selection_method='diff', diff_thresh=0.1, rng=np.random.RandomState()):

        super(APosteriori, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate,
                                          selection_method=selection_method,
                                          diff_thresh=diff_thresh,
                                          rng=rng)
        self._check_predict_proba()

        self.name = 'A Posteriori'

    def fit(self, X, y):
        super(APosteriori, self).fit(X, y)
        self.dsel_scores = self._preprocess_dsel_scores()
        return self

    def estimate_competence(self, query, predictions):
        """estimate the competence of each base classifier :math:`c_{i}` for
        the classification of the query sample using the A Posteriori method.

        The competence level is estimated based on the probability of correct classification of the base classifier
        :math:`c_{i}`, for each neighbor :math:`x_{k}` belonging to a specific class :math:`w_{l}`.
        In this case, :math:`w_{l}` is  the class predicted by the base classifier :math:`c_{i}`, for the query sample.
        This method also weights the influence of each training sample according to its Euclidean distance to the
        query instance. The closest samples have a higher influence in the computation of the competence level. The
        competence level estimate is represented by the following equation:

        .. math:: \\delta_{i,j} = \\frac{\\sum_{\\mathbf{x}_{k} \\in \\omega_{l}}P(\\omega_{l} \\mid
            \\mathbf{x}_{k}, c_{i} )W_{k}}{\\sum_{k = 1}^{K}P(\\omega_{l} \\mid \\mathbf{x}_{k}, c_{i} )W_{k}}

        where :math:`\\delta_{i,j}` represents the competence level of :math:`c_{i}` for the classification of
        query.

        Parameters
        ----------
        query : array cf shape  = [n_features]
                The query sample

        predictions : array of shape = [n_samples, n_classifiers]
                      Contains the predictions of all base classifier for all samples in the query array

        Returns
        -------
        competences : array of shape = [n_classifiers]
                      The competence level estimated for each base classifier
        """
        dists, idx_neighbors = self._get_region_competence(query)
        dists = np.atleast_2d(dists)
        idx_neighbors = np.atleast_2d(idx_neighbors)
        predictions = np.atleast_2d(predictions)

        # Normalize the distances
        dists_normalized = 1.0/dists

        # Expanding the dimensions of the predictions and target arrays in order to compare both.
        predictions_3d = np.expand_dims(predictions, axis=1)
        target_3d = np.expand_dims(self.DSEL_target[idx_neighbors], axis=2)
        # Create a mask to remove the neighbors belonging to a different class than the predicted by the base classifier
        mask = (predictions_3d != target_3d)

        # Broadcast the distance array to the same shape as the pre-processed information for future calculations
        dists_normalized = np.repeat(np.expand_dims(dists_normalized, axis=2), self.n_classifiers, axis=2)

        # Multiply the pre-processed correct predictions by the base classifiers to the distance array
        scores_target_norm = self.dsel_scores[idx_neighbors, :, self.DSEL_target[idx_neighbors]] * dists_normalized

        # Create masked arrays to remove samples with different label in the calculations
        masked_preprocessed = np.ma.MaskedArray(scores_target_norm, mask=mask)
        masked_dist = np.ma.MaskedArray(dists_normalized, mask=mask)

        # Consider only the neighbor samples where the predicted label is equals to the neighbor label
        competences_masked = np.ma.sum(masked_preprocessed, axis=1)/ np.ma.sum(masked_dist, axis=1)

        # Fill 0 to the masked values in the resulting array (when no neighbors belongs to the class predicted by
        # the corresponding base classifier)
        competences = np.ma.filled(competences_masked, 0)

        # dists, idx_neighbors = self._get_region_competence(query)
        # dists_normalized = 1.0/dists
        # competences = np.zeros(self.n_classifiers)
        #
        # for clf_index, clf in enumerate(self.pool_classifiers):
        #
        #     # Check if the dynamic frienemy pruning (DFP) should be used used
        #     if self.DFP_mask[clf_index]:
        #
        #         result = []
        #         dists_temp = []
        #         predicted_label = predictions[clf_index]
        #
        #         for counter, neighbor in enumerate(idx_neighbors):
        #             # Get only neighbors from the same class as predicted by the
        #             # classifier (clf) to form the region of competence
        #             target = self.DSEL_target[neighbor]
        #             if target == predicted_label:
        #                 # get the posterior probability for the target class
        #                 post_prob = self.dsel_scores[neighbor, clf_index, target]
        #                 # weight by distance
        #                 result.append(post_prob * dists_normalized[counter])
        #                 # keep the distance for normalization
        #                 dists_temp.append(dists_normalized[counter])
        #         if len(result) > 0 and len(dists_temp) > 0:
        #             competences[clf_index] = sum(result)/sum(dists_temp)
        #         else:
        #             competences[clf_index] = 0

        return competences
