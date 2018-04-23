# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.dcs.base import DCS


class MLA(DCS):
    """Modified Local Accuracy (MLA).
    
    Similar to the LCA technique. The only difference is that the output of each base classifier
    is weighted by the distance between the test sample and each pattern in the region of competence 
    for the estimation of the classifiers competences. Only the classifier that achieved the highest
    competence level is select to predict the label of the test sample x.

    The MLA method selects the base classifier presenting the highest competence level. In a case
    where more than one base classifier achieves the same competence level, the one that was evaluated first
    is selected. The selection methodology can be modified by changing the hyper-parameter selection_method.

    Parameters
    ----------
    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support the method "predict".

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
    Woods, Kevin, W. Philip Kegelmeyer, and Kevin Bowyer. "Combination of multiple classifiers
    using local accuracy estimates." IEEE transactions on pattern analysis and machine intelligence
    19.4 (1997): 405-410.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.
    """
    def __init__(self, pool_classifiers, k=7, DFP=False, with_IH=False, safe_k=None, IH_rate=0.30,
                 selection_method='best',
                 diff_thresh=0.1,
                 rng=np.random.RandomState()):

        super(MLA, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate,
                                  selection_method=selection_method,
                                  diff_thresh=diff_thresh,
                                  rng=rng)
        self.name = 'Modified Local Accuracy (MLA)'

    def estimate_competence(self, query, predictions=None):
        """estimate the competence of each base classifier :math:`c_{i}` for
        the classification of the query sample using the Modified Local Accuracy (MLA) method.

        The competence level of the base classifiers is estimated by its classification accuracy taking
        into account only the samples belonging to a given class :math:`w_{l}`.In this case, :math:`w_{l}` is
        the class predicted by the base classifier :math:`c_{i}`, for the query sample. This method also weights
        the influence of each training sample according to its Euclidean distance to the query instance.
        The closest samples have a higher influence in the computation of the competence level. The
        competence level estimate is represented by the following equation:

        .. math:: \\delta_{i,j} = \\sum_{k = 1}^{K}P(\\omega_{l} \\mid \\mathbf{x}_{k} \\in \\omega_{l}, c_{i} )W_{k}

        where :math:`\\delta_{i,j}` represents the competence level of :math:`c_{i}` for the classification of
        query.

        Parameters
        ----------
        query : array cf shape  = [n_samples, n_features]
                The query sample.

        predictions : array of shape = [n_samples, n_classifiers]
                      Predictions of the base classifiers for the test examples.

        Returns
        -------
        competences : array of shape = [n_samples, n_classifiers]
                      Competence level estimated for each base classifier.
        """

        dists, idx_neighbors = self._get_region_competence(query)
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
        proc_norm = self.processed_dsel[idx_neighbors, :] * dists_normalized

        # Create masked arrays to remove samples with different label in the calculations
        masked_preprocessed = np.ma.MaskedArray(proc_norm, mask=mask)
        masked_dist = np.ma.MaskedArray(dists_normalized, mask=mask)

        # Consider only the neighbor samples where the predicted label is equals to the neighbor label
        competences_masked = np.ma.sum(masked_preprocessed, axis=1)/ np.ma.sum(masked_dist, axis=1)

        # Fill 0 to the masked values in the resulting array (when no neighbors belongs to the class predicted by
        # the corresponding base classifier)
        competences = np.ma.filled(competences_masked, 0)

        return competences
