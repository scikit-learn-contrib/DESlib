# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.dcs.base import DCS


class APriori(DCS):
    """A Priori dynamic classifier selection.

    The A Priori method uses the probability of correct classification of a given base classifier :math:`c_{i}` for each
    neighbor :math:`x_{k}` for the competence level estimation. Base classifiers with a higher probability of correct
    classification have a higher competence level. Moreover, the method also weights the influence of each
    neighbor :math:`x_{k}` according to its Euclidean distance to the query sample. The closest neighbors have a higher
    influence on the competence level estimate.

    A single classifier is selected only if its competence level is
    significantly higher than that of the other base classifiers in the pool (higher than a pre-defined threshold).
    Otherwise, all classifiers i the pool are combined using the majority voting rule.

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

        super(APriori, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate,
                                      selection_method=selection_method,
                                      diff_thresh=diff_thresh,
                                      rng=rng)
        self._check_predict_proba()

        self.name = 'A Priori'

    def fit(self, X, y):
        super(APriori, self).fit(X, y)
        self.dsel_scores = self._preprocess_dsel_scores()
        return self

    def estimate_competence(self, query, predictions=None):
        """estimate the competence of each base classifier :math:`c_{i}` for
        the classification of the query sample using the A Priori rule:

        The competence level is estimated based on the probability of correct classification of the base classifier
        :math:`c_{i}`, considering all samples in the region of competence. This method also weights the influence
        of each training sample according to its Euclidean distance to the query instance. The closest samples have
        a higher influence in the computation of the competence level.  The
        competence level estimate is represented by the following equation:

        .. math:: 	\\delta_{i,j} = \\frac{\\sum_{k = 1}^{K}P(\\omega_{l} \\mid
            \mathbf{x}_{k} \\in \\omega_{l}, c_{i} )W_{k}}{\\sum_{k = 1}^{K}W_{k}}

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
        dists_normalized = 1.0/dists

        competences = np.zeros(self.n_classifiers)
        for clf_index in range(self.n_classifiers):

            # Check if the dynamic frienemy pruning (DFP) should be used used
            if self.DFP_mask[clf_index]:
                result = np.zeros(self.k)
                for counter, index in enumerate(idx_neighbors):
                    target = self.DSEL_target[index]
                    # get the post_prob for the correct class
                    #post_prob = self._get_scores_dsel(clf_index, index)[target]
                    post_prob = self.dsel_scores[index, clf_index, target]

                    result[counter] = (post_prob * dists_normalized[counter])

                competences[clf_index] = sum(result)/sum(dists_normalized)
        return competences
