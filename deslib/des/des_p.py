# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.des.base import DES


class DESP(DES):
    """Dynamic ensemble selection-Performance(des-p).
    This method selects all base classifiers that achieve a classification
    performance, in the region of competence, that is higher than the random
    classifier (RC). The performance of the random classifier is defined by
    RC = 1/M, where M is the number of classes in the problem.

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
              Whether the hardness level of the region of competence is used to decide between
              using the DS algorithm or the KNN for classification of a given query sample.

    safe_k : int (default = None)
             The size of the indecision region.

    IH_rate : float (default = 0.3)
              Hardness threshold. If the hardness level of the competence region is lower than
              the IH_rate the KNN classifier is used. Otherwise, the DS algorithm is used for classification.

    mode : String (Default = "selection")
           Whether the technique will perform dynamic selection,
           dynamic weighting or an hybrid approach for classification.

    References
    ----------
    Woloszynski, Tomasz, et al. "A measure of competence based on random classification
    for dynamic ensemble selection." Information Fusion 13.3 (2012): 207-213.

    Woloszynski, Tomasz, and Marek Kurzynski. "A probabilistic model of classifier competence
    for dynamic ensemble selection." Pattern Recognition 44.10 (2011): 2656-2668.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.
    """

    def __init__(self, pool_classifiers, k=7, DFP=False, with_IH=False, safe_k=None,
                 IH_rate=0.30,
                 mode='selection'):

        super(DESP, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate,
                                   mode=mode)

        self.name = 'DES-Performance (DES-P)'

    def estimate_competence(self, query):
        """estimate the competence of each base classifier in the pool. The competence level is estimated
         based on the classification accuracy of the base classifier for the region of competence.

        Parameters
        ----------
        query : array of shape = [n_features]
                The test sample

        Returns
        -------
        competences : array of shape = [n_classifiers]
                      The competence level estimated for each base classifier
        """
        dists, idx_neighbors = self._get_region_competence(query)
        competences = np.zeros(self.n_classifiers)
        for clf_index in range(self.n_classifiers):
            # Check if the dynamic frienemy pruning (DFP) should be used used
            if self.DFP_mask[clf_index]:
                competences[clf_index] = np.mean(self.processed_dsel[idx_neighbors, clf_index])

        return competences

    def select(self, competences):
        """Selects all base classifiers that obtained a local classification accuracy higher than the
        Random Classifier. The performance of the random classifier is denoted 1/L, where L is the number of classes
        in the problem.

        Parameters
        ----------
        competences : array of shape = [n_classifiers] containing the competence level estimated
                      for each base classifier.

        Returns
        -------
        indices : List with the indices of the selected base classifiers.

        """
        RC = (1.0 / self.n_classes)
        # Select classifiers with local accuracy superior than the random classifier rc.
        indices = [clf_index for clf_index, clf_competence in enumerate(competences)
                   if clf_competence > RC]

        if len(indices) == 0:
            indices = list(range(self.n_classifiers))
        return indices
