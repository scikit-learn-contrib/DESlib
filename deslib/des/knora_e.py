# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.des.base import DES


class KNORAE(DES):
    """k-Nearest Oracles Eliminate (KNORA-E).
    
    This method searches for a local Oracle, which is a base classifier that correctly classify all
    samples belonging to the region of competence of the test sample. All classifiers
    with a perfect performance in the region of competence is selected. In the case that 
    no classifiers achieve a perfect accuracy, the size of the region of competence is reduced
    (by one neighbor) and the performance of the classifiers are re-evaluated. The outputs
    of the selected ensemble of classifiers is combined using the majority voting scheme.
    
    Parameters
    ----------
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

    References
    ----------
    Ko, Albert HR, Robert Sabourin, and Alceu Souza Britto Jr. "From dynamic classifier selection to dynamic ensemble
    selection." Pattern Recognition 41.5 (2008): 1718-1731.   

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """

    def __init__(self, pool_classifiers, k=7, DFP=False, with_IH=False, safe_k=None,
                 IH_rate=0.30):

        super(KNORAE, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate)

        self.name = 'k-Nearest Oracles Eliminate (KNORA-E)'

    def estimate_competence(self, query):
        """Estimate the competence of the base classifiers. In the case of the KNORA-E technique, the classifiers
        are only considered competent when they achieve a 100% accuracy in the region of competence. For each base,
        we estimate the maximum size of the region of competence that it is a local oracle (achieves 100%). The
        competence level estimate is then the maximum size of the region of competence that the corresponding base
        classifier is a local Oracle.

        Parameters
        ----------
        query : array of shape = [n_features]
                The test sample

        Returns
        -------

        competences : array of shape = [n_classifiers]
                      The competence level estimated for each base classifier in the pool
        """
        _, idx_neighbors = self._get_region_competence(query)
        competences = np.zeros(self.n_classifiers)

        for clf_index in range(self.n_classifiers):
            # Check if the dynamic frienemy pruning (DFP) should be used used
            if self.DFP_mask[clf_index]:
                results_neighbors = self.processed_dsel[idx_neighbors, clf_index]
                indices_errors = np.where(results_neighbors == 0)[0]
                if indices_errors.size != 0:
                    competences[clf_index] = np.min(indices_errors)
                else:
                    competences[clf_index] = self.k
        return competences

    def select(self, competences):
        """Selects all base classifiers that obtained a local accuracy of 100% in the region of competence
        (i.e., local oracle). In the case that no base classifiers obtain 100% accuracy, the size of the region
        of competence is reduced and the search for the local oracle is restarted.

        Notes
        ------
        Instead of re-applying the method several times (reducing the size of the region of competence),
        we compute the number of consecutive correct classification of each base classifier starting from the
        closest neighbor to the more distant in the estimate_competence function. The number of consecutive correct
        classification represents the size of the region of competence in which the corresponding base classifier
        is an Local Oracle. Then, we select all base classifiers with the maximum value for the number of consecutive
        correct classification. This speed up the selection process.

        Parameters
        ----------
        competences : array of shape = [n_classifiers]
                      The competence level estimated for each base classifier

        Returns
        -------
        indices : List with the indices of the selected base classifiers

        """
        max_value = np.max(competences)
        if max_value > 0:
            indices = [clf_index for clf_index, clf_competence in enumerate(
                competences) if clf_competence == max_value]
        else:
            # use the whole pool if no classifier was deemed competent
            indices = list(range(self.n_classifiers))

        return indices
