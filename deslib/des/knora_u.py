# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from scipy.stats import mode

from deslib.des.base import DES


class KNORAU(DES):
    """k-Nearest Oracles Union (KNORA-U).
    
    This method works selects all classifiers that correctly classified at least
    one sample belonging to the region of competence of the test sample x. Each 
    selected classifier has a number of votes equals to the number of samples in the
    region of competence that it predicts the correct label.

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


    aknn : Boolean (Default = False)
           Determines the type of KNN algorithm that is used. set to true for the A-KNN method.

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

        super(KNORAU, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate)

        self.name = 'k-Nearest Oracles Union (KNORA-U)'

    def estimate_competence(self, query):
        """The competence of the base classifiers is simply estimated as the number of samples
        in the region of competence that it correctly classified.

        Parameters
        ----------
        query : array of shape = [n_features] containing the test sample

        Returns
        -------
        competences : array of shape = [n_classifiers] containing the competence level estimated
                     for each base classifier
        """
        dists, idx_neighbors = self._get_region_competence(query)
        competences = np.zeros(self.n_classifiers)

        for clf_index in range(self.n_classifiers):
            # Check if the dynamic frienemy pruning (DFP) should be used used
            if self.DFP_mask[clf_index]:
                competences[clf_index] = np.sum(self.processed_dsel[idx_neighbors, clf_index])

        return competences.astype(dtype=int)

    def select(self, query):
        """Select the base classifiers for the classification of the query sample.

        Each base classifier can be selected more than once. The number of times a base classifier is selected (votes)
        is equals to the number of samples it correctly classified in the region of competence.

        Parameters
        ----------
        query : array of shape = [n_features] containing the test sample

        Returns
        -------
        votes : the number of votes for each class
        """
        weights = self.estimate_competence(query)
        if np.sum(weights) == 0:
            weights = np.ones(self.n_classifiers, dtype=int)

        votes = np.array([], dtype=int)
        for clf_idx, clf in enumerate(self.pool_classifiers):
            votes = np.hstack((votes, np.ones(weights[clf_idx], dtype=int) * clf.predict(query)[0]))

        return votes

    def classify_instance(self, query):
        """Predicts the label of the corresponding query sample.

        The prediction is made by aggregating the votes obtained by all selected base classifiers.

        Parameters
        ----------
        query : array of shape = [n_features]
                The test sample

        Returns
        -------
        predicted_label : Prediction of the ensemble for the input query.
        """
        votes = self.select(query)
        predicted_label = mode(votes)[0]
        return predicted_label
