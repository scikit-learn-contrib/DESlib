# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from scipy.stats import mode

from deslib.des.base import DES


class KNOP(DES):
    """k-Nearest Output Profiles (KNOP).

    Parameters
    ----------
    pool_classifiers : type, the generated_pool of classifiers trained for the corresponding
    classification problem.

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
    Cavalin, Paulo R., Robert Sabourin, and Ching Y. Suen. "LoGID: An adaptive framework combining local and global
    incremental learning for dynamic selection of ensembles of HMMs." Pattern Recognition 45.9 (2012): 3544-3556.

    Cavalin, Paulo R., Robert Sabourin, and Ching Y. Suen. "Dynamic selection approaches for multiple classifier
    systems." Neural Computing and Applications 22.3-4 (2013): 673-688.

    Ko, Albert HR, Robert Sabourin, and Alceu Souza Britto Jr. "From dynamic classifier selection to dynamic ensemble
    selection." Pattern Recognition 41.5 (2008): 1718-1731.   

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """

    def __init__(self, pool_classifiers, k=7, DFP=False, with_IH=False, safe_k=None,
                 IH_rate=0.30,
                 weighted=False):

        super(KNOP, self).__init__(pool_classifiers, k,
                                   DFP=DFP,
                                   with_IH=with_IH,
                                   safe_k=safe_k,
                                   IH_rate=IH_rate)
        self.weighted = weighted
        self.name = 'K-Nearest Output Profiles (KNOP)'

    def fit(self, X, y):
        """Train the DS model by setting the KNN algorithm and
        pre-process the information required to apply the DS
        methods. In this case, the scores of the base classifiers for the dynamic selection dataset (DSEL)
        are pre-calculated to transform each sample in DSEL into an output profile.

         Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            containing the input data.

        y : array of shape = [n_samples]
            Class labels of each sample in X.

        Returns
        -------
        self
        """
        self._set_dsel(X, y)
        self.dsel_scores = self._preprocess_dsel_scores()
        self._fit_region_competence(self.dsel_scores, y, self.k)

        return self
    
    def estimate_competence(self, query):
        """In this method, the competence of the base classifiers is simply computed as the number of samples
        in the region of competence that it correctly classified. However, the region of competence here is
        estimated in the decision space using output profiles.

        Returns an array containing the level of competence estimated.
        The size of the array is equals to the size of the generated_pool of classifiers.

        Parameters
        ----------
        query : array of shape = [n_features]
                The test sample to be classified

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
                competences[clf_index] = np.sum(self.processed_dsel[idx_neighbors, clf_index])

        return competences.astype(dtype=int)

    def select(self, query):
        """Select the base classifiers for the classification of the query sample.

        Each base classifier can be selected more than once. The number of times a base classifier is selected (votes)
        is equals to the number of samples it correctly classified in the region of competence.

        Parameters
        ----------
        query : array of shape = [n_features]
                The test sample to be classified

        Returns
        -------
        votes : array containing the votes of the ensemble for each class
        """
        output_profile_query = self._output_profile_transform(query)
        weights = self.estimate_competence(output_profile_query.reshape(1, -1))

        # If all weights is equals to zero, it means that no classifier was selected. Hence, use all of them with equal
        # weights.
        if np.sum(weights) == 0:
            weights = np.ones(self.n_classifiers, dtype=int)

        votes = np.array([], dtype=int)
        for clf_idx, clf in enumerate(self.pool_classifiers):
            votes = np.hstack(
                (votes, np.ones(weights[clf_idx], dtype=int) * clf.predict(query)[0]))

        return votes

    def classify_instance(self, query):
        """Predicts the label of the corresponding query sample.

        The prediction is made aggregating the votes obtained by all selected base classifiers. The predicted label
        is the class that obtained the highest number of votes

        Parameters
        ----------
        query : array of shape = [n_features]
                The test sample to be classified.

        Returns
        -------
        predicted_label : Prediction of the ensemble for the input query.
        """
        votes = self.select(query)
        predicted_label = mode(votes)[0]

        return predicted_label
