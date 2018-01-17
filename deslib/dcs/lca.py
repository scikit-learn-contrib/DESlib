# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.dcs.base import DCS


class LCA(DCS):
    """Local Classifier Accuracy (LCA).

    Evaluates the competence level of each individual classifiers and
    select the most competent one to predict the label of each test sample.
    The competence of each base classifier is calculated based on its local 
    accuracy with respect to some output class. Consider a classifier that assigns
    a test sample to class Ci. The competence is estimated by the percentage of the local training
    samples assigned to class Ci by this classifier that have been correctly labeled.
    

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
    Woods, Kevin, W. Philip Kegelmeyer, and Kevin Bowyer. "Combination of multiple classifiers
    using local accuracy estimates." IEEE transactions on pattern analysis and machine intelligence
    19.4 (1997): 405-410.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection of classifiers—a comprehensive
    review." Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """
    def __init__(self, pool_classifiers, k=7, DFP=False, with_IH=False, safe_k=None, IH_rate=0.30,
                 selection_method='best', diff_thresh=0.1, rng=np.random.RandomState()):

            super(LCA, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate,
                                      selection_method=selection_method,
                                      diff_thresh=diff_thresh,
                                      rng=rng)

            self.name = 'Local Classifier Accuracy (LCA)'

    def estimate_competence(self, query):
        """estimate the competence of each base classifier ci
        the classification of the query sample using the local class accuracy method.


        In this algorithm the K-Nearest Neighbors of the test sample are estimated. Then, the
        local accuracy of the base classifiers is estimated by its classification accuracy taking into account
        only the samples belonging to the class wl in this neighborhood.


        Returns an array containing the level of competence estimated using the LCA method
        for each base classifier. The size of the array is equals to the size of the pool of classifiers.

        Parameters
        ----------
        query : array cf shape  = [n_features]
                The query sample
        Returns
        -------
        competences : array of shape = [n_classifiers]
                      The competence level estimated for each base classifier
        """
        dists, idx_neighbors = self._get_region_competence(query)
        competences = np.zeros(self.n_classifiers)

        for clf_index, clf in enumerate(self.pool_classifiers):
            # Check if the dynamic frienemy pruning (DFP) should be used used
            if self.DFP_mask[clf_index]:
                result = []
                predicted_label = clf.predict(query)[0]
                for index in idx_neighbors:
                    # Get only neighbors from the same class as predicted by the
                    # classifier (clf) to form the region of competence
                    if self.DSEL_target[index] == predicted_label:
                        result.append(self.processed_dsel[index][clf_index])

                if len(result) == 0:
                    competences[clf_index] = 0.0
                else:
                    competences[clf_index] = np.mean(result)

        return competences
