# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.dcs.base import DCS


class Rank(DCS):
    """Modified Classifier Rank.

    The modified classifier rank method evaluates the competence level of each individual classifiers
    and select the most competent one to predict the label of each test sample :math:`x`.
    The competence of each base classifier is calculated as the number of correctly classified samples,
    starting from the closest neighbor of :math:`x`. The classifier with the highest number of correctly classified
    samples is considered the most competent.

    The Rank method selects the base classifier presenting the highest competence level. In a case
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

    M. Sabourin, A. Mitiche, D. Thomas, G. Nagy, Classifier combination for handprinted digit recognition, International
    Conference on Document Analysis and Recognition (1993) 163–166.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """
    def __init__(self, pool_classifiers, k=7, DFP=False, with_IH=False, safe_k=None,
                 IH_rate=0.30,
                 selection_method='best',
                 diff_thresh=0.1, rng=np.random.RandomState()):

        super(Rank, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate,
                                   selection_method=selection_method,
                                   diff_thresh=diff_thresh, rng=rng)

        self.name = 'Modified Classifier Rank'

    def estimate_competence(self, query, predictions=None):
        """estimate the competence level of each base classifier :math:`c_{i}` for
        the classification of the query sample using the modified ranking scheme.
        The rank of the base classifier is estimated by the number of consecutive correctly classified samples
        in the defined region of competence.

        Parameters
        ----------
        query : array of shape = [n_samples, n_features]
                The test examples.

        predictions : array of shape = [n_samples, n_classifiers]
                      Predictions of the base classifiers for the test examples.

        Returns
        -------
        competences : array of shape = [n_samples, n_classifiers]
                      Competence level estimated for each base classifier and test example.
        """
        _, idx_neighbors = self._get_region_competence(query)
        results_neighbors = self.processed_dsel[idx_neighbors, :]

        # Get the shape of the vector in order to know the number of samples, base classifiers and neighbors considered.
        shape = results_neighbors.shape

        # add an row with zero for the case where the base classifier correctly classifies the whole neighborhood.
        # That way the search will always find a zero after comparing to self.K + 1
        addition = np.zeros((shape[0], shape[2]))
        results_neighbors = np.insert(results_neighbors, shape[1], addition, axis=1)
        competences = np.argmax(results_neighbors == 0, axis=1)

        return competences.astype(np.float32)
