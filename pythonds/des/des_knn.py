# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from pythonds.des.base import DES

from pythonds.util.diversity import double_fault, Q_statistic, ratio_errors


class DESKNN(DES):
    """Dynamic ensemble Selection KNN (DES-KNN).
    This method selects an ensemble of classifiers taking into account the
    accuracy and more_diverse of the base classifiers. First the most accurate classifiers
    are selected. Next, the most diverse classifiers, in relation to the selected classifiers,
    are added to the ensemble

    Parameters
    ----------
    pool_classifiers : type, the generated_pool of classifiers trained for the corresponding
    classification problem.

    k : int (Default = 5)
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
           Determines the type of KNN algorithm that is used. Set to true for the A-KNN method.

    mode : String (Default = "selection")
              whether the technique will perform dynamic selection, dynamic weighting
              or an hybrid approach for classification

    N : float (Default = 0.3)
        Percentage of base classifiers selected based on accuracy

    J : float (Default = 0.3)
        Percentage of base classifiers selected based n diversity

    more_diverse : Boolean (Default = True)
        Whether we select the most or the least diverse classifiers to add to the pre-selected ensemble

    metric : String (Default = 'df')
            Diversity diversity_func used to estimate the diversity of the base classifiers. Can
            be either the double fault (df), Q-statistics (Q), or error correlation (corr)

    References
    ----------
    Soares, R. G., Santana, A., Canuto, A. M., & de Souto, M. C. P. "Using accuracy and more_diverse to select
    classifiers to build ensembles." International Joint Conference on Neural Networks (IJCNN)., 2006.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.
    """

    def __init__(self, pool_classifiers, k=7, DFP=False, with_IH=False, safe_k=None,
                 IH_rate=0.30,
                 aknn=False,
                 mode='selection',
                 N=0.3,
                 J=0.3,
                 more_diverse=False,
                 metric='DF'):

        metric = metric.upper()
        super(DESKNN, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate,
                                     aknn=aknn, mode=mode)

        self.N = int(self.n_classifiers * N)
        self.J = int(self.n_classifiers * J)
        self.more_diverse = more_diverse
        self.name = 'Dynamic Ensemble Selection-KNN (DES-KNN)'
        self.metric = metric
        if metric == 'DF':
            self.diversity_func = double_fault
        elif metric == 'Q':
            self.diversity_func = Q_statistic
        else:
            self.diversity_func = ratio_errors

        self._validate_inputs()

    def estimate_competence(self, query):
        """get the competence estimates of each base classifier ci for the classification of the query sample x.

        The competence is estimated using the accuracy and diversity criteria. First the classification accuracy
        of the base classifiers in the region of competence is estimated. Then the diversity of the base classifiers
        in the region of competence is estimated.

        The method returns two arrays: One containing the accuracy and the other the diversity of each base classifier.

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        Returns
        -------
        competences : array = [n_classifiers] containing the competence level estimated
        for each base classifier

        diversity : array = [n_classifiers] containing the diversity estimated
        for each base classifier
        """
        dists, idx_neighbors = self._get_region_competence(query)
        competences = np.zeros(self.n_classifiers)
        predicted_matrix = np.zeros((self.k, self.n_classifiers))
        for clf_index in range(self.n_classifiers):
            result = [self.processed_dsel[index][clf_index] for index in idx_neighbors]
            predicted_matrix[:, clf_index] = result
            competences[clf_index] = np.mean(result)

        # Calculate the more_diverse matrix. It becomes computationally expensive
        # When the region of competence is high
        targets = [self.DSEL_target[index] for index in idx_neighbors]
        diversity = np.zeros(self.n_classifiers)

        # TODO change the for loop to calculate half of the matrix (symmetric)
        for clf_index in range(self.n_classifiers):
            for clf_index2 in range(self.n_classifiers):
                if clf_index != clf_index2:

                    diversity[clf_index] += \
                        double_fault(targets, predicted_matrix[:, clf_index], predicted_matrix[:, clf_index2])

        return competences, diversity

    def select(self, query):
        """Select an ensemble containing the N most accurate ant the J most diverse classifiers for the classification
        of the query

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        Returns
        -------
        indices : the indices of the selected base classifiers

        competences : array = [n_classifiers] containing the competence level estimated
        for each base classifier
        """
        # sort the array to remove the most accurate classifiers
        competences, diversity = self.estimate_competence(query)
        accuracy_indices = np.argsort(competences)[::-1][0:self.N]

        # delete the classifiers that were already selected
        diversity = np.delete(diversity, accuracy_indices)

        # sort the remaining classifiers to select the most diverse ones
        # Since DF is used, the more_diverse is minimized
        if self.more_diverse:
            diversity_indices = np.argsort(diversity)[::-1][0:self.J]
        else:
            diversity_indices = np.argsort(diversity)[0:self.J]

        indices = np.hstack((accuracy_indices, diversity_indices))
        return indices

    def classify_instance(self, query):
        """Predicts the label of the corresponding query sample.
        Returns the predicted label.

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        Returns
        -------
        predicted_label: The predicted label of the query
        """
        indices = self.select(query)
        predicted_label = self.majority_voting(indices, query)
        return predicted_label

    def _validate_inputs(self):
        """Check if the parameters passed as argument are correct.

        The diversity_func must be either ['DF', 'Q', 'RATIO']

        The values of N and J should be higher than 0, and N >= J
        ----------
        """
        assert self.metric in ['DF', 'Q', 'RATIO']

        if self.N <= 0 or self.J <= 0:
            raise ValueError("The values of N and J should be higher than 0"
                             "N = %s, J= %s " % (self.N, self.J))
        if self.N < self.J:
            raise ValueError("The value of N should be greater or equals than J"
                             "N = %s, J= %s " % (self.N, self.J))
