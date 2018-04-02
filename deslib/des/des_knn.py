# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.des.base import DES
from deslib.util.diversity import negative_double_fault, Q_statistic, ratio_errors


class DESKNN(DES):
    """Dynamic ensemble Selection KNN (DES-KNN).

    This method selects an ensemble of classifiers taking into account the
    accuracy and diversity of the base classifiers. The k-NN algorithm is used to define the region of competence.
    The N most accurate classifiers in the region of competence are first selected.
    Then, the J more diverse classifiers from the N most accurate classifiers are selected to compose the ensemble.

    Parameters
    ----------
    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support the method "predict".

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

    mode : String (Default = "selection")
              whether the technique will perform dynamic selection, dynamic weighting
              or an hybrid approach for classification

    pct_accuracy : float (Default = 0.5)
        Percentage of base classifiers selected based on accuracy

    pct_diversity : float (Default = 0.3)
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
                 mode='selection',
                 pct_accuracy=0.5,
                 pct_diversity=0.3,
                 more_diverse=True,
                 metric='DF'):

        metric = metric.upper()
        super(DESKNN, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate,
                                     mode=mode)

        self.N = int(self.n_classifiers * pct_accuracy)
        self.J = int(np.ceil(self.n_classifiers * pct_diversity))
        self.more_diverse = more_diverse
        self.name = 'Dynamic Ensemble Selection-KNN (DES-KNN)'

        self.metric = metric
        self._validate_parameters()

        if metric == 'DF':
            self.diversity_func = negative_double_fault
        elif metric == 'Q':
            self.diversity_func = Q_statistic
        else:
            self.diversity_func = ratio_errors

    def estimate_competence(self, query, predictions=None):
        """estimate the competence level of each base classifier :math:`c_{i}` for
        the classification of the query sample.

        The competence is estimated using the accuracy and diversity criteria. First the classification accuracy
        of the base classifiers in the region of competence is estimated. Then the diversity of the base classifiers
        in the region of competence is estimated.

        The method returns two arrays: One containing the accuracy and the other the diversity of each base classifier.

        Parameters
        ----------
        query : array cf shape  = [n_features]
                The query sample

        predictions : array of shape = [n_samples, n_classifiers]
                      Contains the predictions of all base classifier for all samples in the query array

        Returns
        -------
        competences : Dictionary containing the accuracy and diversity estimates of all base classifiers for all
                      samples.

        competences : array of shape = [n_classifiers]
                      The competence level estimated for each base classifier

        diversity : array of shape = [n_classifiers]
                    The diversity estimated for each base classifier
        """
        _, idx_neighbors = self._get_region_competence(query)
        # calculate the classifiers mean accuracy for all samples/base classifier
        accuracy = np.mean(self.processed_dsel[idx_neighbors, :], axis=1)

        predicted_matrix = self.BKS_dsel[idx_neighbors, :]
        targets = self.DSEL_target[idx_neighbors]

        # TODO: try to optimize this part with numpy instead of for
        # Calculate the more_diverse matrix. It becomes computationally expensive
        # When the region of competence is high
        diversity = np.zeros((query.shape[0], self.n_classifiers))
        for sample_idx in range(query.shape[0]):
            for clf_index in range(self.n_classifiers):
                for clf_index2 in range(clf_index + 1, self.n_classifiers):
                    this_diversity = self.diversity_func(targets[sample_idx, :],
                                                         predicted_matrix[sample_idx, :, clf_index],
                                                         predicted_matrix[sample_idx, :, clf_index2])
                    diversity[sample_idx, clf_index] += this_diversity
                    diversity[sample_idx, clf_index2] += this_diversity

        competences = {'accuracy' : accuracy, 'diversity' : diversity}

        return competences

    def select(self, competences):
        """Select an ensemble containing the N most accurate ant the J most diverse classifiers for the classification
        of the query sample.

        Parameters
        ----------
        competences : Dictionary containing the accuracy and diversity estimates of all base classifiers for all
                      samples.

        Returns
        -------
        selected_classifiers : array of shape = [n_samples, n_classifiers]
                               Boolean matrix containing True if the base classifier is select, False otherwise

        """
        accuracy = competences['accuracy']
        diversity = competences['diversity']
        # Check if the accuracy and diversity arrays have the correct dimensionality.
        if accuracy.ndim < 2:
            accuracy = accuracy.reshape(1, -1)

        if diversity.ndim < 2:
            diversity = diversity.reshape(1, -1)

        # sort the array to remove the most accurate classifiers
        competent_indices = np.argsort(accuracy)[:, ::-1][:, 0:self.N]
        diversity_of_selected = diversity.take(competent_indices)

        # sort the remaining classifiers to select the most diverse ones
        if self.more_diverse:
            diversity_indices = np.argsort(diversity_of_selected, axis=1)[:, ::-1][:, 0:self.J]
        else:
            diversity_indices = np.argsort(diversity_of_selected, axis=1)[:, 0:self.J]

        # indices = competent_indices[diversity_indices]
        indices = competent_indices.take(diversity_indices)
        selected_classifiers = np.zeros(accuracy.shape, dtype=np.bool)
        # Setting the elements in indices (selected classifiers) to True
        selected_classifiers[np.arange(indices.shape[0])[:, None], indices] = True

        return selected_classifiers


    def _validate_parameters(self):
        """Check if the parameters passed as argument are correct.

        The diversity_func must be either ['DF', 'Q', 'RATIO']

        The values of N and J should be higher than 0, and N >= J
        ----------
        """
        if self.metric not in ['DF', 'Q', 'RATIO']:
            raise ValueError('Diversity metric must be one of the following values: "DF", "Q" or "Ratio"')

        if self.N <= 0 or self.J <= 0:
            raise ValueError("The values of N and J should be higher than 0"
                             "N = {}, J= {} " .format(self.N, self.J))
        if self.N < self.J:
            raise ValueError("The value of N should be greater or equals than J"
                             "N = {}, J= {} " .format(self.N, self.J))
