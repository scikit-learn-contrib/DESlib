# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.des.base import BaseDES


class KNORAE(BaseDES):
    """k-Nearest Oracles Eliminate (KNORA-E).

    This method searches for a local Oracle, which is a base classifier
    that correctly classify all samples belonging to the region of competence
    of the test sample. All classifiers with a perfect performance in the
    region of competence are selected (local Oracles). In the case that no
    classifier achieves a perfect accuracy, the size of the competence region
    is reduced (by removing the farthest neighbor) and the performance of the
    classifiers are re-evaluated. The outputs of the selected ensemble of
    classifiers is combined using the majority voting scheme. If no base
    classifier is selected, the whole pool is used for classification.

    Parameters
    ----------
     pool_classifiers : list of classifiers (Default = None)
        The generated_pool of classifiers trained for the corresponding
        classification problem. Each base classifiers should support the method
        "predict". If None, then the pool of classifiers is a bagging
        classifier.

    k : int (Default = 7)
        Number of neighbors used to estimate the competence of the base
        classifiers.

    DFP : Boolean (Default = False)
        Determines if the dynamic frienemy pruning is applied.

    with_IH : Boolean (Default = False)
        Whether the hardness level of the region of competence is used to
        decide between using the DS algorithm or the KNN for classification of
        a given query sample.

    safe_k : int (default = None)
        The size of the indecision region.

    IH_rate : float (default = 0.3)
        Hardness threshold. If the hardness level of the competence region is
        lower than the IH_rate the KNN classifier is used. Otherwise, the DS
        algorithm is used for classification.


    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    knn_classifier : {'knn', 'faiss', None} (Default = 'knn')
         The algorithm used to estimate the region of competence:

         - 'knn' will use :class:`KNeighborsClassifier` from sklearn
         - 'faiss' will use Facebook's Faiss similarity search through the
           class :class:`FaissKNNClassifier`
         - None, will use sklearn :class:`KNeighborsClassifier`.

    DSEL_perc : float (Default = 0.5)
        Percentage of the input data used to fit DSEL.
        Note: This parameter is only used if the pool of classifier is None or
        unfitted.

    References
    ----------
    Ko, Albert HR, Robert Sabourin, and Alceu Souza Britto Jr.
    "From dynamic classifier selection to dynamic ensemble
    selection." Pattern Recognition 41.5 (2008): 1718-1731.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection
    of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
    selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """

    def __init__(self,
                 pool_classifiers=None,
                 k=7,
                 DFP=False,
                 with_IH=False,
                 safe_k=None,
                 IH_rate=0.30,
                 random_state=None,
                 knn_classifier='knn',
                 DSEL_perc=0.5):

        super(KNORAE, self).__init__(pool_classifiers=pool_classifiers,
                                     k=k,
                                     DFP=DFP,
                                     with_IH=with_IH,
                                     safe_k=safe_k,
                                     IH_rate=IH_rate,
                                     random_state=random_state,
                                     knn_classifier=knn_classifier,
                                     DSEL_perc=DSEL_perc)

    def estimate_competence(self, query, neighbors, distances=None,
                            predictions=None):
        """Estimate the competence of the base classifiers. In the case of
        the KNORA-E technique, the classifiers are only considered competent
        when they achieve a 100% accuracy in the region of competence.
        For each base, we estimate the maximum size of the region of competence
        that it is a local oracle. The competence level estimate is then the
        maximum size of the region of competence that the corresponding base
        classifier is considered a local Oracle.

        Parameters
        ----------
        query : array of shape = [n_samples, n_features]
                The test examples.

        neighbors : array of shale = [n_samples, n_neighbors]
            Indices of the k nearest neighbors according for each test sample

        distances : array of shale = [n_samples, n_neighbors]
            Distances of the k nearest neighbors according for each test
            sample.

        predictions : array of shape = [n_samples, n_classifiers]
            Predictions of the base classifiers for all test examples.

        Returns
        -------
        competences : array of shape = [n_samples, n_classifiers]
            Competence level estimated for each base classifier and test
            example.
        """
        results_neighbors = self.DSEL_processed_[neighbors, :]

        # Get the shape of the vector in order to know the number of samples,
        # base classifiers and neighbors considered.
        shape = results_neighbors.shape

        # add an row with zero for the case where the base classifier correctly
        # classifies the whole neighborhood. That way the search will always
        # find a zero after comparing to self.K + 1 and will return self.K
        # as the Competence level estimate (correctly classified the whole
        # neighborhood)
        addition = np.zeros((shape[0], shape[2]))
        results_neighbors = np.insert(results_neighbors, shape[1], addition,
                                      axis=1)

        # Look for the first occurrence of a zero in the processed predictions
        # (first misclassified sample). The np.argmax can be used here, since
        # in case of multiple occurrences of the maximum values, the indices_
        # corresponding to the first occurrence are returned.
        competences = np.argmax(results_neighbors == 0, axis=1)

        return competences.astype(np.float)

    def select(self, competences):
        """Selects all base classifiers that obtained a local accuracy of 100%
        in the region of competence (i.e., local oracle). In the case that no
        base classifiers obtain 100% accuracy, the size of the region of
        competence is reduced and the search for the local oracle is restarted.

        Notes
        ------
        Instead of re-applying the method several times (reducing the size of
        the region of competence), we compute the number of consecutive correct
        classification of each base classifier starting from the closest
        neighbor to the more distant in the estimate_competence function.
        The number of consecutive correct classification represents the size
        of the region of competence in which the corresponding base classifier
        is an Local Oracle. Then, we select all base classifiers with the
        maximum value for the number of consecutive correct classification.
        This speed up the selection process.

        Parameters
        ----------
        competences : array of shape = [n_samples, n_classifiers]
            Competence level estimated for each base classifier and test
            example.

        Returns
        -------
        selected_classifiers : array of shape = [n_samples, n_classifiers]
            Boolean matrix containing True if the base classifier is selected,
            False otherwise.

        """
        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        # Checks which was the max value for each sample
        # (i.e., the maximum number of consecutive predictions)
        max_value = np.max(competences, axis=1)

        # Select all base classifiers with the maximum number of
        #  consecutive correct predictions for each sample.
        selected_classifiers = (
                    competences == max_value.reshape(competences.shape[0], -1))

        return selected_classifiers
