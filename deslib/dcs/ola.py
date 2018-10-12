# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.dcs.base import BaseDCS


class OLA(BaseDCS):
    """Overall Classifier Accuracy (OLA).

    The OLA method evaluates the competence level of each individual
    classifiers and select the most competent one to predict the label of each
    test sample x. The competence of each base classifier is calculated as its
    classification accuracy in the neighborhood of x (region of competence).

    The LCA method selects the base classifier presenting the highest
    competence level. In a case where more than one base classifier achieves
    the same competence level, the one that was evaluated first is selected.
    The selection methodology can be modified by changing the hyper-parameter
    selection_method.


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

    selection_method : String (Default = "best")
        Determines which method is used to select the base classifier after
        the competences are estimated.

    diff_thresh : float (Default = 0.1)
        Threshold to measure the difference between the competence level of the
        base classifiers for the random and diff selection schemes. If the
        difference is lower than the threshold, their performance are
        considered equivalent.

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
    Woods, Kevin, W. Philip Kegelmeyer, and Kevin Bowyer. "Combination of
    multiple classifiers using local accuracy estimates." IEEE transactions on
    pattern analysis and machine intelligence 19.4 (1997): 405-410.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection
    of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
    selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """

    def __init__(self, pool_classifiers=None, k=7, DFP=False, with_IH=False,
                 safe_k=None, IH_rate=0.30, selection_method='best',
                 diff_thresh=0.1, random_state=None, knn_classifier='knn',
                 DSEL_perc=0.5):
        super(OLA, self).__init__(pool_classifiers=pool_classifiers, k=k,
                                  DFP=DFP, with_IH=with_IH, safe_k=safe_k,
                                  IH_rate=IH_rate,
                                  selection_method=selection_method,
                                  diff_thresh=diff_thresh,
                                  random_state=random_state,
                                  knn_classifier=knn_classifier,
                                  DSEL_perc=DSEL_perc)

    def estimate_competence(self, query, neighbors, distances=None,
                            predictions=None):
        """estimate the competence level of each base classifier :math:`c_{i}`
        for the classification of the query sample.

        The competences for each base classifier :math:`c_{i}` is estimated by
        its classification accuracy considering the k-Nearest Neighbors (region
        of competence). The competence level estimate is represented by the
        following equation:

        .. math:: \\delta_{i,j} = \\frac{1}{K}\\sum_{k = 1}^{K}
            P(\\omega_{l} \\mid \\mathbf{x}_{k} \\in \\omega_{l}, c_{i} )

        where :math:`\\delta_{i,j}` represents the competence level of
        :math:`c_{i}` for the classification of query.

        Parameters
        ----------
        query : array of shape  = [n_samples, n_features]
            The test examples.

        neighbors : array of shale = [n_samples, n_neighbors]
            Indices of the k nearest neighbors according for each test sample

        distances : array of shale = [n_samples, n_neighbors]
            Distances of the k nearest neighbors according for each test sample

        predictions : array of shape = [n_samples, n_classifiers]
            Predictions of the base classifiers for the test examples.

        Returns
        -------
        competences : array of shape = [n_samples, n_classifiers]
            Competence level estimated for each base classifier and test
            example.
        """
        competences = np.mean(self.DSEL_processed_[neighbors, :], axis=1)

        return competences
