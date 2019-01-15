# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.dcs.base import BaseDCS


class APriori(BaseDCS):
    """A Priori dynamic classifier selection.

    The A Priori method uses the probability of correct classification of a
    given base classifier :math:`c_{i}` for each neighbor :math:`x_{k}` for the
    competence level estimation. Base classifiers with a higher probability of
    correct classification have a higher competence level. Moreover, the method
    also weights the influence of each neighbor :math:`x_{k}` according to its
    Euclidean distance to the query sample. The closest neighbors have a higher
    influence on the competence level estimate.

    A single classifier is selected only if its competence level is
    significantly higher than that of the other base classifiers in the pool
    (higher than a pre-defined threshold). Otherwise, all classifiers i the
    pool are combined using the majority voting rule.

    Parameters
    ----------
    pool_classifiers : list of classifiers (Default = None)
        The generated_pool of classifiers trained for the corresponding
        classification problem. Each base classifiers should support the method
        "predict" and "predict_proba". If None, then the pool of classifiers is
        a bagging classifier.

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
    G. Giacinto and F. Roli, Methods for Dynamic Classifier Selection
    10th Int. Conf. on Image Anal. and Proc., Venice, Italy (1999), 659-664.

    Ko, Albert HR, Robert Sabourin, and Alceu Souza Britto Jr. "From dynamic
    classifier selection to dynamic ensemble selection."
    Pattern Recognition 41.5 (2008): 1718-1731.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection
    of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
    selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """

    def __init__(self, pool_classifiers=None, k=7, DFP=False, with_IH=False,
                 safe_k=None, IH_rate=0.30, selection_method='diff',
                 diff_thresh=0.1, random_state=None, knn_classifier='knn',
                 DSEL_perc=0.5):
        super(APriori, self).__init__(pool_classifiers=pool_classifiers, k=k,
                                      DFP=DFP, with_IH=with_IH, safe_k=safe_k,
                                      IH_rate=IH_rate,
                                      selection_method=selection_method,
                                      diff_thresh=diff_thresh,
                                      random_state=random_state,
                                      knn_classifier=knn_classifier,
                                      DSEL_perc=DSEL_perc)

    def fit(self, X, y):
        """Prepare the DS model by setting the KNN algorithm and
        pre-processing the information required to apply the DS
        method.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            Data used to fit the model.

        y : array of shape = [n_samples]
            class labels of each example in X.

        Returns
        -------
        self
        """
        super(APriori, self).fit(X, y)
        self._check_predict_proba()

        self.dsel_scores_ = self._preprocess_dsel_scores()
        return self

    def estimate_competence(self, query, neighbors, distances,
                            predictions=None):
        """estimate the competence of each base classifier :math:`c_{i}` for
        the classification of the query sample using the A Priori rule:

        The competence level is estimated based on the probability of correct
        classification of the base classifier :math:`c_{i}`, considering all
        samples in the region of competence. This method also weights the
        influence of each training sample according to its Euclidean distance
        to the query instance. The closest samples have a higher influence in
        the computation of the competence level. The competence level estimate
        is represented by the following equation:

        .. math::   \\delta_{i,j} = \\frac{\\sum_{k = 1}^{K}P(\\omega_{l} \\mid
         \\mathbf{x}_{k} \\in \\omega_{l},
          c_{i} )W_{k}}{\\sum_{k = 1}^{K}W_{k}}

        where :math:`\\delta_{i,j}` represents the competence level of
        :math:`c_{i}` for the classification of query.

        Parameters
        ----------
        query : array cf shape  = [n_samples, n_features]
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
        distances[distances == 0] = 1e-10
        dists_normalized = 1.0 / distances

        # Get the ndarray containing the scores obtained for the correct class
        # for each neighbor (and test sample)
        scores_target_class = self.dsel_scores_[neighbors, :,
                                                self.DSEL_target_[neighbors]]

        # Multiply the scores obtained for the correct class to the distances
        # of each corresponding neighbor
        scores_target_class *= np.expand_dims(dists_normalized, axis=2)

        # Sum the scores obtained for each neighbor and divide by the sum of
        # all distances
        competences = np.sum(scores_target_class, axis=1) / np.sum(
            dists_normalized, axis=1, keepdims=True)

        return competences
