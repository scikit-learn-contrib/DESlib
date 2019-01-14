# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

from abc import abstractmethod, ABCMeta

import numpy as np

from deslib.des.base import BaseDES


class BaseProbabilistic(BaseDES):
    """Base class for a DS method based on the potential function model.
    All DS methods based on the Potential function should inherit from this
    class.

    Warning: This class should not be used directly.
    Use derived classes instead.

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

    mode : String (Default = "selection")
           Whether the technique will perform dynamic selection,
           dynamic weighting or an hybrid approach for classification.

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
    T.Woloszynski, M. Kurzynski, A probabilistic model of classifier competence
    for dynamic ensemble selection,
    Pattern Recognition 44 (2011) 2656–2668.

    L. Rastrigin, R. Erenstein, Method of collective recognition,
    Vol. 595, 1981, (in Russian).

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira.
    "Dynamic selection of classifiers—a comprehensive
    review." Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti,
    “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """
    __metaclass__ = ABCMeta

    def __init__(self, pool_classifiers=None, k=None, DFP=False, with_IH=False,
                 safe_k=None, IH_rate=0.30,
                 mode='selection', selection_threshold=None, random_state=None,
                 knn_classifier='knn', DSEL_perc=0.5):

        super(BaseProbabilistic, self).__init__(
            pool_classifiers=pool_classifiers,
            k=k,
            DFP=DFP,
            with_IH=with_IH,
            safe_k=safe_k,
            IH_rate=IH_rate,
            mode=mode,
            random_state=random_state,
            knn_classifier=knn_classifier,
            DSEL_perc=DSEL_perc)

        self.selection_threshold = selection_threshold

    def fit(self, X, y):
        """Train the DS model by setting the KNN algorithm and
        pre-processing the information required to apply the DS
        methods. In the case of probabilistic techniques, the source of
        competence (C_src) is calculated for each data point in DSEL in order
        to speed up the process during the testing phases.

        C_src is estimated with the source_competence() function that is
        overridden by each DS method based on this paradigm.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            Data used to fit the model.

        y : array of shape = [n_samples]
            class labels of each example in X.

        Returns
        -------
        self : object
            Returns self.
        """

        super(BaseProbabilistic, self).fit(X, y)

        if self.n_classes_ == 1:
            raise ValueError(
                "Error. This class does not accept one class datasets!")

        self._check_predict_proba()

        self.dsel_scores_ = self._preprocess_dsel_scores()

        # Pre process the source of competence for the entire DSEL,
        # making the method faster during generalization.
        self.C_src_ = self.source_competence()
        return self

    def estimate_competence(self,
                            query,
                            neighbors,
                            distances,
                            predictions=None):
        """estimate the competence of each base classifier :math:`c_{i}`
        using the source of competence :math:`C_{src}` and the potential
        function model. The source of competence :math:`C_{src}` for all
        data points in DSEL is already pre-computed in the fit() steps.

        .. math:: \\delta_{i,j} = \\frac{\\sum_{k=1}^{N}C_{src} \\:
          exp(-d (\\mathbf{x}_{k}, \\mathbf{x}_{q})^{2})}
          {exp( -d (\\mathbf{x}_{k}, \\mathbf{x}_{q})^{2} )}

        Parameters
        ----------
        query : array of shape = [n_samples, n_features]
                The test examples.

        neighbors : array of shale = [n_samples, n_neighbors]
            Indices of the k nearest neighbors according for each test sample.

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
        potential_dists = self.potential_func(distances)
        sum_potential = np.sum(potential_dists, axis=1)

        competences = np.einsum('ijk,ij->ik', self.C_src_[neighbors, :],
                                potential_dists)
        competences = competences / sum_potential.reshape(-1, 1)

        return competences

    def select(self, competences):
        """Selects the base classifiers that obtained a competence level higher
        than the predefined threshold. In this case, the threshold indicates
        the competence of the random classifier.

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

        # Set the threshold as the performance of the random classifier
        if self.selection_threshold is None:
            self.selection_threshold = 1.0 / self.n_classes_

        selected_classifiers = (competences > self.selection_threshold)
        # For the rows that are all False (i.e., no base classifier was
        # selected, select all classifiers (all True)
        selected_classifiers[~np.any(selected_classifiers, axis=1), :] = True

        return selected_classifiers

    @staticmethod
    def potential_func(dist):
        """Gaussian potential function to decrease the
        influence of the source of competence as the distance between
        :math:`\\mathbf{x}_{k}` and the query :math:`\\mathbf{x}_{q}`
        increases. The function is computed using the following equation:

        .. math:: potential = exp( -dist (\\mathbf{x}_{k},
         \\mathbf{x}_{q})^{2} )

        where dist represents the Euclidean distance between
        :math:`\\mathbf{x}_{k}` and :math:`\\mathbf{x}_{q}`

        Parameters
        ----------
        dist : array of shape = [self.n_samples]
               distance between the corresponding sample to the query

        Returns
        -------
        The result of the potential function for each value in (dist)
        """
        return np.exp(- (dist ** 2))

    @abstractmethod
    def source_competence(self):
        """ Method used to estimate the source of competence at each data
        point.

        Each DS technique based on this paradigm should define its
        computation of C_src

        Returns
        ----------
        C_src : array of shape = [n_samples, n_classifiers]
            The competence source for each base classifier at each data point.
        """
        pass
