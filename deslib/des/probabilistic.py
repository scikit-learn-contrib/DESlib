# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

from abc import abstractmethod, ABCMeta

import numpy as np

from deslib.des.base import DES
from deslib.util.prob_functions import entropy_func, ccprmod, log_func, exponential_func, min_difference


class Probabilistic(DES):
    """Base class for a DS method based on the potential function model.
    ALL DS methods based on the Potential function should inherit from this class

    Warning: This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support methods "predict" and "predict_proba".

    k : int (Default = None)
        Number of neighbors used to estimate the competence of the base classifiers. If k = None, the whole dynamic
        selection dataset is used, and the influence of each sample is based on its distance to the query.

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
           Whether the technique will perform dynamic selection,
           dynamic weighting or an hybrid approach for classification.


    References
    ----------
    T.Woloszynski, M. Kurzynski, A probabilistic model of classifier competence for dynamic ensemble selection,
    Pattern Recognition 44 (2011) 2656–2668.

    L. Rastrigin, R. Erenstein, Method of collective recognition, Vol. 595, 1981, (in Russian).

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection of classifiers—a comprehensive
    review." Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """
    __metaclass__ = ABCMeta

    def __init__(self, pool_classifiers, k=None, DFP=False, with_IH=False, safe_k=None, IH_rate=0.30,
                 mode='selection', selection_threshold=None):

        super(Probabilistic, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k,
                                            IH_rate=IH_rate,
                                            mode=mode)
        self._check_predict_proba()

        self.C_src = None
        self.selection_threshold = selection_threshold

    def fit(self, X, y):
        """Train the DS model by setting the KNN algorithm and
        pre-processing the information required to apply the DS
        methods. In the case of probabilistic techniques, the source of competence (C_src)
        is calculated for each data point in DSEL in order to speed up the process during the
        testing phases.

        C_src is estimated with the source_competence() function that is overridden by each DS method
        based on this paradigm

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

        y_ind = self.setup_label_encoder(y)
        self._set_dsel(X, y_ind)
        if self.k is None:
            self.k = self.n_samples

        self._fit_region_competence(X, y_ind, self.n_samples)
        # Pre process the scores in DSEL (it is required only for the source of competence estimation
        # Maybe I should not keep this matrix in order to reduce memory requirement.
        self.dsel_scores = self._preprocess_dsel_scores()

        # Pre process the source of competence for the entire DSEL, making the method faster during generalization.
        self.C_src = self.source_competence()
        return self

    def estimate_competence(self, query, predictions=None):
        """estimate the competence of each base classifier :math:`c_{i}` using the source of competence :math:`C_{src}`
        and the potential function model. The source of competence :math:`C_{src}` for all data points in DSEL
        is already pre-computed in the fit() steps.

        .. math:: \\delta_{i,j} = \\frac{\\sum_{k=1}^{N}C_{src} \\: exp( -d (\\mathbf{x}_{k}, \\mathbf{x}_{q})^{2} )}
            {exp( -d (\\mathbf{x}_{k}, \\mathbf{x}_{q})^{2} )}

        Parameters
        ----------
        query : array of shape = [n_samples, n_features]
                The test examples.

        predictions : array of shape = [n_samples, n_classifiers]
                      Predictions of the base classifiers for all test examples.
        Returns
        -------
        competences : array of shape = [n_samples, n_classifiers]
                      Competence level estimated for each base classifier and test example.
        """
        dists, idx_neighbors = self._get_region_competence(query)
        sorted_neighbors = np.argsort(idx_neighbors, axis=1)
        dists_organized = np.take(dists, sorted_neighbors)

        potential_dists = self.potential_func(dists_organized)
        sum_potential = np.sum(potential_dists, axis=1)

        competences = self.C_src[np.newaxis, :, :] * potential_dists[:, :, np.newaxis]
        competences = competences.sum(axis=1)/sum_potential.reshape(-1, 1)

        return competences

    def select(self, competences):
        """Selects the base classifiers that obtained a competence level higher than the predefined threshold.
        In this case, the threshold indicates the competence of the random classifier.

        Parameters
        ----------
        competences : array of shape = [n_samples, n_classifiers]
                      Competence level estimated for each base classifier and test example.

        Returns
        -------
        selected_classifiers : array of shape = [n_samples, n_classifiers]
                               Boolean matrix containing True if the base classifier is select, False otherwise.

        """
        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        # Set the threshold as the performance of the random classifier
        if self.selection_threshold is None:
            self.selection_threshold = 1.0/self.n_classes

        selected_classifiers = (competences > self.selection_threshold)
        # For the rows that are all False (i.e., no base classifier was selected, select all classifiers (all True)
        selected_classifiers[~np.any(selected_classifiers, axis=1), :] = True

        return selected_classifiers

    @staticmethod
    def potential_func(dist):
        """Gaussian potential function to decrease the
        influence of the source of competence as the distance between :math:`\mathbf{x}_{k}` and the query
        :math:`\mathbf{x}_{q}` increases. The function is computed using the following equation:

        .. math:: potential = exp( -dist (\mathbf{x}_{k}, \mathbf{x}_{q})^{2} )

        where dist represents the Euclidean distance between :math:`\mathbf{x}_{k}` and :math:`\mathbf{x}_{q}`

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
        """ Method used to estimate the source of competence at each data point.

        Each DS technique based on this paradigm should define its computation of C_src

        Returns
        ----------
        C_src : array of shape = [n_samples, n_classifiers]
                The competence source for each base classifier at each data point.
        """
        pass


class Logarithmic(Probabilistic):
    """ This method estimates the competence of the classifier based on the logarithmic
    difference between the supports obtained by the base classifier.

    Parameters
    ----------
    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support methods "predict" and "predict_proba".

    k : int (Default = None)
        Number of neighbors used to estimate the competence of the base classifiers. If k = None, the whole dynamic
        selection dataset is used, and the influence of each sample is based on its distance to the query.

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
           Whether the technique will perform dynamic selection,
           dynamic weighting or an hybrid approach for classification.

    References
    ----------
    B. Antosik, M. Kurzynski, New measures of classifier competence – heuristics and application to the design of
    multiple classifier systems., in: Computer recognition systems 4., 2011, pp. 197–206.

    T.Woloszynski, M. Kurzynski, A measure of competence based on randomized reference classifier for dynamic
    ensemble selection, in: International Conference on Pattern Recognition (ICPR), 2010, pp. 4194–4197.
    """
    def __init__(self, pool_classifiers, k=None, DFP=False, with_IH=False, safe_k=None, IH_rate=0.30, mode='selection'):

        super(Logarithmic, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate,
                                          mode=mode)
        self.name = "DES-Logarithmic"

    def source_competence(self):
        """The source of competence C_src at the validation point :math:`\mathbf{x}_{k}` is calculated by
        logarithm function in the support obtained by the base classifier.

        Returns
        ----------
        C_src : array of shape = [n_samples, n_classifiers]
                The competence source for each base classifier at each data point.
        """
        C_src = np.zeros((self.n_samples, self.n_classifiers))
        for clf_index in range(self.n_classifiers):
            supports = self.dsel_scores[:, clf_index, :]
            support_correct = supports[np.arange(self.n_samples), self.DSEL_target]

            C_src[:, clf_index] = log_func(self.n_classes, support_correct)

        return C_src


class Exponential(Probabilistic):
    """The source of competence C_src at the validation point :math:`\mathbf{x}_{k}` is a product of two factors:
    The absolute value of
    the competence and the sign. The value of the source competence is inverse proportional to the normalized entropy
    of its supports vector. The sign of competence is simply determined by correct/incorrect classification
    of :math:`\mathbf{x}_{k}` [1].

    The influence of each sample :math:`\mathbf{x}_{k}` is defined according to a Gaussian function model[2].
    Samples that are closer to the query have a higher influence in the competence estimation.

    Parameters
    ----------
    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support methods "predict" and "predict_proba".

    k : int (Default = None)
        Number of neighbors used to estimate the competence of the base classifiers. If k = None, the whole dynamic
        selection dataset is used, and the influence of each sample is based on its distance to the query.

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
           Whether the technique will perform dynamic selection,
           dynamic weighting or an hybrid approach for classification.

    References
    ----------
    [1] B. Antosik, M. Kurzynski, New measures of classifier competence – heuristics and application to the design of
    multiple classifier systems., in: Computer recognition systems 4., 2011, pp. 197–206.

    [2] Woloszynski, Tomasz, and Marek Kurzynski. "A probabilistic model of classifier competence
    for dynamic ensemble selection." Pattern Recognition 44.10 (2011): 2656-2668.

    """
    def __init__(self, pool_classifiers, k=None, DFP=False, safe_k=None, with_IH=False, IH_rate=0.30,
                 mode='selection'):

        super(Exponential, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate,
                                          mode=mode)

        self.selection_threshold = 0
        self.name = "DES-Exponential"

    def source_competence(self):
        """The source of competence C_src at the validation point :math:`\mathbf{x}_{k}` is a product of two factors:
        The absolute value of  the competence and the sign. The value of the source competence is inverse proportional
        to the normalized entropy of its supports vector.The sign of competence is simply determined by
        correct/incorrect classification of the instance :math:`\mathbf{x}_{k}`.

        Returns
        ----------
        C_src : array of shape = [n_samples, n_classifiers]
                The competence source for each base classifier at each data point.
        """
        C_src = np.zeros((self.n_samples, self.n_classifiers))
        for clf_index in range(self.n_classifiers):
            supports = self.dsel_scores[:, clf_index, :]
            support_correct = supports[np.arange(self.n_samples), self.DSEL_target]

            C_src[:, clf_index] = exponential_func(self.n_classes, support_correct)
        return C_src


class RRC(Probabilistic):
    """DES technique based on the Randomized Reference Classifier method (DES-RRC).

    Parameters
    ----------
    pool_classifiers : type, the generated_pool of classifiers trained for the corresponding
    classification problem.

    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support methods "predict" and "predict_proba".

    k : int (Default = None)
        Number of neighbors used to estimate the competence of the base classifiers. If k = None, the whole dynamic
        selection dataset is used, and the influence of each sample is based on its distance to the query.

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
           Whether the technique will perform dynamic selection,
           dynamic weighting or an hybrid approach for classification.

    References
    ----------
    Woloszynski, Tomasz, and Marek Kurzynski. "A probabilistic model of classifier competence
    for dynamic ensemble selection." Pattern Recognition 44.10 (2011): 2656-2668.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """
    def __init__(self, pool_classifiers, k=None, DFP=False, with_IH=False, safe_k=None, IH_rate=0.30, mode='selection'):

        super(RRC, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate,
                                  mode=mode)
        self.name = "DES-RRC"
        self.selection_threshold = None

    def source_competence(self):
        """
        Calculates the source of competence using the randomized reference classifier (RRC) method.

        The source of competence C_src at the validation point :math:`\mathbf{x}_{k}` calculated using the
        probabilistic model based on the supports obtained by the base classifier and randomized reference
        classifier (RRC) model. The probabilistic modeling of the classifier competence is calculated using
        the ccprmod function.

        Returns
        ----------
        C_src : array of shape = [n_samples, n_classifiers]
                The competence source for each base classifier at each data point.
        """
        c_src = np.zeros((self.n_samples, self.n_classifiers))

        for clf_index in range(self.n_classifiers):
            # Get supports for all samples in DSEL
            supports = self.dsel_scores[:, clf_index, :]
            c_src[:, clf_index] = ccprmod(supports, self.DSEL_target)

        return c_src


class DESKL(Probabilistic):
    """Dynamic Ensemble Selection-Kullback-Leibler divergence (DES-KL).

    This method estimates the competence of the classifier from the
    information theory perspective. The competence of the base classifiers
    is calculated as the KL divergence between the vector of class supports
    produced by the base classifier and the outputs of a random classifier (RC).
    RC = 1/L, L being the number of classes in the problem. Classifiers with a
    competence higher than the competence of the random classifier is selected.

    Parameters
    ----------
    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support methods "predict" and "predict_proba".

    k : int (Default = None)
        Number of neighbors used to estimate the competence of the base classifiers. If k = None, the whole dynamic
        selection dataset is used, and the influence of each sample is based on its distance to the query.

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
           Whether the technique will perform dynamic selection,
           dynamic weighting or an hybrid approach for classification.

    References
    ----------
    Woloszynski, Tomasz, et al. "A measure of competence based on random classification
    for dynamic ensemble selection." Information Fusion 13.3 (2012): 207-213.

    Woloszynski, Tomasz, and Marek Kurzynski. "A probabilistic model of classifier competence
    for dynamic ensemble selection." Pattern Recognition 44.10 (2011): 2656-2668.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """
    def __init__(self, pool_classifiers, k=None, DFP=False, with_IH=False, safe_k=None, IH_rate=0.30, mode='selection'):

        super(DESKL, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate,
                                    mode=mode)
        self.selection_threshold = 0.0
        self.name = 'DES-Kullback-Leibler (DES-KL)'

    def source_competence(self):
        """Calculates the source of competence using the KL divergence method.

        The source of competence C_src at the validation point :math:`\mathbf{x}_{k}` is calculated by the KL divergence
        between the vector of class supports produced by the base classifier and the outputs of a random classifier (RC)
        RC = 1/L, L being the number of classes in the problem. The value of C_src is negative if the base classifier
        misclassified the instance :math:`\mathbf{x}_{k}`.

        Returns
        ----------
        C_src : array of shape = [n_samples, n_classifiers]
                The competence source for each base classifier at each data point.
        """

        C_src = np.zeros((self.n_samples, self.n_classifiers))
        for clf_index in range(self.n_classifiers):
            supports = self.dsel_scores[:, clf_index, :]
            is_correct = self.processed_dsel[:, clf_index]
            C_src[:, clf_index] = entropy_func(self.n_classes, supports, is_correct)

        return C_src


class MinimumDifference(Probabilistic):
    """
    Computes the competence level of the classifiers based on the difference between the support obtained by each class.
    The competence level at a data point :math:`\mathbf{x}_{k}` is equal to the minimum difference between the
    support obtained to the correct class and the support obtained for different classes.

    The influence of each sample xk is defined according to a Gaussian function model[2]. Samples that are closer to
    the query have a higher influence in the competence estimation.

    Parameters
    ----------
    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support methods "predict" and "predict_proba".

    k : int (Default = None)
        Number of neighbors used to estimate the competence of the base classifiers. If k = None, the whole dynamic
        selection dataset is used, and the influence of each sample is based on its distance to the query.

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
           Whether the technique will perform dynamic selection,
           dynamic weighting or an hybrid approach for classification.

    References
    ----------
    [1] B. Antosik, M. Kurzynski, New measures of classifier competence – heuristics and application to the design of
    multiple classifier systems., in: Computer recognition systems 4., 2011, pp. 197–206.

    [2] Woloszynski, Tomasz, and Marek Kurzynski. "A probabilistic model of classifier competence
    for dynamic ensemble selection." Pattern Recognition 44.10 (2011): 2656-2668.

    """
    def __init__(self, pool_classifiers, k=None, DFP=False, with_IH=False, safe_k=None, IH_rate=0.30,
                 mode='selection'):
        super(MinimumDifference, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k,
                                                IH_rate=IH_rate, mode=mode)

        # Threshold is 0 since incompetent classifiers should have a negative competence level
        self.selection_threshold = 0.0
        self.name = "DES-Minimum Difference (DES-MD)"

    def source_competence(self):
        """Calculates the source of competence using the Minimum Difference method.

        The source of competence C_src at the validation point :math:`\mathbf{x}_{k}` calculated by the
        Minimum Difference between the supports obtained to the correct class and the support obtained by
        the other classes

        Returns
        ----------
        C_src : array of shape = [n_samples, n_classifiers]
                The competence source for each base classifier at each data point.
        """
        C_src = np.zeros((self.n_samples, self.n_classifiers))
        for clf_index in range(self.n_classifiers):
            supports = self.dsel_scores[:, clf_index, :]
            C_src[:, clf_index] = min_difference(supports, self.DSEL_target)

        return C_src
