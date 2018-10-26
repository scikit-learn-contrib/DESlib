import numpy as np

from deslib.des.probabilistic import BaseProbabilistic
from deslib.util import log_func


class Logarithmic(BaseProbabilistic):
    """ This method estimates the competence of the classifier based on
    the logarithmic difference between the supports obtained by the
    base classifier.

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
    B. Antosik, M. Kurzynski, New measures of classifier competence
    – heuristics and application to the design of
    multiple classifier systems., in: Computer recognition systems
    4., 2011, pp. 197–206.

    T.Woloszynski, M. Kurzynski, A measure of competence based on randomized
    reference classifier for dynamic ensemble selection, in: International
    Conference on Pattern Recognition (ICPR), 2010, pp. 4194–4197.
    """

    def __init__(self, pool_classifiers=None, k=None, DFP=False, with_IH=False,
                 safe_k=None,
                 IH_rate=0.30, mode='selection', random_state=None,
                 knn_classifier='knn', DSEL_perc=0.5):
        super(Logarithmic, self).__init__(pool_classifiers=pool_classifiers,
                                          k=k,
                                          DFP=DFP,
                                          with_IH=with_IH,
                                          safe_k=safe_k,
                                          IH_rate=IH_rate,
                                          mode=mode,
                                          random_state=random_state,
                                          knn_classifier=knn_classifier,
                                          DSEL_perc=DSEL_perc)

    def source_competence(self):
        """The source of competence C_src at the validation point
        :math:`\\mathbf{x}_{k}` is calculated by
        logarithm function in the support obtained by the base classifier.

        Returns
        ----------
        C_src : array of shape = [n_samples, n_classifiers]
            The competence source for each base classifier at each data point.
        """
        C_src = np.zeros((self.n_samples_, self.n_classifiers_))
        for clf_index in range(self.n_classifiers_):
            supports = self.dsel_scores_[:, clf_index, :]
            support_correct = supports[
                np.arange(self.n_samples_), self.DSEL_target_]

            C_src[:, clf_index] = log_func(self.n_classes_, support_correct)

        return C_src
