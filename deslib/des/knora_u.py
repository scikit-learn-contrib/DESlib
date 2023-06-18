# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.des.base import BaseDES


class KNORAU(BaseDES):
    """k-Nearest Oracles Union (KNORA-U).

    This method selects all classifiers that correctly classified at least
    one sample belonging to the region of competence of the query sample. Each
    selected classifier has a number of votes equals to the number of samples
    in the region of competence that it predicts the correct label. The votes
    obtained by all base classifiers are aggregated to obtain the final
    ensemble decision.

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
          :class:`KNNE` available on `deslib.utils.knne`

         - 'faiss' will use Facebook's Faiss similarity search through the
           class :class:`FaissKNNClassifier`

         - None, will use sklearn :class:`KNeighborsClassifier`.

    knn_metric : {'minkowski', 'cosine', 'mahalanobis'} (Default = 'minkowski')
        The metric used by the k-NN classifier to estimate distances.

        - 'minkowski' will use minkowski distance.

        - 'cosine' will use the cosine distance.

        - 'mahalanobis' will use the mahalonibis distance.

    knne : bool (Default=False)
        Whether to use K-Nearest Neighbor Equality (KNNE) for the region
        of competence estimation.

    DSEL_perc : float (Default = 0.5)
        Percentage of the input data used to fit DSEL.
        Note: This parameter is only used if the pool of classifier is None or
        unfitted.

    voting : {'hard', 'soft'}, default='hard'
            If 'hard', uses predicted class labels for majority rule voting.
            Else if 'soft', predicts the class label based on the argmax of
            the sums of the predicted probabilities, which is recommended for
            an ensemble of well-calibrated classifiers.

    n_jobs : int, default=-1
        The number of parallel jobs to run. None means 1 unless in
        a joblib.parallel_backend context. -1 means using all processors.
        Doesn’t affect fit method.

    References
    ----------
    Ko, Albert HR, Robert Sabourin, and Alceu Souza Britto Jr.
    "From dynamic classifier selection to dynamic ensemble
    selection." Pattern Recognition 41.5 (2008): 1718-1731.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira.
    "Dynamic selection of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
    selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.
    """

    def __init__(self, pool_classifiers=None, k=7, DFP=False, with_IH=False,
                 safe_k=None, IH_rate=0.30, random_state=None, voting='hard',
                 knn_classifier='knn', knn_metric='minkowski', knne=False,
                 DSEL_perc=0.5, n_jobs=-1):
        super(KNORAU, self).__init__(pool_classifiers, k,
                                     DFP=DFP,
                                     with_IH=with_IH,
                                     safe_k=safe_k,
                                     IH_rate=IH_rate,
                                     mode='weighting',
                                     random_state=random_state,
                                     knn_classifier=knn_classifier,
                                     knn_metric=knn_metric,
                                     knne=knne,
                                     DSEL_perc=DSEL_perc,
                                     n_jobs=n_jobs,
                                     voting=voting)

    def estimate_competence(self, competence_region, distances=None,
                            predictions=None):
        """The competence of the base classifiers is simply estimated as the
        number of samples in the region of competence that it
        correctly classified.

        This information is later used to determine the number of votes
        obtained for each base classifier.

        Parameters
        ----------
        competence_region : array of shape (n_samples, n_neighbors)
            Indices of the k nearest neighbors.

        distances : array of shape (n_samples, n_neighbors)
            Distances from the k nearest neighbors to the query.

        predictions : array of shape (n_samples, n_classifiers)
            Predictions of the base classifiers for all test examples.

        Returns
        -------
        competences : array of shape (n_samples, n_classifiers)
            Competence level estimated for each base classifier and test
            example.

        """
        competences = np.sum(self.DSEL_processed_[competence_region, :],
                             axis=1, dtype=float)

        return competences

    def select(self, competences):
        """Select the base classifiers for the classification of the query
        sample.

        Each base classifier can be selected more than once. The number of
        times a base classifier is selected (votes) is equals to the number
        of samples it correctly classified in the region of competence.

        Parameters
        ----------
        competences : array of shape (n_samples, n_classifiers)
            Competence level estimated for each base classifier and test
            example.

        Returns
        -------
        selected_classifiers : array of shape (n_samples, n_classifiers)
            Boolean matrix containing True if the base classifier is selected,
            False otherwise.
        """
        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        # Select classifier if it correctly classified at least one sample
        selected_classifiers = (competences > 0)

        # For the rows that are all False (i.e., no base classifier was
        # selected, select all classifiers (set all True)
        selected_classifiers[~np.any(selected_classifiers, axis=1), :] = True

        return selected_classifiers
