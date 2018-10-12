# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.des.base import BaseDES


class KNOP(BaseDES):
    """k-Nearest Output Profiles (KNOP).

    This method selects all classifiers that correctly classified at least
    one sample belonging to the region of competence of the query sample.
    In this case, the region of competence is estimated using the decisions
    of the base classifier (output profiles). Thus, the similarity between
    the query and the validation samples are measured in the decision space
    rather than the feature space. Each selected classifier has a number of
    votes equals to the number of samples in the region of competence that
    it predicts the correct label. The votes obtained by all
    base classifiers are aggregated to obtain the final ensemble decision.

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
    Cavalin, Paulo R., Robert Sabourin, and Ching Y. Suen.
    "LoGID: An adaptive framework combining local and global
    incremental learning for dynamic selection of ensembles of HMMs."
    Pattern Recognition 45.9 (2012): 3544-3556.

    Cavalin, Paulo R., Robert Sabourin, and Ching Y. Suen.
    "Dynamic selection approaches for multiple classifier
    systems." Neural Computing and Applications 22.3-4 (2013): 673-688.

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

    def __init__(self, pool_classifiers=None, k=7, DFP=False, with_IH=False,
                 safe_k=None,
                 IH_rate=0.30, random_state=None, knn_classifier='knn',
                 DSEL_perc=0.5):

        super(KNOP, self).__init__(pool_classifiers, k,
                                   DFP=DFP,
                                   with_IH=with_IH,
                                   safe_k=safe_k,
                                   IH_rate=IH_rate,
                                   mode='weighting',
                                   needs_proba=True,
                                   random_state=random_state,
                                   knn_classifier=knn_classifier,
                                   DSEL_perc=DSEL_perc)

    def fit(self, X, y):
        """Train the DS model by setting the KNN algorithm and
        pre-process the information required to apply the DS
        methods. In this case, the scores of the base classifiers for
        the dynamic selection dataset (DSEL) are pre-calculated to
        transform each sample in DSEL into an output profile.

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
        super(KNOP, self).fit(X, y)
        if self.n_classes_ == 1:
            raise ValueError(
                "Error. KNOP  does not accept one class datasets!")
        self._check_predict_proba()
        self.dsel_scores_ = self._preprocess_dsel_scores()
        # Reshape DSEL_scores as a 2-D array for nearest neighbor calculations
        dsel_output_profiles = self.dsel_scores_.reshape(self.n_samples_,
                                                         self.n_classifiers_ *
                                                         self.n_classes_)

        self._fit_OP(dsel_output_profiles, self.DSEL_target_, self.k_)

        return self

    def _fit_OP(self, X_op, y_op, k):
        """ Fit the set of output profiles.

        Parameters
        ----------
        X_op : array of shape = [n_samples, n_features]
            Output profiles of the training data. n_features is equals
            to (n_classifiers x n_classes).

        y_op : array of shape = [n_samples]
               Class labels of each sample in X_op.

        k : int
            Number of output profiles used in the region of competence
            estimation.

        """
        self.op_knn_ = self.knn_class_(k)

        if self.n_classes_ == 2:
            # Get only the scores for one class since they are complementary
            X_temp = X_op[:, ::2]
            self.op_knn_.fit(X_temp, y_op)
        else:
            self.op_knn_.fit(X_op, y_op)

    def _get_similar_out_profiles(self, probabilities):
        """Get the most similar output profiles of the query sample.

        Parameters
        ----------
        probabilities : array of shape = [n_samples, n_classifiers, n_classes]
                        predictions of each base classifier for all samples.

        Returns
        -------
        dists : list of shape = [n_samples, k]
                The distances between the query and each sample in the region
                of competence. The vector is ordered in an ascending fashion.

        idx : list of shape = [n_samples, k]
            Indices of the instances belonging to the region of competence of
            the given query sample.
        """

        if self.n_classes_ == 2:
            # Get only the scores for one class since they are complementary
            query_op = probabilities[:, :, 0]
        else:
            query_op = probabilities.reshape((probabilities.shape[0],
                                              self.n_classifiers_ *
                                              self.n_classes_))

        dists, idx = self.op_knn_.kneighbors(query_op, n_neighbors=self.k_,
                                             return_distance=True)
        return dists, np.atleast_2d(idx)

    def estimate_competence_from_proba(self, query, probabilities,
                                       neighbors=None, distances=None):

        """The competence of the base classifiers is simply estimated as
        the number of samples in the region of competence that it correctly
        classified. This method received an array with
        the pre-calculated probability  estimates for each query.

        This information is later used to determine the number of votes
        obtained for each base classifier.

        Parameters
        ----------
        query : array of shape = [n_samples, n_features]
                The test examples.

        neighbors : array of shale = [n_samples, n_neighbors]
            Indices of the k nearest neighbors according for each test sample

        distances : array of shale = [n_samples, n_neighbors]
            Distances of the k nearest neighbors according for each test
            sample.

        probabilities : array of shape = [n_samples, n_classifiers, n_classes]
            Probabilities estimates obtained by each each base classifier
            for each query sample.

        Returns
        -------
        competences : array of shape = [n_samples, n_classifiers]
            Competence level estimated for each base classifier and test
            example.
        """
        _, idx_neighbors = self._get_similar_out_profiles(probabilities)
        competences = np.sum(self.DSEL_processed_[idx_neighbors, :], axis=1,
                             dtype=np.float)

        return competences

    def select(self, competences):
        """Select the base classifiers for the classification of the query
        sample.

        Each base classifier can be selected more than once. The number of
        times a base classifier is selected (votes) is equals to the number
        of samples it correctly classified in the region of competence.

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

        # Select classifier if it correctly classified at least one sample
        selected_classifiers = (competences > 0)

        # For the rows that are all False (i.e., no base classifier
        # was selected, select all classifiers (set all True)
        selected_classifiers[~np.any(selected_classifiers, axis=1), :] = True

        return selected_classifiers
