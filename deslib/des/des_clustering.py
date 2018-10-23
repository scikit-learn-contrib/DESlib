# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans

from deslib.base import BaseDS
from deslib.util.aggregation import majority_voting_rule
from deslib.util.diversity import Q_statistic, ratio_errors, \
    negative_double_fault, compute_pairwise_diversity


class DESClustering(BaseDS):
    """Dynamic ensemble selection-Clustering (DES-Clustering).

    This method selects an ensemble of classifiers taking into account the
    accuracy and diversity of the base classifiers. The K-means algorithm is
    used to define the region of competence. For each cluster, the N most
    accurate classifiers are first selected. Then, the J more diverse
    classifiers from the N most accurate classifiers are selected to
    compose the ensemble.

    Parameters
    ----------
     pool_classifiers : list of classifiers (Default = None)
        The generated_pool of classifiers trained for the corresponding
        classification problem. Each base classifiers should support the method
        "predict". If None, then the pool of classifiers is a bagging
        classifier.

    clustering : sklearn.cluster (Default = None)
        The clustering model used to estimate the region of competence.
        If None, a KMeans with K = 5 is used.

    pct_accuracy : float (Default = 0.5)
                   Percentage of base classifiers selected based on accuracy

    pct_diversity : float (Default = 0.33)
                    Percentage of base classifiers selected based n diversity

    more_diverse : Boolean (Default = True)
                   Whether we select the most or the least diverse classifiers
                   to add to the pre-selected ensemble

    metric : String (Default = 'df')
        Metric used to estimate the diversity of the base classifiers. Can be
        either the double fault (df), Q-statistics (Q), or error correlation.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    DSEL_perc : float (Default = 0.5)
        Percentage of the input data used to fit DSEL.
        Note: This parameter is only used if the pool of classifier is None or
        unfitted.


    References
    ----------
    Soares, R. G., Santana, A., Canuto, A. M., & de Souto, M. C. P.
    "Using accuracy and more_diverse to select classifiers to build ensembles."
    International Joint Conference on Neural Networks (IJCNN)., 2006.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection
    of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
    selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.
    """

    def __init__(self, pool_classifiers=None,
                 clustering=None,
                 with_IH=False,
                 safe_k=None,
                 IH_rate=0.30,
                 pct_accuracy=0.5,
                 pct_diversity=0.33,
                 more_diverse=True,
                 metric='DF',
                 random_state=None,
                 DSEL_perc=0.5):

        super(DESClustering, self).__init__(pool_classifiers=pool_classifiers,
                                            with_IH=with_IH,
                                            safe_k=safe_k,
                                            IH_rate=IH_rate,
                                            random_state=random_state,
                                            DSEL_perc=DSEL_perc)

        self.metric = metric
        self.clustering = clustering
        self.pct_accuracy = pct_accuracy
        self.pct_diversity = pct_diversity
        self.more_diverse = more_diverse

    def fit(self, X, y):
        """ Train the DS model by setting the Clustering algorithm and
        pre-processing the information required to apply the DS
        methods.

        First the data is divided into K clusters. Then, for each cluster,
        the N most accurate classifiers are first selected. Then, the J more
        diverse classifiers from the N most accurate classifiers are selected
        to compose the ensemble of the corresponding cluster. An ensemble of
        classifiers is assigned to each of the K clusters.

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
        super(DESClustering, self).fit(X, y)

        self.N_ = int(self.n_classifiers_ * self.pct_accuracy)
        self.J_ = int(np.ceil(self.n_classifiers_ * self.pct_diversity))

        self._check_parameters()

        if self.clustering is None:
            self.clustering_ = KMeans(n_clusters=5,
                                      random_state=self.random_state)
            self.clustering_.fit(self.DSEL_data_)
        else:
            self.clustering_ = self.clustering.fit(self.DSEL_data_)

        # set the diversity metric used
        self._set_diversity_func()

        # Since the clusters are fixed, we can pre-compute the accuracy and
        # diversity of each cluster as well as the # selected classifiers
        # (indices) for each one. These pre-computed information will be kept
        # on those three variables:
        self.accuracy_cluster_ = np.zeros(
            (self.clustering_.n_clusters, self.n_classifiers_))
        self.diversity_cluster_ = np.zeros(
            (self.clustering_.n_clusters, self.n_classifiers_))
        self.indices_ = np.zeros((self.clustering_.n_clusters, self.J_),
                                 dtype=int)

        self._preprocess_clusters()
        return self

    def _preprocess_clusters(self):
        """Preprocess the competence as well as the average diversity of each
        base classifier for each specific cluster.

        This process makes the test routines faster, since the ensemble of
        classifiers of each cluster is already predefined.

        The class attributes Accuracy_cluster_ and diversity_cluster_ stores
        the accuracy and diversity information respectively of each base
        classifier for each cluster. The attribute indices_ stores the
        pre-selected base classifiers for each cluster.
        """
        labels = self.clustering_.predict(self.DSEL_data_)

        for cluster_index in range(self.clustering_.n_clusters):

            # Get the indices_ of the samples in the corresponding cluster.
            sample_indices = np.where(labels == cluster_index)[0]

            # Compute accuracy of each classifier in this cluster
            accuracy = np.mean(self.DSEL_processed_[sample_indices, :], axis=0)
            self.accuracy_cluster_[cluster_index, :] = accuracy

            # Get the N_ most accurate classifiers in the cluster
            accuracy_indices = np.argsort(accuracy)[::-1][0:self.N_]

            # Get the target labels for the samples in the corresponding
            #  cluster for the diversity calculation.

            targets = self.DSEL_target_[sample_indices]
            self.diversity_cluster_[cluster_index, :] = \
                compute_pairwise_diversity(targets,
                                           self.BKS_DSEL_[sample_indices, :],
                                           self.diversity_func_)

            diversity_of_selected = self.diversity_cluster_[
                cluster_index, accuracy_indices]

            if self.more_diverse:
                diversity_indices = np.argsort(diversity_of_selected)[::-1][
                                    0:self.J_]
            else:
                diversity_indices = np.argsort(diversity_of_selected)[
                                    0:self.J_]

            self.indices_[cluster_index, :] = accuracy_indices[
                diversity_indices]

    def estimate_competence(self, query, predictions=None):
        """Get the competence estimates of each base classifier :math:`c_{i}`
        for the classification of the query sample.

        In this case, the competences were already pre-calculated for each
        cluster. So this method computes the nearest cluster and get the
        pre-calculated competences of the base classifiers for the
        corresponding cluster.

        Parameters
        ----------
        query : array of shape = [n_samples, n_features]
                The query sample.

        predictions : array of shape = [n_samples, n_classifiers]
            Predictions of the base classifiers for all test examples.

        Returns
        -------
        competences : array = [n_samples, n_classifiers]
                      The competence level estimated for each base classifier.
        """
        cluster_index = self.clustering_.predict(query)
        competences = self.accuracy_cluster_[cluster_index][:]
        return competences

    def select(self, query):
        """Select an ensemble with the most accurate and most diverse
        classifier for the classification of the query.

        The ensemble for each cluster was already pre-calculated in the fit
        method. So, this method calculates the closest cluster, and returns
        the ensemble associated to this cluster.

        Parameters
        ----------
        query : array of shape = [n_samples, n_features]
                The test examples.

        Returns
        -------
        selected_classifiers : array of shape = [n_samples, self.k]
            Indices of the selected base classifier for each test example.

        """
        cluster_index = self.clustering_.predict(query)
        selected_classifiers = self.indices_[cluster_index, :]
        return selected_classifiers

    def classify_with_ds(self, query, predictions, probabilities=None,
                         neighbors=None, distances=None, DFP_mask=None):
        """Predicts the label of the corresponding query sample.

        Parameters
        ----------
        query : array of shape = [n_features]
                The test sample.

        predictions : array of shape = [n_samples, n_classifiers]
            Predictions of the base classifiers for all test examples.

        probabilities : array of shape = [n_samples, n_classifiers, n_classes]
            Probabilities estimates of each base classifier for all test
            examples.

        neighbors : array of shale = [n_samples, n_neighbors]
            Indices of the k nearest neighbors according for each test sample.

        distances : array of shale = [n_samples, n_neighbors]
            Distances of the k nearest neighbors according for each test
            sample.

        DFP_mask : array of shape = [n_samples, n_classifiers]
            Mask containing 1 for the selected base classifier and 0 otherwise.

        Returns
        -------
        predicted_label : array of shape = [n_samples]
                          Predicted class label for each test example.
        """
        if query.ndim < 2:
            query = query.reshape(1, -1)

        if predictions.ndim < 2:
            predictions = predictions.reshape(1, -1)

        if query.shape[0] != predictions.shape[0]:
            raise ValueError(
                'The arrays query and predictions must have the same number'
                ' of samples. query.shape is {}'
                'and predictions.shape is {}'.format(query.shape,
                                                     predictions.shape))

        selected_classifiers = self.select(query)
        votes = predictions[
            np.arange(predictions.shape[0])[:, None], selected_classifiers]
        predicted_label = majority_voting_rule(votes)

        return predicted_label

    def predict_proba_with_ds(self, query, predictions, probabilities,
                              neighbors=None, distances=None, DFP_mask=None):
        """Predicts the label of the corresponding query sample.

        Parameters
        ----------
        query : array of shape = [n_samples, n_features]
                The test examples.

        predictions : array of shape = [n_samples, n_classifiers]
            Predictions of the base classifiers for all test examples.

        probabilities : array of shape = [n_samples, n_classifiers, n_classes]
            Probabilities estimates of each base classifier for all test
            examples.

        neighbors : array of shale = [n_samples, n_neighbors]
            Indices of the k nearest neighbors according for each test sample.

        distances : array of shale = [n_samples, n_neighbors]
            Distances of the k nearest neighbors according for each test sample

        DFP_mask : array of shape = [n_samples, n_classifiers]
            Mask containing 1 for the selected base classifier and 0 otherwise.

        Returns
        -------
        predicted_proba : array of shape = [n_samples, n_classes]
            Posterior probabilities estimates for each test example.
        """
        if query.shape[0] != probabilities.shape[0]:
            raise ValueError(
                'The arrays query and predictions must have the same number of'
                ' samples. query.shape is {}'
                'and predictions.shape is {}'.format(query.shape,
                                                     predictions.shape))

        selected_classifiers = self.select(query)
        ensemble_proba = probabilities[
                         np.arange(probabilities.shape[0])[:, None],
                         selected_classifiers, :]
        predicted_proba = np.mean(ensemble_proba, axis=1)

        return predicted_proba

    def _check_parameters(self):
        """Check if the parameters passed as argument are correct.

        Raises
        ------
        ValueError
            If the hyper-parameters are incorrect.
        """
        if self.metric not in ['DF', 'Q', 'ratio']:
            raise ValueError(
                'Diversity metric must be one of the following values:'
                ' "DF", "Q" or "Ratio"')

        if self.N_ <= 0 or self.J_ <= 0:
            raise ValueError("The values of N_ and J_ should be higher than 0"
                             "N_ = {}, J_= {} ".format(self.N_, self.J_))
        if self.N_ < self.J_:
            raise ValueError(
                "The value of N_ should be greater or equals than J_"
                "N_ = {}, J_= {} ".format(self.N_, self.J_))

        if self.clustering is not None:
            if not isinstance(self.clustering, ClusterMixin):
                raise ValueError(
                    "Parameter clustering must be a sklearn"
                    " cluster estimator.")

    def _set_diversity_func(self):
        """Set the diversity function to be used according to the
        hyper-parameter metric

        The diversity_func_ can be either the Double Fault, Q-Statistics
        or Ratio of errors.

        """
        if self.metric == 'DF':
            self.diversity_func_ = negative_double_fault
        elif self.metric == 'Q':
            self.diversity_func_ = Q_statistic
        else:
            self.diversity_func_ = ratio_errors
