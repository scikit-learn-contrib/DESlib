# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans

from deslib.base import DS
from deslib.util.aggregation import majority_voting_rule
from deslib.util.diversity import Q_statistic, ratio_errors, negative_double_fault, compute_pairwise_diversity


class DESClustering(DS):
    """Dynamic ensemble selection-Clustering (DES-Clustering).

    This method selects an ensemble of classifiers taking into account the
    accuracy and diversity of the base classifiers. The K-means algorithm is used to define the region of competence.
    For each cluster, the N most accurate classifiers are first selected. Then, the J more diverse classifiers from the
    N most accurate classifiers are selected to compose the ensemble.

    Parameters
    ----------
    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support the method "predict".

    k : int (Default = 5)
        Number of neighbors used to estimate the competence of the base classifiers.

    pct_accuracy : float (Default = 0.5)
        Percentage of base classifiers selected based on accuracy

    pct_diversity : float (Default = 0.33)
        Percentage of base classifiers selected based n diversity

    more_diverse : Boolean (Default = True)
        Whether we select the most or the least diverse classifiers to add to the pre-selected ensemble

    metric : String (Default = 'df')
            Diversity diversity_func_ used to estimate the diversity of the base classifiers. Can
            be either the double fault (df), Q-statistics (Q), or error correlation (corr)

    rng : numpy.random.RandomState instance
          Random number generator to assure reproducible results.

    References
    ----------
    Soares, R. G., Santana, A., Canuto, A. M., & de Souto, M. C. P. "Using accuracy and more_diverse to select
    classifiers  to build ensembles." International Joint Conference on Neural Networks (IJCNN)., 2006.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.
    """

    def __init__(self, pool_classifiers=None, clustering=KMeans(n_clusters=5),
                 pct_accuracy=0.5,
                 pct_diversity=0.33,
                 more_diverse=True,
                 metric='DF',
                 rng=np.random.RandomState()):

        super(DESClustering, self).__init__(pool_classifiers)

        self.name = 'DES-Clustering'
        self.clustering = clustering
        self.pct_accuracy = pct_accuracy
        self.pct_diversity = pct_diversity
        self.more_diverse = more_diverse
        self.metric = metric.upper()
        self.rng = rng

        self.N = int(self.n_classifiers * pct_accuracy)
        self.J = int(np.ceil(self.n_classifiers * pct_diversity))

    def fit(self, X, y):
        """ Train the DS model by setting the Clustering algorithm and
        pre-processing the information required to apply the DS
        methods.

        First the data is divided into K clusters. Then, for each cluster, the N most accurate classifiers
        are first selected. Then, the J more diverse classifiers from the N most accurate classifiers are
        selected to compose the ensemble of the corresponding cluster. An ensemble of classifiers is assigned
        to each of the K clusters.

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
        self._validate_parameters()
        super(DESClustering, self).fit(X, y)
        self.clustering_ = self.clustering.fit(self.DSEL_data_)

        # y_ind = self.setup_label_encoder(y)
        # self._set_dsel(X, y_ind)
        # self.clustering_.fit(self.DSEL_data_)

        if self.metric == 'DF':
            self.diversity_func_ = negative_double_fault
        elif self.metric == 'Q':
            self.diversity_func_ = Q_statistic
        else:
            self.diversity_func_ = ratio_errors

        # Since the clusters are fixed, we can pre-compute the accuracy and diversity of each cluster as well as the
        # selected classifiers (indices) for each one. These pre-computed information will be kept on
        # those three variables:
        self.accuracy_cluster_ = np.zeros((self.clustering.n_clusters, self.n_classifiers))
        self.diversity_cluster_ = np.zeros((self.clustering.n_clusters, self.n_classifiers))
        self.indices_ = np.zeros((self.clustering.n_clusters, self.J), dtype=int)

        self._preprocess_clusters()

    def _preprocess_clusters(self):

        labels = self.clustering_.predict(self.DSEL_data_)
        # For each cluster estimate the most accurate and most competent classifiers for it.
        for cluster_index in range(self.clustering_.n_clusters):

            # Get the indices_ of the samples in the corresponding cluster.
            sample_indices = np.where(labels == cluster_index)[0]

            # Compute accuracy of each classifier in this cluster
            accuracy = np.mean(self.DSEL_processed_[sample_indices, :], axis=0)
            self.accuracy_cluster_[cluster_index, :] = accuracy

            # Get the N most accurate classifiers for the corresponding cluster
            accuracy_indices = np.argsort(accuracy)[::-1][0:self.N]

            # Get the target labels for the samples in the corresponding cluster for the diversity calculation.

            targets = self.DSEL_target_[sample_indices]
            self.diversity_cluster_[cluster_index, :] = \
                compute_pairwise_diversity(targets, self.BKS_DSEL_[sample_indices, :], self.diversity_func_)

            diversity_of_selected = self.diversity_cluster_[cluster_index, accuracy_indices]

            if self.more_diverse:
                diversity_indices = np.argsort(diversity_of_selected)[::-1][0:self.J]
            else:
                diversity_indices = np.argsort(diversity_of_selected)[0:self.J]

            self.indices_[cluster_index, :] = accuracy_indices[diversity_indices]

    def estimate_competence(self, query, predictions=None):
        """Get the competence estimates of each base classifier :math:`c_{i}`
        for the classification of the query sample.

        In this case, the competences were already pre-calculated for each cluster. So this method computes the
        nearest cluster and get the pre-calculated competences of the base classifiers
        for the corresponding cluster.

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
        """Select an ensemble with the most accurate and most diverse classifier for the classification of the query.

        The ensemble for each cluster was already pre-calculated in the fit method. So, this method calculates the
        closest cluster, and returns the ensemble associated to this cluster.

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

    def classify_with_ds(self, query, predictions, probabilities=None):
        """Predicts the label of the corresponding query sample.

        Parameters
        ----------
        query : array of shape = [n_features]
                The test sample.

        predictions : array of shape = [n_samples, n_classifiers]
                      Predictions of the base classifiers for all test examples.

        probabilities : array of shape = [n_samples, n_classifiers, n_classes]
                        Probabilities estimates of each base classifier for all test examples.

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
            raise ValueError('The arrays query and predictions must have the same number of samples. query.shape is {}'
                             'and predictions.shape is {}' .format(query.shape, predictions.shape))

        selected_classifiers = self.select(query)
        votes = predictions[np.arange(predictions.shape[0])[:, None], selected_classifiers]
        predicted_label = majority_voting_rule(votes)

        return predicted_label

    def predict_proba_with_ds(self, query, predictions, probabilities):
        """Predicts the label of the corresponding query sample.

        Parameters
        ----------
        query : array of shape = [n_samples, n_features]
                The test examples.

        predictions : array of shape = [n_samples, n_classifiers]
                      Predictions of the base classifiers for all test examples.

        probabilities : array of shape = [n_samples, n_classifiers, n_classes]
                        Probabilities estimates of each base classifier for all test examples.

        Returns
        -------
        predicted_proba : array of shape = [n_samples, n_classes]
                          Posterior probabilities estimates for each test example
        """
        if query.shape[0] != probabilities.shape[0]:
            raise ValueError('The arrays query and predictions must have the same number of samples. query.shape is {}'
                             'and predictions.shape is {}' .format(query.shape, predictions.shape))

        selected_classifiers = self.select(query)
        ensemble_proba = probabilities[np.arange(probabilities.shape[0])[:, None], selected_classifiers, :]
        predicted_proba = np.mean(ensemble_proba, axis=1)

        return predicted_proba

    def _validate_parameters(self):
        """Check if the parameters passed as argument are correct.

        The diversity_func_ must be either ['DF', 'Q', 'RATIO']

        The values of N and J should be higher than 0, and N >= J
        ----------
        """
        if self.metric not in ['DF', 'Q', 'RATIO']:
            raise ValueError('Diversity metric must be one of the following values: "DF", "Q" or "Ratio"')

        if self.N <= 0 or self.J <= 0:
            raise ValueError("The values of N and J should be higher than 0"
                             "N, J" .format(self.N, self.J))
        if self.N < self.J:
            raise ValueError("The value of N should be greater or equals than J"
                             "N, J" .format(self.N, self.J))

        if not isinstance(self.clustering, ClusterMixin):
            raise ValueError("Parameter clustering must be a sklearn cluster estimator.")
