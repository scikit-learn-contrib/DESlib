# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from deslib.des.base import DES
from deslib.util.aggregation import majority_voting
from deslib.util.diversity import Q_statistic, ratio_errors, negative_double_fault


class DESClustering(DES):
    """Dynamic ensemble selection-Clustering (DES-Clustering).
    This method selects an ensemble of classifiers taking into account the
    accuracy and more_diverse of the base classifiers. The K-means algorithm is used to define the region of competence
    First the most accurate classifiers are selected. Next, the most diverse classifiers, in relation to the selected
    classifiers, are added to the ensemble

    Parameters
    ----------
    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support methods "predict" and "predict_proba".

    k : int (Default = 5)
        Number of neighbors used to estimate the competence of the base classifiers.

    mode : String (Default = "selection")
              whether the technique will perform dynamic selection, dynamic weighting
              or an hybrid approach for classification

    pct_accuracy : float (Default = 0.5)
        Percentage of base classifiers selected based on accuracy

    pct_diversity : float (Default = 0.33)
        Percentage of base classifiers selected based n diversity

    more_diverse : Boolean (Default = True)
        Whether we select the most or the least diverse classifiers to add to the pre-selected ensemble

    metric : String (Default = 'df')
            Diversity diversity_func used to estimate the diversity of the base classifiers. Can
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

    def __init__(self, pool_classifiers, k=5, mode='selection',
                 pct_accuracy=0.5,
                 pct_diversity=0.33,
                 more_diverse=True,
                 metric='DF',
                 rng=np.random.RandomState()):

        super(DESClustering, self).__init__(pool_classifiers, k, mode=mode)

        self.name = 'DES-Clustering'
        self.N = int(self.n_classifiers * pct_accuracy)
        self.J = int(np.ceil(self.n_classifiers * pct_diversity))
        self.metric = metric.upper()
        self._validate_parameters()

        if metric == 'DF':
            self.diversity_func = negative_double_fault
        elif metric == 'Q':
            self.diversity_func = Q_statistic
        else:
            self.diversity_func = ratio_errors

        self.more_diverse = more_diverse
        self.roc_algorithm = KMeans(n_clusters=k, random_state=rng)
        self.accuracy_cluster = np.zeros((self.k, self.n_classifiers))
        self.diversity_cluster = np.zeros((self.k, self.n_classifiers))

        # Since the clusters are fixed, we can pre-compute the ensemble for each cluster
        self.classifiers_cluster = np.zeros((self.k, self.J))
        self.indices = np.zeros((self.k, self.J),  dtype=int)

    def fit(self, X, y):
        """Train the DS model by setting the Clustering algorithm and
        pre-processing the information required to apply the DS
        methods. In this case, after fitting the roc_algorithm method, the ensemble containing
        most competent classifiers taking into account accuracy and diversity are
        estimated for each cluster.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The input data.

        y : class labels of each sample in X.

        Returns
        -------
        self
        """

        self._set_dsel(X, y)
        labels = self.roc_algorithm.fit_predict(X)

        # For each cluster estimate the most accurate and most competent classifiers for it.
        for cluster_index in range(self.k):

            # Get the indices of the samples in the corresponding cluster.
            sample_indices = np.where(labels == cluster_index)[0]

            # Get the target labels for the samples in the corresponding cluster.
            targets = self.DSEL_target[sample_indices]

            # Compute accuracy of each classifier in this cluster
            prediction_matrix = np.zeros((len(targets), self.n_classifiers))
            for clf_index in range(self.n_classifiers):
                predictions = self.BKS_dsel[sample_indices, clf_index]
                prediction_matrix[:, clf_index] = predictions
                # Check if the dynamic frienemy pruning (DFP) should be used used
                self.accuracy_cluster[cluster_index][clf_index] = accuracy_score(np.array(targets), predictions)

            # Get the N most accurate classifiers for the corresponding cluster
            accuracies = self.accuracy_cluster[cluster_index, :]
            accuracy_indices = np.argsort(accuracies)[::-1][0:self.N]

            self.diversity_cluster[cluster_index, :] = self._compute_diversity(targets, prediction_matrix)
            diversity_of_selected = self.diversity_cluster[cluster_index, accuracy_indices]

            if self.more_diverse:
                diversity_indices = np.argsort(diversity_of_selected)[::-1][0:self.J]
            else:
                diversity_indices = np.argsort(diversity_of_selected)[0:self.J]

            self.indices[cluster_index, :] = accuracy_indices[diversity_indices]

    def _compute_diversity(self, targets, prediction_matrix):
        """Computes the pairwise diversity matrix.

         Parameters
         ----------
         targets : array of shape = [n_samples]:
                   Class labels of each sample in X.

         prediction_matrix : array of shape = [n_samples, n_classifiers]:
                             Predicted class labels for each classifier in the pool

         Returns
         -------
         diversity : array of shape = [n_classifiers, n_classifiers]
                     The pairwise diversity matrix calculated for the pool of classifiers

         """
        diversity = np.zeros(self.n_classifiers)

        for clf_index in range(self.n_classifiers):
            for clf_index2 in range(clf_index + 1, self.n_classifiers):
                this_diversity = self.diversity_func(targets,
                                                     prediction_matrix[:, clf_index],
                                                     prediction_matrix[:, clf_index2])

                diversity[clf_index] += this_diversity
                diversity[clf_index2] += this_diversity
        return diversity

    def estimate_competence(self, query):
        """get the competence estimates of each base classifier ci for the classification of the query sample x.

        In this case, the competences are pre-calculated based on each cluster. So this method computes the
        nearest cluster of the query sample and get the pre-calculated competences of the base classifiers
        for the nearest cluster.

        Parameters
        ----------
        query : array of shape = [n_features]
                The query sample

        Returns
        -------
        competences : array = [n_classifiers]
                      The competence level estimated for each base classifier
        """
        cluster_index = self.roc_algorithm.predict(query)
        competences = self.accuracy_cluster[cluster_index][:]
        return competences

    def select(self, query):
        """Select an ensemble with the most accurate and most diverse classifier for the classification of the query.

        Since the method is based on roc_algorithm, the ensemble for each cluster is already pre-calculated. So, we only
        need to estimate which is the nearest cluster and then get the classifiers that were pre-selected for this
        cluster

        Parameters
        ----------
        query : array of shape = [n_features]
                The query sample

        Returns
        -------
        indices : List containing the indices of the selected base classifiers

        """
        cluster_index = self.roc_algorithm.predict(query)[0]
        indices = self.indices[cluster_index, :]
        return indices

    def classify_instance(self, query):
        """Predicts the label of the corresponding query sample.
        Returns the predicted label.

        Parameters
        ----------
        query : array of shape = [n_features]
                The test sample

        Returns
        -------
        predicted_label: The predicted label of the query
        """
        indices = self.select(query)
        classifier_ensemble = self._get_classifier_ensemble(indices)
        predicted_label = majority_voting(classifier_ensemble, query)
        return predicted_label

    def _validate_parameters(self):
        """Check if the parameters passed as argument are correct.

        The diversity_func must be either ['DF', 'Q', 'RATIO']

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
