# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from pythonds.des.base import DES
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from pythonds.util.diversity import double_fault, Q_statistic, ratio_errors


class DESClustering(DES):
    """Dynamic ensemble selection-Performance(DSKNN).
    This method selects an ensemble of classifiers taking into account the
    accuracy and more_diverse of the base classifiers. First the most accurate classifiers
    are selected. Next, the most diverse classifiers, in relation to the selected classifiers,
    are added to the ensemble

    Parameters
    ----------
    pool_classifiers : type, the generated_pool of classifiers trained for the corresponding
    classification problem.

    k : int (Default = 7), Number of clusters for the KMeans clustering method.

    version : String (Default = selection), Wether the technique will perform
    dynamic selection, dynamic weighting or an hybrid approach for classification

    N : float (Default = 0.3), Percentage of base classifiers selected based on accuracy

    J : float (Default = 0.3), Percentage of base classifiers selected based n diversity

    more_diverse : Boolean (Default = True), Whether we select the most or the least diverse classifiers
    to add to the pre-selected ensemble

    metric : String (Default = 'df'), Diversity diversity_func used to estimate the diversity of the base classifiers. Can
    be either the double fault (df), Q-statistics (Q), or error correlation (corr)

    DFP : Boolean (Default = False), Determines if the dynamic frienemy prunning is applied.

    with_IH : Boolean (Default = False), Whether the hardness level of the region of competence is used to decide
    between using the DS algorithm or the KNN for classification of a given query sample.

    safe_k : int (default = None), the size of the indecision region.

    IH_rate : float (default = 0.3), Hardness threshold. If the hardness level of the competence region is lower than
    the IH_rate the KNN classifier is used. Otherwise, the DS algorithm is used for classification.

    References
    ----------
    Soares, R. G., Santana, A., Canuto, A. M., & de Souto, M. C. P. "Using accuracy and more_diverse to select
    classifiers  to build ensembles." International Joint Conference on Neural Networks (IJCNN)., 2006.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.
    """

    def __init__(self, pool_classifiers, k=5, version='selection', N=0.3, J=0.3, more_diverse=True, metric='DF',
                 DFP=False, with_IH=False, safe_k=None, IH_rate=0.30):

        metric.upper()
        super(DESClustering, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate,
                                            version=version)

        self.N = int(self.n_classifiers * N)
        self.J = int(self.n_classifiers * J)
        self.more_diverse = more_diverse
        self.metric = metric
        self.name = 'DESClustering (des-C)'
        if metric == 'DF':
            self.diversity_func = double_fault
        elif metric == 'Q':
            self.diversity_func = Q_statistic
        else:
            self.diversity_func = ratio_errors

        self.clustering = KMeans(n_clusters=k)
        self.accuracy_cluster = np.zeros((self.k, self.n_classifiers))
        self.diversity_cluster = np.zeros((self.k, self.n_classifiers))
        # Since the clusters are fixed, we can pre-compute the ensemble for each cluster
        self.classifiers_cluster = np.zeros((self.k, self.N + self.J))
        self.indices = np.zeros((self.k, self.N + self.J),  dtype=int)

        self._validate_inputs()

    def fit(self, X, y):
        """Train the DS model by setting the Clustering algorithm and
        pre-processing the information required to apply the DS
        methods. In this case, after applyng the clustering method, the esemble containing
        most competente classifiers taking into account accuracy and diversity are
        estimated for each cluster.

        Parameters
        ----------
        X : matrix of shape = [n_samples, n_features] with the data.

        y : class labels of each sample in X.

        Returns
        -------
        self
        """
        self._set_dsel(X, y)
        labels = self.clustering.fit_predict(X)
        for cluster_index in range(self.k):
            # create a dataset with this cluster
            data = []
            targets = []
            for sample_index, label in enumerate(labels):
                if label == cluster_index:
                    data.append(self.DSEL_data[sample_index, :])
                    targets.append(self.DSEL_target[sample_index])

            # compute accuracy of each classifier in this cluster
            prediction_matrix = np.empty((len(targets), self.n_classifiers))
            for clf_index, clf in enumerate(self.pool_classifiers):
                predictions = clf.predict(np.array(data))
                self.accuracy_cluster[cluster_index][clf_index] = accuracy_score(np.array(targets), predictions)
                prediction_matrix[:, clf_index] = predictions

            # get the most accurate classifiers for the corresponding cluster
            accuracies = self.accuracy_cluster[cluster_index, :]
            accuracy_indices = np.argsort(accuracies)[::-1][0:self.N]

            # compute the more_diverse of each base classifier in this cluster
            for clf_index in range(self.n_classifiers):
                for clf_index2 in range(self.n_classifiers):
                    if clf_index != clf_index2:
                        self.diversity_cluster[cluster_index][clf_index] += \
                            double_fault(targets, prediction_matrix[:, clf_index], prediction_matrix[:, clf_index2])

            diversity = self.diversity_cluster[cluster_index, :]
            diversity = np.delete(diversity, accuracy_indices)
            if self.more_diverse:
                diversity_indices = np.argsort(diversity)[::-1][0:self.J]
            else:
                diversity_indices = np.argsort(diversity)[0:self.J]

            self.indices[cluster_index, :] = np.hstack((accuracy_indices, diversity_indices))

    def estimate_competence(self, query):
        """get the competence estimates of each base classifier ci for the classification of the query sample x.

        In this case, the competences are pre-calculated based on each cluster. So this method computes the
        nearest cluster of the query sample and get the pre-calculated competences of the base classifiers
        for the nearest cluster.

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        Returns
        -------
        competences : array = [n_classifiers] containing the competence level estimated
        for each base classifier
        """
        cluster_index = self.clustering.predict(query)
        competences = self.accuracy_cluster[cluster_index][:]
        diversity = self.diversity_cluster[cluster_index][:]
        return competences[0, :], diversity[0, :]

    def select(self, query):
        """Select an ensemble with the most accurate and most diverse classifier for the classification of the query.

        Since the method is based on clustering, the ensemble for each cluster is already pre-calculated. So, we only
        need to estimate which is the nearest cluster and then get the classifiers that were pre-selected for this
        cluster

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        Returns
        -------
        indices : the indices of the selected base classifiers

        competences : array = [n_classifiers] containing the competence level estimated
        for each base classifier
        """
        cluster_index = self.clustering.predict(query)[0]
        indices = self.indices[cluster_index, :]

        return indices

    def classify_instance(self, query):
        """Predicts the label of the corresponding query sample.
        Returns the predicted label.

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        Returns
        -------
        predicted_label: The predicted label of the query
        """
        indices = self.select(query)
        predicted_label = self.majority_voting(indices, query)
        return predicted_label

    def _validate_inputs(self):
        """Check if the parameters passed as argument are correct.

        The diversity_func must be either ['DF', 'Q', 'RATIO']

        The values of N and J should be higher than 0, and N >= J
        ----------
        """
        assert self.metric in ['DF', 'Q', 'RATIO']

        if self.N <= 0 or self.J <= 0:
            raise ValueError("The values of N and J should be higher than 0"
                             "N = %s, J= %s " % (self.N, self.J))
        if self.N < self.J:
            raise ValueError("The value of N should be greater or equals than J"
                             "N = %s, J= %s " % (self.N, self.J))
