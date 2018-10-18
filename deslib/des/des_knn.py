# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.base import BaseDS
from deslib.util.aggregation import majority_voting_rule
from deslib.util.diversity import negative_double_fault, Q_statistic, \
    ratio_errors, compute_pairwise_diversity


class DESKNN(BaseDS):
    """Dynamic ensemble Selection KNN (DES-KNN).

    This method selects an ensemble of classifiers taking into account the
    accuracy and diversity of the base classifiers. The k-NN algorithm is used
    to define the region of competence. The N most accurate classifiers in the
    region of competence are first selected. Then, the J more diverse
    classifiers from the N most accurate classifiers are selected to compose
    the ensemble.

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

    pct_accuracy : float (Default = 0.5)
                   Percentage of base classifiers selected based on accuracy

    pct_diversity : float (Default = 0.3)
                    Percentage of base classifiers selected based n diversity

    more_diverse : Boolean (Default = True)
        Whether we select the most or the least diverse classifiers to add
        to the pre-selected ensemble

    metric : String (Default = 'df')
        Metric used to estimate the diversity of the base classifiers. Can be
        either the double fault (df), Q-statistics (Q), or error correlation.

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

    def __init__(self, pool_classifiers=None, k=7, DFP=False, with_IH=False,
                 safe_k=None,
                 IH_rate=0.30,
                 pct_accuracy=0.5,
                 pct_diversity=0.3,
                 more_diverse=True,
                 metric='DF',
                 random_state=None,
                 knn_classifier='knn',
                 DSEL_perc=0.5):

        super(DESKNN, self).__init__(pool_classifiers=pool_classifiers,
                                     k=k,
                                     DFP=DFP,
                                     with_IH=with_IH,
                                     safe_k=safe_k,
                                     IH_rate=IH_rate,
                                     random_state=random_state,
                                     knn_classifier=knn_classifier,
                                     DSEL_perc=DSEL_perc)

        self.metric = metric
        self.pct_accuracy = pct_accuracy
        self.pct_diversity = pct_diversity
        self.more_diverse = more_diverse

    def fit(self, X, y):
        """ Prepare the DS model by setting the KNN algorithm and
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
        super(DESKNN, self).fit(X, y)

        self.N_ = int(self.n_classifiers_ * self.pct_accuracy)

        self.J_ = int(np.ceil(self.n_classifiers_ * self.pct_diversity))

        self._check_parameters()

        return self

    def estimate_competence(self, query, neighbors, distances=None,
                            predictions=None):
        """estimate the competence level of each base classifier :math:`c_{i}`
        for the classification of the query sample.

        The competence is estimated using the accuracy and diversity criteria.
        First the classification accuracy of the base classifiers in the
        region of competence is estimated. Then the diversity of the
        base classifiers is estimated.

        The method returns two arrays: One containing the accuracy and the
        other the diversity of each base classifier.

        Parameters
        ----------
        query : array cf shape  = [n_samples, n_features]
                The query sample.

        neighbors : array of shale = [n_samples, n_neighbors]
            Indices of the k nearest neighbors according for each test sample.

        distances : array of shale = [n_samples, n_neighbors]
            Distances of the k nearest neighbors according for each test
            sample.


        predictions : array of shape = [n_samples, n_classifiers]
            Predictions of the base classifiers for all test examples.

        Notes
        ------
        This technique uses both the accuracy and diversity information to
        perform dynamic selection. For this reason the function returns a
        dictionary containing these two values instead of a single ndarray
        containing the competence level estimates for each base classifier.

        Returns
        -------
        accuracy : array of shape = [n_samples, n_classifiers}
                   Local Accuracy estimates (competences) of the base
                   classifiers for all query samples.

        diversity : array of shape = [n_samples, n_classifiers}
                    Average pairwise diversity of each base classifiers for
                    all test examples.

        """
        accuracy = np.mean(self.DSEL_processed_[neighbors, :], axis=1)

        predicted_matrix = self.BKS_DSEL_[neighbors, :]
        targets = self.DSEL_target_[neighbors]

        self._set_diversity_func()

        # TODO: optimize this part with numpy instead of for loops
        diversity = np.zeros((query.shape[0], self.n_classifiers_))
        for sample_idx in range(query.shape[0]):
            this_diversity = compute_pairwise_diversity(targets[sample_idx, :],
                                                        predicted_matrix[
                                                        sample_idx, :, :],
                                                        self.diversity_func_)

            diversity[sample_idx, :] = this_diversity

        return accuracy, diversity

    def select(self, accuracy, diversity):
        """Select an ensemble containing the N most accurate ant the J most
        diverse classifiers for the classification of the query sample.

        Parameters
        ----------
        accuracy : array of shape = [n_samples, n_classifiers]
            Local Accuracy estimates (competence) of each base classifiers.

        diversity : array of shape = [n_samples, n_classifiers]
                    Average pairwise diversity of each base classifiers.

        Returns
        -------
        selected_classifiers : array of shape = [n_samples, self.J]
            Array containing the indices of the J selected base classifier
            for each test example.
        """
        # Check if the accuracy and diversity arrays have
        # the correct dimensionality.
        if accuracy.ndim < 2:
            accuracy = accuracy.reshape(1, -1)

        if diversity.ndim < 2:
            diversity = diversity.reshape(1, -1)

        # sort the array to remove the most accurate classifiers
        competent_indices = np.argsort(accuracy, axis=1)[:, ::-1][:, 0:self.N_]
        diversity_of_selected = diversity[
            np.arange(diversity.shape[0])[:, None], competent_indices]
        # diversity_of_selected = diversity.take(competent_indices)

        # sort the remaining classifiers to select the most diverse ones
        if self.more_diverse:
            diversity_indices = np.argsort(diversity_of_selected, axis=1)
            diversity_indices = diversity_indices[:, ::-1][:, 0:self.J_]
        else:
            diversity_indices = np.argsort(diversity_of_selected, axis=1)
            diversity_indices = diversity_indices[:, 0:self.J_]

        # Getting the index of all selected base classifiers.
        selected_classifiers = competent_indices[
            np.arange(competent_indices.shape[0])[:, None], diversity_indices]

        return selected_classifiers

    def classify_with_ds(self, query, predictions, probabilities=None,
                         neighbors=None, distances=None, DFP_mask=None):
        """Predicts the label of the corresponding query sample.

        Parameters
        ----------
        query : array of shape = [n_samples, n_features]
                The test examples

        predictions : array of shape = [n_samples, n_classifiers]
                      Predictions of the base classifiers for all test examples

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

        Notes
        ------
        Different than other DES techniques, this method is based on a two
        stage selection, where first the most accurate classifier are selected,
        then the diversity information is used to get the most diverse ensemble
        for the probability estimation. Hence, the weighting mode is not
        defined. Also, the selected ensemble size is fixed (self.J), so there
        is no need to use masked arrays in this class.

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

        accuracy, diversity = self.estimate_competence(query,
                                                       neighbors,
                                                       distances=distances,
                                                       predictions=predictions)

        if self.DFP:
            accuracy = accuracy * DFP_mask

        selected_classifiers = self.select(accuracy, diversity)
        votes = predictions[
            np.arange(predictions.shape[0])[:, None], selected_classifiers]
        predicted_label = majority_voting_rule(votes)

        return predicted_label

    def predict_proba_with_ds(self, query, predictions, probabilities,
                              neighbors=None, distances=None, DFP_mask=None):
        """Predicts the posterior probabilities.

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
            Indices of the k nearest neighbors according for each test sample

        distances : array of shale = [n_samples, n_neighbors]
            Distances of the k nearest neighbors according for each test sample

        DFP_mask : array of shape = [n_samples, n_classifiers]
            Mask containing 1 for the selected base classifier and 0 otherwise.

        Notes
        ------
        Different than other DES techniques, this method is based on a two
        stage selection, where first the most accurate classifier are selected,
        then the diversity information is used to get the most diverse ensemble
        for the probability estimation. Hence, the weighting mode is not
        available.

        Returns
        -------
        predicted_proba : array = [n_samples, n_classes]
                          Probability estimates for all test examples.
        """

        if query.shape[0] != probabilities.shape[0]:
            raise ValueError(
                'The arrays query and predictions must have the same number'
                ' of samples. query.shape is {}'
                'and predictions.shape is {}'.format(query.shape,
                                                     predictions.shape))

        accuracy, diversity = self.estimate_competence(query,
                                                       neighbors,
                                                       distances=distances,
                                                       predictions=predictions)
        if self.DFP:
            accuracy = accuracy * DFP_mask

        # This method always performs selection. There is no weighted version.
        selected_classifiers = self.select(accuracy, diversity)
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

    def _set_diversity_func(self):
        """Set the diversity function to be used according to the
        hyper-parameter metric

        The diversity_func_ can be either the Double Fault, Q-Statistics
        or Ratio of errors.
        ----------
        """
        if self.metric == 'DF':
            self.diversity_func_ = negative_double_fault
        elif self.metric == 'Q':
            self.diversity_func_ = Q_statistic
        else:
            self.diversity_func_ = ratio_errors
