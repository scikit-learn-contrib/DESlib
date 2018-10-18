# coding=utf-8

# Author: Qiushi Wang <wqiushi@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.preprocessing import normalize

from deslib.base import BaseDS
from deslib.util.aggregation import majority_voting_rule


class DESMI(BaseDS):
    """Dynamic ensemble Selection for multi-class imbalanced datasets (DES-MI).


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

    alpha : float (Default = 0.9)
            Scaling coefficient to regulate the weight value

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
    García, S.; Zhang, Z.-L.; Altalhi, A.; Alshomrani, S. & Herrera, F.
    "Dynamic ensemble selection for multi-class
    imbalanced datasets." Information Sciences, 2018, 445-446, 22 - 37

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection
    of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
    selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.
    """

    def __init__(self, pool_classifiers=None, k=7, pct_accuracy=0.4, alpha=0.9,
                 DFP=False, with_IH=False, safe_k=None,
                 IH_rate=0.30, random_state=None, knn_classifier='knn',
                 DSEL_perc=0.5):

        super(DESMI, self).__init__(pool_classifiers=pool_classifiers,
                                    k=k,
                                    DFP=DFP,
                                    with_IH=with_IH,
                                    safe_k=safe_k,
                                    IH_rate=IH_rate,
                                    random_state=random_state,
                                    knn_classifier=knn_classifier,
                                    DSEL_perc=DSEL_perc)

        self.alpha = alpha
        self.pct_accuracy = pct_accuracy

    def estimate_competence(self, query, neighbors, distances=None,
                            predictions=None):
        """estimate the competence level of each base classifier :math:`c_{i}`
        for the classification of the query sample. Returns a ndarray
        containing the competence level of each base classifier.

        The competence is estimated using the accuracy criteria.
        The accuracy is estimated by the weighted results of classifiers who
        correctly classify the members in the competence region. The weight
        of member :math:`x_i` is related to the number of samples of the same
        class of :math:`x_i` in the training dataset.
        For detail, please see the first reference, Algorithm 2.

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

        Returns
        -------
        accuracy : array of shape = [n_samples, n_classifiers}
            Local Accuracy estimates (competences) of the base classifiers
            for all query samples.


        """
        # calculate the weight
        class_frequency = np.bincount(self.DSEL_target_)
        targets = self.DSEL_target_[neighbors]  # [n_samples, K_neighbors]
        num = class_frequency[targets]
        weight = 1. / (1 + np.exp(self.alpha * num))
        weight = normalize(weight, norm='l1')
        correct_num = self.DSEL_processed_[neighbors, :]
        correct = np.zeros((query.shape[0], self.k_, self.n_classifiers_))
        for i in range(self.n_classifiers_):
            correct[:, :, i] = correct_num[:, :, i] * weight

        # Apply the weights to each sample for each base classifier
        competence = correct_num * weight[:, :, np.newaxis]
        # calculate the classifiers mean competence for all
        # samples/base classifier
        competence = np.sum(competence, axis=1)

        return competence

    def select(self, competences):
        """Select an ensemble containing the N most accurate classifiers for
        the classification of the query sample.

        Parameters
        ----------
        competences : array of shape = [n_samples, n_classifiers]
            Competence estimates of each base classifiers for all query
            samples.

        Returns
        -------
        selected_classifiers : array of shape = [n_samples, self.N]
            Matrix containing the indices of the N selected base classifier
            for each test example.
        """
        # Check if the accuracy and diversity arrays have
        # the correct dimensionality.
        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        # sort the array to remove the most accurate classifiers
        selected_classifiers = np.argsort(competences, axis=1)
        selected_classifiers = selected_classifiers[:, ::-1][:, 0:self.N_]

        return selected_classifiers

    def classify_with_ds(self, query, predictions, probabilities=None,
                         neighbors=None, distances=None, DFP_mask=None):
        """Predicts the label of the corresponding query sample.

        Parameters
        ----------
        query : array of shape = [n_samples, n_features]
                The test examples

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

        Notes
        ------
        Different than other DES techniques, this method only select N
        candidates from the pool of classifiers.

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

        accuracy = self.estimate_competence(query,
                                            neighbors=neighbors,
                                            predictions=predictions)

        if self.DFP:
            accuracy = accuracy * DFP_mask

        selected_classifiers = self.select(accuracy)
        votes = predictions[
            np.arange(predictions.shape[0])[:, None], selected_classifiers]
        predicted_label = majority_voting_rule(votes)

        return predicted_label

    def predict_proba_with_ds(self, query, predictions, probabilities,
                              neighbors=None, distances=None, DFP_mask=None):
        """Predicts the posterior probabilities of the corresponding
        query sample.

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
            Distances of the k nearest neighbors according for each test
            sample.

        DFP_mask : array of shape = [n_samples, n_classifiers]
            Mask containing 1 for the selected base classifier and 0 otherwise.

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

        accuracy = self.estimate_competence(query,
                                            neighbors=neighbors,
                                            distances=distances)

        if self.DFP:
            accuracy = accuracy * DFP_mask

        selected_classifiers = self.select(accuracy)
        ensemble_proba = probabilities[
                         np.arange(probabilities.shape[0])[:, None],
                         selected_classifiers, :]

        predicted_proba = np.mean(ensemble_proba, axis=1)

        return predicted_proba

    def _validate_parameters(self):
        """Check if the parameters passed as argument are correct.

        Raises
        ------
        ValueError
            If the hyper-parameters are incorrect.
        """
        super(DESMI, self)._validate_parameters()

        self.N_ = int(self.n_classifiers_ * self.pct_accuracy)

        if self.N_ <= 0:
            raise ValueError("The value of N_ should be higher than 0"
                             "N_ = {}".format(self.N_))

        # The value of Scaling coefficient (alpha) should be positive
        # to add more weight to the minority class
        if self.alpha <= 0:
            raise ValueError("The value of alpha should be higher than 0"
                             "alpha = {}".format(self.alpha))

        if not isinstance(self.alpha, np.float):
            raise TypeError("parameter alpha should be a float!")

        if self.pct_accuracy <= 0. or self.pct_accuracy > 1:
            raise ValueError(
                "The value of pct_accuracy should be higher than 0 and lower"
                " or equal to 1, "
                "pct_accuracy = {}".format(self.pct_accuracy))
