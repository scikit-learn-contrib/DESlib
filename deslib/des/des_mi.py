# coding=utf-8

# Author: Qiushi Wang <wqiushi@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.preprocessing import normalize

from deslib.base import DS
from deslib.util.aggregation import majority_voting_rule


class DESMI(DS):
    """Dynamic ensemble Selection for multi-class imbalanced datasets (DES-MI).


    Parameters
    ----------
    pool_classifiers : list of classifiers (Default = None)
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       Each base classifiers should support the method "predict".
                       If None, then the pool of classifiers is a bagging classifier.

    k : int (Default = 7)
        Number of neighbors used to estimate the competence of the base classifiers.

    pct_accuracy : float (Default = 0.4)
        Percentage of base classifiers selected based on accuracy


    alpha : float (Default = 0.9)
            Scaling coefficient to regulate the weight value

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

    random_state : int, RandomState instance or None, optional (default=None)
                   If int, random_state is the seed used by the random number generator;
                   If RandomState instance, random_state is the random number generator;
                   If None, the random number generator is the RandomState instance used
                   by `np.random`.

    References
    ----------
    García, S.; Zhang, Z.-L.; Altalhi, A.; Alshomrani, S. & Herrera, F. "Dynamic ensemble selection for multi-class
    imbalanced datasets." Information Sciences, 2018, 445-446, 22 - 37

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.
    """

    def __init__(self, pool_classifiers=None, k=7, pct_accuracy=0.4, alpha=0.9, DFP=False, with_IH=False, safe_k=None,
                 IH_rate=0.30, random_state=None):

        super(DESMI, self).__init__(pool_classifiers, k,
                                    DFP=DFP,
                                    with_IH=with_IH,
                                    safe_k=safe_k,
                                    IH_rate=IH_rate,
                                    random_state=random_state)

        self.name = 'Dynamic Ensemble Selection for multi-class imbalanced datasets (DES-MI)'
        self.alpha = alpha
        self.pct_accuracy = pct_accuracy

    def estimate_competence(self, query, predictions=None):
        """estimate the competence level of each base classifier :math:`c_{i}` for
        the classification of the query sample.

        The competence is estimated using the accuracy criteria. The classification accuracy of the base
        classifiers in the region of competence is estimated. The accuracy is estimated by the weighted results
        of classifiers who correctly classify the members in the competence region. The weight of member 'x_i' is
        related to the number of samples of the same class of 'x_i' in the training dataset. For detail, please see
        the first reference, algorithm 2.

        The method returns one array which contains the accuracy of each base classifier.

        Parameters
        ----------
        query : array cf shape  = [n_samples, n_features]
                The query sample.

        predictions : array of shape = [n_samples, n_classifiers]
                      Predictions of the base classifiers for all test examples.

        Returns
        -------
        accuracy : array of shape = [n_samples, n_classifiers}
                   Local Accuracy estimates (competences) of the base classifiers for all query samples.


        """
        _, idx_neighbors = self._get_region_competence(query)
        # calculate the weight
        class_frequency = np.bincount(self.DSEL_target_)
        targets = self.DSEL_target_[idx_neighbors]       # [n_samples, K_neighbors]
        num = class_frequency[targets]
        weight = 1./(1 + np.exp(self.alpha * num))
        weight = normalize(weight, norm='l1')
        correct_num = self.DSEL_processed_[idx_neighbors, :]
        correct = np.zeros((query.shape[0], self.k_, self.n_classifiers_))
        for i in range(self.n_classifiers_):
            correct[:, :, i] = correct_num[:, :, i] * weight

        # calculate the classifiers mean accuracy for all samples/base classifier
        accuracy = np.mean(correct, axis=1)

        return accuracy

    def select(self, accuracy):
        """Select an ensemble containing the N_ most accurate classifiers for the classification of the query sample.

        Parameters
        ----------
        accuracy : array of shape = [n_samples, n_classifiers]
                   Local Accuracy estimates (competence) of each base classifiers for all query samples.

        Returns
        -------
        selected_classifiers : array of shape = [n_samples, self.N_]
                               Matrix containing the indices_ of the N_ selected base classifier for each test example.
        """
        # Check if the accuracy and diversity arrays have the correct dimensionality.
        if accuracy.ndim < 2:
            accuracy = accuracy.reshape(1, -1)

        # sort the array to remove the most accurate classifiers
        competent_indices = np.argsort(accuracy, axis=1)[:, ::-1][:, 0:self.N_]

        return competent_indices

    def classify_with_ds(self, query, predictions, probabilities=None):
        """Predicts the label of the corresponding query sample.

        Parameters
        ----------
        query : array of shape = [n_samples, n_features]
                The test examples

        predictions : array of shape = [n_samples, n_classifiers]
                      Predictions of the base classifiers for all test examples

        probabilities : array of shape = [n_samples, n_classifiers, n_classes]
                        Probabilities estimates of each base classifier for all test examples.

        Notes
        ------
        Different than other DES techniques, this method only select N_ candidates from the pool of classifiers.

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

        accuracy = self.estimate_competence(query)

        if self.DFP:
            accuracy = accuracy * self.DFP_mask

        selected_classifiers = self.select(accuracy)
        votes = predictions[np.arange(predictions.shape[0])[:, None], selected_classifiers]
        predicted_label = majority_voting_rule(votes)

        return predicted_label

    def predict_proba_with_ds(self, query, predictions, probabilities):
        """Predicts the posterior probabilities of the corresponding query sample.

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
        predicted_proba : array = [n_samples, n_classes]
                          Probability estimates for all test examples.
        """

        if query.shape[0] != probabilities.shape[0]:
            raise ValueError('The arrays query and predictions must have the same number of samples. query.shape is {}'
                             'and predictions.shape is {}' .format(query.shape, predictions.shape))

        accuracy = self.estimate_competence(query)

        if self.DFP:
            accuracy = accuracy * self.DFP_mask

        selected_classifiers = self.select(accuracy)
        ensemble_proba = probabilities[np.arange(probabilities.shape[0])[:, None], selected_classifiers, :]

        predicted_proba = np.mean(ensemble_proba, axis=1)

        return predicted_proba

    def _validate_parameters(self):
        """Check if the parameters passed as argument are correct.

        The values of N_ should be higher than 0.
        ----------
        """
        super(DESMI, self)._validate_parameters()

        self.N_ = int(self.n_classifiers_ * self.pct_accuracy)

        if self.N_ <= 0:
            raise ValueError("The values of N_ should be higher than 0"
                             "N_ = {}" .format(self.N))
        
        # The value of Scaling coefficient (alpha) should be positive to add more weight to the minority clas
        if self.alpha <= 0:
            raise ValueError("The values of alpha should be higher than 0"
                             "alpha = {}".format(self.alpha))
