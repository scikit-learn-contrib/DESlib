from abc import ABCMeta

import numpy as np

from deslib.base import BaseDS
from deslib.util.aggregation import majority_voting_rule
from sklearn.utils.validation import check_random_state


class BaseDCS(BaseDS):
    """Base class for a Dynamic Classifier Selection (dcs) method.
    All dynamic classifier selection classes should inherit from this class.

    Warning: This class should not be used directly, use derived classes
    instead.

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

    selection_method : String (Default = "best")
        Determines which method is used to select the base classifier after
        the competences are estimated.

    diff_thresh : float (Default = 0.1)
        Threshold to measure the difference between the competence level of the
        base classifiers for the random and diff selection schemes. If the
        difference is lower than the threshold, their performance are
        considered equivalent.

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
    Woods, Kevin, W. Philip Kegelmeyer, and Kevin Bowyer. "Combination of
    multiple classifiers using local accuracy estimates." IEEE transactions on
    pattern analysis and machine intelligence 19.4 (1997): 405-410.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection
    of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    G. Giacinto and F. Roli, Methods for Dynamic Classifier Selection. 10th
    Int. Conference on Image Analysis and Proc., Venice, Italy (1999), 659-664.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
    selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """

    __metaclass__ = ABCMeta

    def __init__(self, pool_classifiers=None, k=7, DFP=False, safe_k=None,
                 with_IH=False, IH_rate=0.30, selection_method='best',
                 diff_thresh=0.1, random_state=None, knn_classifier='knn',
                 DSEL_perc=0.5):

        super(BaseDCS, self).__init__(pool_classifiers=pool_classifiers, k=k,
                                      DFP=DFP, with_IH=with_IH, safe_k=safe_k,
                                      IH_rate=IH_rate,
                                      random_state=random_state,
                                      knn_classifier=knn_classifier,
                                      DSEL_perc=DSEL_perc)

        self.selection_method = selection_method
        self.diff_thresh = diff_thresh

    def estimate_competence(self, query, neighbors, distances=None,
                            predictions=None):
        """Estimate the competence of each base classifier for the
        classification of the query sample.

        Parameters
        ----------
        query : array of shape  = [n_samples, n_features]
            The test examples.

        neighbors : array of shale = [n_samples, n_neighbors]
            Indices of the k nearest neighbors according for each test sample

        distances : array of shale = [n_samples, n_neighbors]
            Distances of the k nearest neighbors according for each test sample

        predictions : array of shape = [n_samples, n_classifiers]
            Predictions of the base classifiers for the test examples.

        Returns
        -------
        competences : array of shape = [n_samples, n_classifiers]
            Competence level estimated for each base classifier and test
            example.
        """
        pass

    def select(self, competences):
        """Select the most competent classifier for the classification of the
        query sample given the competence level estimates. Four selection
        schemes are available.

        Best : The base classifier with the highest competence level is
        selected. In cases where more than one base classifier achieves the
        same competence level, the one with the lowest index is selected. This
        method is the standard for the LCA, OLA, MLA techniques.

        Diff : Select the base classifier that is significantly better than the
        others in the pool (when the difference between its competence level
        and the competence level of the other base classifiers is higher than a
        predefined threshold). If no base classifier is significantly better,
        the base classifier is selected randomly among the member with
        equivalent competence level.

        Random : Selects a random base classifier among all base classifiers
        that achieved the same competence level.

        ALL : all base classifiers with the max competence level estimates are
        selected (note that in this case the
        DCS technique becomes a DES technique).

        Parameters
        ----------
        competences : array of shape = [n_samples, n_classifiers]
            Competence level estimated for each base classifier and test
            example.

        Returns
        -------
        selected_classifiers : array of shape [n_samples]
            Indices of the selected base classifier for each sample. If the
            selection_method is set to 'all', a boolean matrix is returned,
            containing True for the selected base classifiers, otherwise false.

        """
        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        selected_classifiers = []
        best_index = np.argmax(competences, axis=1)

        if self.selection_method == 'best':
            # Select the classifier with highest competence level
            selected_classifiers = best_index

        elif self.selection_method == 'diff':
            # Selects a base classifier if its competence level is significant
            # better than the rest. If there is no such classifier, select
            # randomly a base model.
            #
            # the best classifier will always have diff < diff_thresh. In a
            # case it is superior than all others, it will be the only member
            # selected. Otherwise, a random classifier from this list is
            # selected.

            rng = check_random_state(self.random_state)
            best_competence = competences[
                np.arange(competences.shape[0]), best_index]
            # best_competence = np.max(competences)
            diff = best_competence.reshape(-1, 1) - competences
            # TODO: Improve this part of the code
            selected_classifiers = np.zeros(diff.shape[0], dtype=np.int)
            for row in range(diff.shape[0]):
                diff_list = list(diff[row, :])
                indices = [idx for idx, _ in enumerate(diff_list) if
                           diff_list[idx] < self.diff_thresh]

                if len(indices) == 0:
                    indices = range(self.n_classifiers_)

                selected_classifiers[row] = rng.choice(indices)

        elif self.selection_method == 'random':
            # TODO: Improve this part of the code
            rng = check_random_state(self.random_state)
            selected_classifiers = np.zeros(competences.shape[0], dtype=np.int)
            best_competence = competences[
                np.arange(competences.shape[0]), best_index]
            for row in range(competences.shape[0]):
                competence_list = list(competences[row, :])

                # Select a random classifier among all with same competence
                # level
                indices = [idx for idx, _ in enumerate(competence_list) if
                           competence_list[idx] == best_competence[row]]

                selected_classifiers[row] = rng.choice(indices)

        elif self.selection_method == 'all':
            # select all base classifiers with max competence estimates.
            max_value = np.max(competences, axis=1)
            selected_classifiers = (
                        competences == max_value.reshape(competences.shape[0],
                                                         -1))

        return selected_classifiers

    def classify_with_ds(self, query, predictions, probabilities=None,
                         neighbors=None, distances=None, DFP_mask=None):
        """Predicts the class label of the corresponding query sample.

        If self.selection_method == "all", the majority voting scheme is used
        to aggregate the predictions of all classifiers with the max competence
        level estimates for each test examples.

        Parameters
        ----------
        query : array of shape  = [n_samples, n_features]
            The test examples.

        predictions : array of shape = [n_samples, n_classifiers]
            Predictions of the base classifiers for all test examples

        probabilities : array of shape = [n_samples, n_classifiers, n_classes]
            Probabilities estimates of each base classifier for all test
            examples (For methods that always require probabilities from the
            base classifiers)

        neighbors : array of shale = [n_samples, n_neighbors]
            Indices of the k nearest neighbors according for each test sample

        distances : array of shale = [n_samples, n_neighbors]
            Distances of the k nearest neighbors according for each test sample

        DFP_mask : array of shape = [n_samples, n_classifiers]
            Mask containing 1 for the selected base classifier and 0 otherwise.

        Returns
        -------
        predicted_label : array of shape = [n_samples]
            The predicted label for each query
        """
        if query.ndim < 2:
            query = query.reshape(1, -1)

        if predictions.ndim < 2:
            predictions = predictions.reshape(1, -1)

        if query.shape[0] != predictions.shape[0]:
            raise ValueError(
                'The arrays query and predictions must have the same shape. '
                'query.shape is {}'
                'and predictions.shape is {}'.format(query.shape,
                                                     predictions.shape))

        competences = self.estimate_competence(query, neighbors,
                                               distances=distances,
                                               predictions=predictions)

        if self.DFP:
            competences = competences * DFP_mask

        if self.selection_method != 'all':
            # only one classifier is selected
            clf_index = self.select(competences)
            predicted_label = predictions[
                np.arange(predictions.shape[0]), clf_index]
        else:
            # Selected ensemble of classifiers is combined using Majority
            # Voting
            indices = self.select(competences)
            votes = np.ma.MaskedArray(predictions, ~indices)
            predicted_label = majority_voting_rule(votes)

        return predicted_label

    def predict_proba_with_ds(self, query, predictions, probabilities,
                              neighbors=None, distances=None, DFP_mask=None):
        """Predicts the posterior probabilities of the corresponding query
        sample.

        If self.selection_method == "all", get the probability estimates of the
        selected ensemble. Otherwise, the technique gets the probability
        estimates from the selected base classifier

        Parameters
        ----------
        query : array of shape  = [n_samples, n_features]
            The test examples.

        predictions : array of shape = [n_samples, n_classifiers]
            Predictions of the base classifiers for all test examples

        probabilities : array of shape = [n_samples, n_classifiers, n_classes]
            The predictions of each base classifier for all samples (For
            methods that always require probabilities from the base
            classifiers).

        neighbors : array of shape = [n_samples, n_neighbors]
            Indices of the k nearest neighbors according for each test sample

        distances : array of shale = [n_samples, n_neighbors]
            Distances of the k nearest neighbors according for each test sample

        DFP_mask : array of shape = [n_samples, n_classifiers]
           Mask containing 1 for the selected base classifier and 0 otherwise.

        Returns
        -------
        predicted_proba: array of shape = [n_samples, n_classes]
            Posterior probabilities estimates for each test example.
        """
        if query.shape[0] != probabilities.shape[0]:
            raise ValueError(
                'The arrays query and predictions must have the same number '
                'of samples. query.shape is {}'
                'and predictions.shape is {}'.format(query.shape,
                                                     predictions.shape))

        competences = self.estimate_competence(query, neighbors,
                                               distances=distances,
                                               predictions=predictions)

        if self.DFP:
            competences = competences * DFP_mask

        if self.selection_method != 'all':
            # only one classifier is selected
            clf_index = self.select(competences)
            predicted_proba = probabilities[
                np.arange(probabilities.shape[0]), clf_index]
        else:
            # Selected ensemble of classifiers is combined using average
            # probability
            selected_classifiers = self.select(competences)

            # Broadcast the selected classifiers mask (to cover the last axis
            # (nClasses):
            selected_classifiers = np.expand_dims(selected_classifiers, axis=2)
            selected_classifiers = np.broadcast_to(selected_classifiers,
                                                   probabilities.shape)
            masked_proba = np.ma.MaskedArray(probabilities,
                                             ~selected_classifiers)

            predicted_proba = np.mean(masked_proba, axis=1)

        return predicted_proba

    def _validate_parameters(self):

        super(BaseDCS, self)._validate_parameters()

        if not isinstance(self.selection_method, str):
            raise TypeError(
                'The parameter selection_method should be a string.'
                ' selection_method = ', type(self.selection_method))

        if self.selection_method not in ['best', 'all', 'random', 'diff']:
            raise ValueError(
                'Invalid value for parameter "selection_method." '
                'The possible values are: '
                '"best", "all", "random", "diff"')

        if not isinstance(self.diff_thresh, float):
            raise TypeError(
                'The parameter diff_thresh should be a float. diff_thresh = ',
                self.diff_thresh)

        if self.diff_thresh >= 0.5 or self.diff_thresh < 0.0 or np.isnan(
                self.diff_thresh):
            raise ValueError(
                'diff_thresh should be lower than 0.5. diff_thresh = ',
                self.diff_thresh)
