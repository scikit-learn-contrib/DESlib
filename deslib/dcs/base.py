from abc import ABCMeta

import numpy as np

from deslib.base import DS
from deslib.util.aggregation import majority_voting, predict_proba_ensemble


class DCS(DS):
    """Base class for a Dynamic Classifier Selection (dcs) method.
    All dynamic classifier selection classes should inherit from this class.

    Warning: This class should not be used directly, use derived classes instead.

    Parameters
    ----------
    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support methods "predict" and "predict_proba".

    k : int (Default = 7)
        Number of neighbors used to estimate the competence of the base classifiers.

    DFP : Boolean (Default = False)
          Determines if the dynamic frienemy pruning is applied.

    with_IH : Boolean (Default = False)
              Whether the hardness level of the region of competence is used to decide
              between using the DS algorithm or the KNN for classification of a given query sample.

    safe_k : int (default = None)
             The size of the indecision region.

    IH_rate : float (default = 0.3)
              Hardness threshold. If the hardness level of the competence region is lower than
              the IH_rate the KNN classifier is used. Otherwise, the DS algorithm is used for classification.

    selection_method : String (Default = "best")
                       Determines which method is used to select the base classifier
                       after the competences are estimated.

    diff_thresh : float (Default = 0.1)
                  Threshold to measure the difference between the competence level of the base
                  classifiers for the random and diff selection schemes. If the difference is lower than the
                  threshold, their performance are considered equivalent.

    rng : numpy.random.RandomState instance
          Random number generator to assure reproducible results.

    References
    ----------
    Woods, Kevin, W. Philip Kegelmeyer, and Kevin Bowyer. "Combination of multiple classifiers
    using local accuracy estimates." IEEE transactions on pattern analysis and machine intelligence
    19.4 (1997): 405-410.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    G. Giacinto and F. Roli, Methods for Dynamic Classifier Selection
    10th Int. Conference on Image Analysis and Proc., Venice, Italy (1999), 659-664.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """

    __metaclass__ = ABCMeta

    def __init__(self, pool_classifiers, k=7, DFP=False, safe_k=None, with_IH=False, IH_rate=0.30,
                 selection_method='best', diff_thresh=0.1, rng=np.random.RandomState()):

        if not isinstance(selection_method, str):
            raise TypeError('The parameter selection_method should be a string.'
                            ' selection_method = ', type(selection_method))

        selection_method = selection_method.lower()

        if selection_method not in ['best', 'all', 'random', 'diff']:
            raise ValueError('Invalid value for parameter "selection_method." The possible values are: '
                             '"best", "all", "random", "diff"')

        if not isinstance(diff_thresh, float):
            raise TypeError('The parameter diff_thresh should be a float. diff_thresh = ', diff_thresh)

        if diff_thresh >= 0.5 or diff_thresh < 0.0 or np.isnan(diff_thresh):
            raise ValueError('diff_thresh should be lower than 0.5. diff_thresh = ', diff_thresh)

        super(DCS, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH,
                                  safe_k=safe_k, IH_rate=IH_rate)
        self.selection_method = selection_method
        self.diff_thresh = diff_thresh
        self.rng = rng

    def estimate_competence(self, query):
        """estimate the competence of each base classifier for the classification of the query sample.

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        Returns
        -------
        competences : array of shape = [n_classifiers]
                      The competence level estimated for each base classifier in the pool
        """
        pass

    def select(self, competences):
        """Select the most competent classifier for the classification of the query sample given the competence level
        estimates. Four selection schemes are available.

        Best : The base classifier with the highest competence level is selected. In cases where more than one
        base classifier achieves the same competence level, the one with the lowest index is selected. This method
        is the standard for the LCA, OLA, MLA techniques.

        Diff : Select the base classifier that is significantly better than the others in the pool (when the difference
        between its competence level and the competence level of the other base classifiers is higher than a predefined
        threshold). If no base classifier is significantly better, the base classifier is selected randomly among the
        member with equivalent competence level.

        Random : Selects a random base classifier among all base classifiers that achieved the same competence level.

        ALL : all base classifiers with the max competence level estimates are selected (note that in this case the
        dcs technique becomes a des).

        Parameters
        ----------
        competences : array = [n_classifiers] containing the estimated competence level for the base classifiers

        Returns
        -------
        selected_clf : index of the selected base classifier(s)

        """
        selected_clf = []
        best_index = np.argmax(competences)

        if self.selection_method == 'best':
            # Select the classifier with highest competence level
            selected_clf = best_index

        elif self.selection_method == 'diff':
            """Selects a base classifier if its competence level is significant better than the rest. 
            If there is no such classifier, select randomly a base model.


             the best classifier will always have diff < diff_thresh. In a case it is
             superior than all others, it will be the only member selected. Otherwise,
             a random classifier from this list is selected
            """
            best_competence = np.max(competences)
            diff = best_competence - competences
            indices = [idx for idx, _ in enumerate(diff) if diff[idx] < self.diff_thresh]
            if len(indices) == 0:
                indices = range(self.n_classifiers)

            selected_clf = self.rng.choice(indices)

        elif self.selection_method == 'random':
            # Select a random classifier among all with same competence level
            indices = [idx for idx, competence in enumerate(competences) if competence == competences[best_index]]
            selected_clf = self.rng.choice(indices)

        elif self.selection_method == 'all':
            # select all base classifiers with max competence estimates.
            selected_clf = [idx for idx, competence in enumerate(competences) if competence == competences[best_index]]

        return selected_clf

    def classify_instance(self, query):
        """Predicts the class label of the corresponding query sample.

        If self.mode == "all", the majority voting scheme is used to aggregate the predictions of all classifiers with
        the max competence level estimate.

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        Returns
        -------
        The predicted label of the query
        """
        competences = self.estimate_competence(query)
        if self.selection_method != 'all':
            # only one classifier is selected
            clf_index = self.select(competences)
            predicted_label = self.pool_classifiers[clf_index].predict(query)[0]
        else:
            # Selected ensemble of classifiers is combined using Majority Voting
            indices = self.select(competences)
            classifier_ensemble = self._get_classifier_ensemble(indices)
            predicted_label = majority_voting(classifier_ensemble, query)

        return predicted_label

    def predict_proba_instance(self, query):
        """Predicts the posterior probabilities of the corresponding query sample.

        If self.mode == "all", get the probability estimates of the selected ensemble. Otherwise,
        the technique gets the probability estimates from the selected base classifier

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        Returns
        -------
        predicted_proba : array = [n_classes] with the probability estimates for all classes
        """
        competences = self.estimate_competence(query)
        if self.selection_method != 'all':
            # only one classifier is selected
            clf_index = self.select(competences)
            predicted_proba = self.pool_classifiers[clf_index].predict_proba(query)
        else:
            # Selected ensemble of classifiers is combined using Majority Voting
            indices = self.select(competences)
            classifier_ensemble = self._get_classifier_ensemble(indices)
            predicted_proba = predict_proba_ensemble(classifier_ensemble, query)

        return predicted_proba
