from abc import ABCMeta

from pythonds.base import DS


class DES(DS):
    """Base class for a Dynamic Ensemble Selection (des) technique.
    All dynamic ensemble selection techniques should inherit from this class.

    Warning: This class should not be instantiated directly, use derived classes instead.

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
              Whether the hardness level of the region of competence is used to decide between
              using the DS algorithm or the KNN for classification of a given query sample.

    safe_k : int (default = None)
             The size of the indecision region.

    IH_rate : float (default = 0.3)
              Hardness threshold. If the hardness level of the competence region is lower than
              the IH_rate the KNN classifier is used. Otherwise, the DS algorithm is used for classification.

    aknn : Boolean (Default = False)
           Determines the type of KNN algorithm that is used. Set to true for the A-KNN method.

    mode : String (Default = "selection")
           Whether the technique will perform dynamic selection,
           dynamic weighting or an hybrid approach for classification.

    References
    ----------
    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.
    """
    __metaclass__ = ABCMeta

    def __init__(self, pool_classifiers, k=7, DFP=False, with_IH=False,
                 safe_k=None, IH_rate=0.30, aknn=False, mode='selection'):

        super(DES, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH,
                                  safe_k=safe_k, IH_rate=IH_rate, aknn=aknn)

        mode = mode.lower()

        if mode not in ['selection', 'hybrid', 'weighting']:
            raise ValueError('Invalid value for parameter "version"')

        self.version = mode

    def estimate_competence(self, query):
        """estimate the competence of each base classifier ci
        the classification of the query sample x.
        Returns an array containing the level of competence estimated
        for each base classifier. The size of the vector is equals to
        the size of the generated_pool of classifiers.

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        Returns
        -------
        competences : array = [n_classifiers] containing the competence level estimated
        for each base classifier
        """
        pass

    def select(self, competences):
        """Select the most competent classifier for
        the classification of the query sample x.
        The most competent classifier (dcs) or an ensemble
        with the most competent classifiers (des) is returned

        Parameters
        ----------
        competences : array = [n_classifiers] containing the estimated competence level for the base classifiers

        Returns
        -------
        indices : index of the selected base classifier(s)

        """
        pass

    def classify_instance(self, query):
        """Predicts the label of the corresponding query sample.
        Returns the predicted label.

        If self.version == "selection", the selected ensemble is combined using the
        majority voting rule

        If self.version == "weighting", all base classifiers are used for classification, however their influence
        in the final decision are weighted according to their estimated competence level. The weighted majority voting
        scheme is used to combine the decisions of the base classifiers.

        If self.version == "hybrid",  A hybrid Dynamic selection and weighting approach is used. First an
        ensemble with the competent base classifiers are selected. Then, their decisions are aggregated using the
        weighted majority voting rule according to its competence level estimates.

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        Returns
        -------
        predicted_label: The predicted label of the query
        """
        competences = self.estimate_competence(query)
        if self.version == "selection":
            indices = self.select(competences)
            predicted_label = self.majority_voting(indices, query)

        elif self.version == "weighting":
            indices = range(self.n_classifiers)
            predicted_label = self.weighted_majority_voting(indices, competences, query)

        else:
            indices = self.select(competences)
            predicted_label = self.weighted_majority_voting(indices, competences, query)

        return predicted_label

    def predict_proba_instance(self, query):
        """Predicts the posterior probabilities of the corresponding query sample.
        Returns the probability estimates of each class.

        If self.version == "selection", the selected ensemble is combined using the
        majority voting rule

        If self.version == "weighting", all base classifiers are used for classification, however their influence
        in the final decision are weighted according to their estimated competence level. The weighted majority voting
        scheme is used to combine the decisions of the base classifiers.

        If self.version == "Hybrid",  A hybrid Dynamic selection and weighting approach is used. First an
        ensemble with the competent base classifiers are selected. Then, their decisions are aggregated using the
        weighted majority voting rule according to its competence level estimates.

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        Returns
        -------
        predicted_proba : array = [n_classes] with the probability estimates for all classes
        """
        if self.version == "selection":
            indices, _ = self.select(query)
            predicted_proba = self.predict_proba_ensemble(query, indices)

        elif self.version == "weighting":
            competences = self.estimate_competence(query)
            indices = range(self.n_classifiers)
            predicted_proba = self.predict_proba_ensemble(query, indices, competences)

        else:
            indices, competences = self.select(query)
            predicted_proba = self.predict_proba_ensemble(query, indices, competences)

        return predicted_proba
