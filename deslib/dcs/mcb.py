# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.dcs.base import DCS


class MCB(DCS):
    """Multiple Classifier Behaviour (MCB).

    The MCB method evaluates the competence level of each individual classifiers taking into account
    both the local accuracy of the base 

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
    Giacinto, Giorgio, and Fabio Roli. "Dynamic classifier selection based on multiple classifier behaviour."
    Pattern Recognition 34.9 (2001): 1879-1881.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    Huang, Yea S., and Ching Y. Suen. "A method of combining multiple experts for the recognition of unconstrained
    handwritten numerals." IEEE Transactions on Pattern Analysis and Machine Intelligence 17.1 (1995): 90-94.

    Huang, Yea S., and Ching Y. Suen. "The behavior-knowledge space method for combination of multiple classifiers."
    IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 1993.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.
    """

    def __init__(self, pool_classifiers, k=7, DFP=False, with_IH=False, safe_k=None, IH_rate=0.30,
                 similarity_threshold=0.7, selection_method='diff', diff_thresh=0.1, rng=np.random.RandomState()):

        super(MCB, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate,
                                  selection_method=selection_method,
                                  diff_thresh=diff_thresh,
                                  rng=rng)

        if not isinstance(similarity_threshold, float):
            raise TypeError('The parameter similarity_threshold must be a float.'
                            ' similarity_threshold = ', type(similarity_threshold))

        if similarity_threshold > 1 or similarity_threshold < 0:
            raise ValueError('The parameter similarity_threshold should be between [0 and 1]. '
                             'similarity_threshold = ', similarity_threshold)

        self.similarity_threshold = similarity_threshold
        self.name = 'Multiple Classifier Behaviour (MCB)'

    def estimate_competence(self, query):
        """estimate the competence of each base classifier ci
        the classification of the query sample using the Multiple Classifier Behaviour criterion.

        The region of competence in this method is estimated taking into account the feature space and the decision
        space (using the behaviour knowledge space method [4]). First, the k-Nearest Neighbors of the query sample
        are defined in the feature space to compose the region of competence. Then, the similarity in the BKS space
        between the query and the instances in its region of competence are estimated. Instances with similarity lower
        than a predefined threshold are removed from the region of competence.

        Then, the competence level of the base classifiers are estimated based on their classification accuracy in the
        final region of competence.


        Returns an array containing the level of competence estimated using the MCB method
        for each base classifier. The size of the array is equals to the size of the generated_pool of classifiers.

        Parameters
        ----------
        query : array cf shape  = [n_features]
                The query sample
        Returns
        -------
        competences : array of shape = [n_classifiers]
                      The competence level estimated for each base classifier
        """
        dists, idx_neighbors = self._get_region_competence(query)
        competences = np.zeros(self.n_classifiers)

        BKS_query = self._BKS_transform(query)

        # Use the BKS to filter the competence region
        selected_idx = []
        for sample_index in idx_neighbors:
            T = (self.BKS_dsel[sample_index][:] == BKS_query)
            S = sum(T) / self.n_classifiers
            if S > self.similarity_threshold:
                selected_idx.append(sample_index)

        # Use the whole neighborhood if no sample is selected to form the region of competence
        if len(selected_idx) == 0:
            selected_idx = idx_neighbors
        # Estimate the classifier competence for the filtered region of competence
        for clf_index in range(self.n_classifiers):

            # Check if the dynamic frienemy pruning (DFP) should be used used
            if self.DFP_mask[clf_index]:
                clf_competence = [self.processed_dsel[sample_idx][clf_index] for sample_idx in selected_idx]
                competences[clf_index] = np.mean(np.array(clf_competence))

        return competences

    def _BKS_transform(self, query):
        """Transform the query sample to the decision space using the Behaviour Knowledge Space (BKS) method [4].

        The BKS space is a vector = [n_classifiers], in which each position i is equals to the class label predicted
        by the base classifier ci for the query sample.

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        Returns
        -------
        bks_query : array = [n_classifiers] containing the transformation of the query sample to the decision space
        """
        BKS_query = np.zeros(self.n_classifiers)
        for clf_index, clf in enumerate(self.pool_classifiers):
            BKS_query[clf_index] = clf.predict(query.reshape(1, -1))[0]

        return BKS_query
