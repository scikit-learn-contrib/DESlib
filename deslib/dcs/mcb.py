# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.dcs.base import DCS


class MCB(DCS):
    """Multiple Classifier Behaviour (MCB).

    The MCB method evaluates the competence level of each individual classifiers taking into account the local
    accuracy of the base classifier in the region of competence. The region of competence is defined using the
    k-NN and behavioral knowledge space (BKS) method. First the k-nearest neighbors of the test sample are computed.
    Then, the set containing the k-nearest neighbors is filtered based on the similarity of the query sample and its
    neighbors using the decision space (BKS representation).

    A single classifier :math:`c_{i}` is selected only if its competence level is
    significantly higher than that of the other base classifiers in the pool (higher than a pre-defined threshold).
    Otherwise, all classifiers in the pool are combined using the majority voting rule. The selection methodology can
    be modified by changing the hyper-parameter selection_method.

    Parameters
    ----------
    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support the method "predict".

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

    def estimate_competence(self, query, predictions=None):
        """estimate the competence of each base classifier :math:`c_{i}` for
        the classification of the query sample using the Multiple Classifier Behaviour criterion.

        The region of competence in this method is estimated taking into account the feature space and the decision
        space (using the behaviour knowledge space method [4]). First, the k-Nearest Neighbors of the query sample
        are defined in the feature space to compose the region of competence. Then, the similarity in the BKS space
        between the query and the instances in the region of competence are estimated using the following equations:

        .. math:: S(\\tilde{\\mathbf{x}}_{j},\\tilde{\\mathbf{x}}_{k}) = \\frac{1}{M}
            \\sum\\limits_{i = 1}^{M}T(\\mathbf{x}_{j},\\mathbf{x}_{k})

        .. math:: T(\\mathbf{x}_{j},\\mathbf{x}_{k}) = \\left\\{\\begin{matrix} 1 & \\text{if} &
            c_{i}(\\mathbf{x}_{j}) =  c_{i}(\\mathbf{x}_{k}),\\\\
            0 & \\text{if} & c_{i}(\\mathbf{x}_{j}) \\neq  c_{i}(\\mathbf{x}_{k}). \\end{matrix}\\right.

        Where :math:`S(\\tilde{\\mathbf{x}}_{j},\\tilde{\\mathbf{x}}_{k})` denotes the similarity between two samples
        based on the behaviour knowledge space method (BKS). Instances with similarity lower than a predefined threshold
        are removed from the region of competence. The competence level of the base classifiers are estimated as their
        classification accuracy in the final region of competence.

        Parameters
        ----------
        query : array cf shape  = [n_samples, n_features]
                The test samples.

        predictions : array of shape = [n_samples, n_classifiers]
                      Predictions of the base classifiers for the test examples.

        Returns
        -------
        competences : array of shape = [n_samples, n_classifiers]
                      Competence level estimated for each base classifier and test example.
        """

        _, idx_neighbors = self._get_region_competence(query)

        # Use the pre-compute decisions to transform the query to the BKS space
        BKS_query = predictions

        T = (self.BKS_dsel[idx_neighbors] == BKS_query.reshape(BKS_query.shape[0], -1, BKS_query.shape[1]))
        S = np.sum(T, axis=2) / self.n_classifiers

        # get a mask with the neighbors that will be considered for the competence estimation for all samples.
        boolean_mask = (S > self.similarity_threshold)
        boolean_mask[~np.any(boolean_mask, axis=1), :] = True
        # Expanding this mask to the third axis (n_classifiers) since it is the same for each classifier.
        boolean_mask = np.repeat(np.expand_dims(boolean_mask, axis=2), self.n_classifiers, axis=2)

        # Use the masked array mean to take into account the removed neighbors
        processed_pred = np.ma.MaskedArray(self.processed_dsel[idx_neighbors, :], mask=~boolean_mask)
        competences = np.ma.mean(processed_pred, axis=1)

        return competences
