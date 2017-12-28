# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from pythonds.dcs.base import DCS


class LCA(DCS):
    """Local Classifier Accuracy (LCA).

    Evaluates the competence level of each individual classifiers and
    select the most competent one to predict the label of each test sample.
    The competence of each base classifier is calculated based on its local 
    accuracy with respect to some output class. Consider a classifier that assigns
    a test sample to class Ci. The competence is estimated by the percentage of the local training
    samples assigned to class Ci by this classifier that have been correctly labeled.
    

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

    aknn : Boolean (Default = False)
           Determines the type of KNN algorithm that is used. Set to true for the A-KNN method.

    selection_method : String (Default = "best")
                       Determines which method is used to select the base classifier
                       after the competences are estimated.

    diff_thresh : float (Default = 0.1)
                  Threshold to measure the difference between the competence level of the base
                  classifiers for the random and diff selection schemes. If the difference is lower than the
                  threshold, their performance are considered equivalent.

    version : String (Default = "woods")
              Change the implementation of the LCA according to Woods [1] or Britto [2] implementations.

    References
    ----------
    [1] Woods, Kevin, W. Philip Kegelmeyer, and Kevin Bowyer. "Combination of multiple classifiers
    using local accuracy estimates." IEEE transactions on pattern analysis and machine intelligence
    19.4 (1997): 405-410.

    [2] Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection of classifiers—a comprehensive
    review." Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """
    def __init__(self, pool_classifiers, k=7, DFP=False, with_IH=False, safe_k=None, IH_rate=0.30,
                 aknn=False, selection_method='best', diff_thresh=0.1, rng=np.random.RandomState(), version='woods'):

            super(LCA, self).__init__(pool_classifiers, k, DFP=DFP, with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate,
                                      aknn=aknn,
                                      selection_method=selection_method,
                                      diff_thresh=diff_thresh,
                                      rng=rng)

            if not isinstance(version, str):
                raise TypeError('Parameter version should be a string. version = ', type(version))

            version = version.lower()
            if version not in ['woods', 'britto']:
                raise ValueError('Invalid value for parameter "version". version should be either "woods" or "britto"')

            self.version = version
            self.name = 'Local Classifier Accuracy (LCA)'

    def estimate_competence(self, query):
        """estimate the competence of each base classifier ci
        the classification of the query sample using the local class accuracy method.

        Two versions of the LCA are considered for the competence estimates:

        Woods : In this algorithm the K-Nearest Neighbors of the test sample are estimated. Then, the
        local accuracy of the base classifiers is estimated by its classification accuracy taking into account
        only the samples belonging to the class wl in this neighborhood. In this case, wl is the predict class
        of the base classifier ci for the query sample.

        Britto : Is the algorithm presented in Britto et al. [2]. In this method, the neighborhood is composed
        of the K-Nearest Neighbors instances that belongs to a specific class wl.

        Returns an array containing the level of competence estimated using the LCA method
        for each base classifier. The size of the array is equals to the size of the pool of classifiers.

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        Returns
        -------
        competences : array = [n_classifiers] containing the competence level estimated
        for each base classifier
        """
        competences = np.zeros(self.n_classifiers)

        if self.version == 'britto':
            # the whole DSEL is considered until k samples of the predicted class if found
            dists, idx_neighbors = self._get_region_competence(query, k=self.n_samples)
        else:
            # consider only samples from the predicted class among the k-Nearest neighbors
            dists, idx_neighbors = self._get_region_competence(query)

        for clf_index, clf in enumerate(self.pool_classifiers):
            # Check if the dynamic frienemy pruning (DFP) should be used used
            if self.mask[clf_index]:
                result = []
                predicted_label = clf.predict(query)
                # Counter is needed for the survey method
                counter = 0
                for index in idx_neighbors:
                    # Get only neighbors from the same class as predicted by the
                    # classifier (clf) to form the region of competence
                    if self.DSEL_target[index] == predicted_label[0]:
                        result.append(self.processed_dsel[index][clf_index])
                        counter += 1

                    # This check is for britto implementation to use a maximum of k neighbors of the predicted class
                    if counter >= self.k:
                        break
                if len(result) == 0:
                    competences[clf_index] = 0
                else:
                    competences[clf_index] = np.mean(result)

        return competences
