# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import warnings

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from pythonds.des.base import DES


class METADES(DES):
    """Meta learning for dynamic ensemble selection (META-DES).

    This method works selects all classifiers that correctly classified at least
    one sample belonging to the region of competence of the test sample x. Each
    selected classifier has a number of votes equals to the number of samples in the
    region of competence that it predicts the correct label.

    Parameters
    ----------
    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support methods "predict" and "predict_proba".

    k : int (Default = 7)
        Number of neighbors used to estimate the competence of the base classifiers.
    
    kp : int (Default = 5)
         Number of output profiles used to estimate the competence of the base classifiers.

    aknn : Boolean (Default = False)
           Determines the type of KNN algorithm that is used. set to true for the A-KNN method.

    mode : String (Default = "selection")
              Determines the mode of META-des that is used (selection, weighting or hybrid).

    DFP : Boolean (Default = False)
          Determines if the dynamic frienemy pruning is applied.

    with_IH : Boolean (Default = False)
              Whether the hardness level of the region of competence is used to decide between using the DS
              algorithm or the KNN for classification of a given query sample.

    safe_k : int (default = None)
             The size of the indecision region.

    IH_rate : float (default = 0.3)
              Hardness threshold. If the hardness level of the competence region is lower than
              the IH_rate the KNN classifier is used. Otherwise, the DS algorithm is used for classification.

    References
    ----------
    Cruz, R.M., Sabourin, R., Cavalcanti, G.D. and Ren, T.I., 2015. META-DES: A dynamic ensemble selection framework
    using meta-learning. Pattern Recognition, 48(5), pp.1925-1935.

    Cruz, R.M., Sabourin, R. and Cavalcanti, G.D., 2015, July. META-des. H: a dynamic ensemble selection technique
    using meta-learning and a dynamic weighting approach. In Neural Networks (IJCNN), 2015 International Joint
    Conference on (pp. 1-8).

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """
    def __init__(self, pool_classifiers, selector=MultinomialNB(), k=7, kp=5, Hc=0.8, gamma=0.5, mode='hybrid',
                 DFP=False, with_IH=False, safe_k=None, IH_rate=0.30, aknn=False):

        super(METADES, self).__init__(pool_classifiers, k, DFP=DFP,
                                      with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate, aknn=aknn)
        mode.lower()
        assert Hc > 0.5
        assert gamma > 0
        assert mode in ['selection', 'hybrid', 'weighting']
        self.name = 'META-des'
        self.version = mode
        self.kp = kp
        self.Hc = Hc
        self.gamma = gamma
        if selector is None:
            warnings.warn("No classifier model passed for the Meta-Classifier. Using a Naive Bayes instead")
            self.meta_classifier = MultinomialNB()
        else:
            self.meta_classifier = selector

        self.OPKNN = None
        self.meta_training_dataset = []
        self.meta_training_target = []
        self.n_meta_features = (self.k * 2) + self.Kp + 2

    def fit(self, X, y):
        """Prepare the DS model by setting the KNN algorithm and
        pre-processing the information required to apply the DS
        methods

        Parameters
        ----------
        X : array of shape = [n_samples, n_features] with the data.

        y : class labels of each sample in X.

        Returns
        -------
        self
        """
        self._set_dsel(X, y)
        self._fit_region_competence(X, y, self.k)
        self.dsel_scores = self._preprocess_dsel_scores()
        self._fit_OP(self.dsel_scores, y, self.kp)

        # check whether the meta-classifier was already trained
        # since it could have been pre-processed before
        try:
            check_is_fitted(self.meta_classifier, "classes_")
        except NotFittedError as _:
            if not self.meta_training_dataset:
                self._generate_meta_training_set()
            self._train_meta_classifier()

    def _fit_OP(self, X, y, kp):
        """Fit the set of output profiles.
        -------
        query_idx : int containing the index of the query sample in DSEL

        Returns
        -------
        agreement : float with the percentage of the base classifier that predicted the correct label
        for the query sample.

        """
        self.OPKNN = KNeighborsClassifier(n_neighbors=kp, n_jobs=-1, algorithm='auto')
        self.OPKNN.fit(X, y)

    def _sample_selection(self, query_idx):
        """Check the number of base classifier that predict the correct label for the query sample.
        Parameters
        ----------
        query_idx : int containing the index of the query sample in DSEL

        Returns
        -------
        agreement : float with the percentage of the base classifier that predicted the correct label
        for the query sample.

        """
        return np.sum(self.processed_dsel[query_idx, :]) / 100

    def compute_meta_features(self, query, idx_neighbors, idx_neighbors_op, clf, clf_index):
        """Compute the five sets of meta-features used in the META-des framework. Returns
        the meta-features vector V_i,j.

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        idx_neighbors : list containing the index of K-nearest neighbors of the query

        idx_neighbors_op : list containing the index of the most similar output profiles

        clf : The base classifier to estimate its level of competence

        clf_index : Index of the base classifier (clf) in the pool

        Returns
        -------
        v : array containing the five sets of meta-features estimated for the base classifier clf

        """
        # Meta-Features computation
        f1 = [self.processed_dsel[idx][clf_index] for idx in idx_neighbors]
        f2 = [self._get_scores_dsel(clf_index, idx)[self.DSEL_target[idx]] for idx in idx_neighbors]
        f3 = np.mean(f1)
        f4 = [self.processed_dsel[idx][clf_index] for idx in idx_neighbors_op]
        # check with the classifier model how to compute f5
        f5 = np.max(clf.predict_proba(query))
        v = f1 + f2 + [f3] + f4 + [f5]

        return v

    def _generate_meta_training_set(self):
        """Routine to generate the meta-training dataset that is further used to train the meta-classifier (Lambda)

        The first step is to apply the sample selection mechanism in order to decide whether  or not the corresponding
        sample should be used for meta-training process. Then, for each base classifier, five sets of meta-features
        are calculated and added to the meta-training dataset.
        """
        for idx_sample, sample in enumerate(self.DSEL_data):
            sample = sample.reshape(1, -1)
            # sample selection mechanism
            # TODO: review this sample selection part
            disagreement = self._sample_selection(idx_sample)
            if self.Hc > disagreement > (1 - self.Hc):
                # Extract meta-features
                _, idx_neighbors = self._get_region_competence(sample, self.k + 1)
                _, idx_neighbors_op = self._get_similar_out_profiles(sample, self.kp + 1)
                # Remove itself from the list of NN
                idx_neighbors = idx_neighbors[1:]
                idx_neighbors_op = idx_neighbors_op[1:]
                for clf_index, clf in enumerate(self.pool_classifiers):
                    vector = self.compute_meta_features(sample, idx_neighbors, idx_neighbors_op, clf, clf_index)
                    # get whether the classifier predicted the correct label or not
                    target = self.processed_dsel[idx_sample][clf_index]
                    self.meta_training_dataset.append(vector)
                    self.meta_training_target.append(target)

                    # train the meta classifier

    def _train_meta_classifier(self):
        """Train the meta-classifier (lambda), using the meta-training dataset.

        """
        self.meta_training_dataset = np.array(self.meta_training_dataset)
        self.meta_training_target = np.array(self.meta_training_target)
        if isinstance(self.selector, MultinomialNB):
            try:
                from mdlp.discretization import MDLP
                self.mdlp = MDLP()
                self.meta_training_dataset = self.mdlp.fit_transform(self.meta_training_dataset,
                                                                     self.meta_training_target)
            except ImportError:
                raise ValueError('In order to use you need to install the MDLP discretization package. '
                                 'Call pip install git+https://github.com/hlin117/mdlp-discretization to install'
                                 'the package.')

        self.meta_classifier.fit(self.meta_training_dataset, self.meta_training_target)

    def _get_similar_out_profiles(self, query, kp=None):
        """Get the most similar output profiles of the query sample.

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        kp : The number of output profiles (most similar) to be selected.

        Returns
        -------
        dists : A vector with the distances between the output profile of the query and
        those of the instances in the region of competence. The vector is ordered
        in an ascending fashion

        idx : Indices of the most similar output profiles
        """
        query_op = self._output_profile_transform(query).reshape(1, -1)
        if kp is None:
            kp = self.kp

        [dists], [idx] = self.OPKNN.kneighbors(query_op, n_neighbors=kp, return_distance=True)
        return dists, idx

    def select(self, competences):
        """Selects the base classifiers that obtained a competence level higher than the predefined
        threshold Gamma.

        Parameters
        ----------
        competences : array of shape = [n_classifiers] containing the competence level estimated
        for each base classifier

        Returns
        -------
        indices : the indices of the selected base classifiers
        """
        indices = [idx for idx, _ in enumerate(competences) if competences[idx] >= self.gamma]

        # if no classifier was selected, use the whole pool
        if len(indices) == 0:
            indices = range(self.n_classifiers)

        return indices

    def estimate_competence(self, query):
        """Estimate the competence of each base classifier ci
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
        _, idx_neighbors = self._get_region_competence(query)
        _, idx_neighbors_op = self._get_similar_out_profiles(query)
        vectors = []
        for clf_index, clf in enumerate(self.pool_classifiers):
            # Check if the dynamic frienemy pruning (DFP) should be used used
            if self.mask[clf_index]:
                vectors.append(self.compute_meta_features(query, idx_neighbors, idx_neighbors_op, clf, clf_index))
            else:
                # TODO: Check if that pruning scheme works by setting everything to zero.
                vectors.append(np.zeros(self.n_meta_features))

        vectors = np.array(vectors)
        if isinstance(self.selector, MultinomialNB):
            vectors = self.mdlp.transform(vectors)
        if self.version == "selection":
            competences = self.meta_classifier.predict(vectors)
        else:
            # Get the probability for class 1 (Competent)
            competences = self.meta_classifier.predict_proba(vectors)[:, 1]

        return competences
