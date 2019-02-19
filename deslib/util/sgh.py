# coding=utf-8

# Authors: Mariana A. Souza <mas2@cin.ufpe.br>
#          Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

from sklearn.ensemble import BaseEnsemble
from sklearn.linear_model import SGDClassifier
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.utils.validation import check_X_y
from math import sqrt


def _build_Perceptron(X, y, curr_training_samples, centroids):
    """
    Calculates the parameters (weight and bias) of the hyperplane placed in the midpoint between the centroids of most
    distant classes in X[curr_training_samples].


    Parameters
    ----------
    X : array of shape = [n_samples, n_features]
        The training data.

    y : array of shape = [n_samples]
        class labels of each example in X.

    curr_training_samples : array of shape = [n_samples]
        array of ones and zeros ('1','0'), indicating which samples in X are to be used for placing the hyperplane.

    centroids : array of shape = [n_classes,n_features]
        centroids of each class considering the previous distribution of X[curr_training_samples].

    Returns
    -------

    centroids : array of shape = [n_classes,n_features]
        centroids of each class considering the current distribution of X[curr_training_samples].

    weights : array of shape = [n_classes,n_features]
        weights of the hyperplane.

    bias : array of shape = [n_classes,1]
        bias of the hyperplane.

    """
    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = classes.size
    weights = np.zeros((n_classes, n_features), dtype=float)
    bias = np.zeros(n_classes, dtype=float)

    # Identify remaining training samples
    idx_curr_training_samples = np.where(curr_training_samples > 0)

    # Set of remaining samples
    eval_X = X[idx_curr_training_samples[0]]
    eval_y = y[idx_curr_training_samples[0]]

    # Vector indicating the remaining classes in eval_X/eval_y
    curr_classes = np.zeros(n_classes, dtype=int)

    for i in range(0, n_classes):
        # Select instances from a single class
        c = eval_X[np.where(classes[i] == eval_y)]
        if c.size:
            # Update centroid of class
            centroids[i, ] = c.mean(axis=0)
            # Indicate its presence
            curr_classes[i] = 1

    idx_curr_classes = np.where(curr_classes > 0)

    if curr_classes.sum() >= 2:      # More than 2 classes remain in eval_X
        # Pairwise distance between current classes
        dist_classes = squareform(pdist(centroids[idx_curr_classes[0]]))
        np.fill_diagonal(dist_classes, np.inf)

        # Identify the two farthest away
        closest_dist = np.unravel_index(np.argmin(dist_classes), dist_classes.shape)

        idx_class_1 = idx_curr_classes[0][closest_dist[0]]
        idx_class_2 = idx_curr_classes[0][closest_dist[1]]

    else:       # Only one class remains
        # Pairwise distance between all classes in the problem
        dist_classes = squareform(pdist(centroids))
        np.fill_diagonal(dist_classes, np.inf)

        # Remaining class
        idx_class_1 = idx_curr_classes[0][0]
        # Most distant from class_1
        idx_class_2 = np.argmin(dist_classes[idx_class_1, ])

    # Difference vector between selected classes
    diff_vec = centroids[idx_class_1, ] - centroids[idx_class_2, ]

    if not np.any(diff_vec):
        w_p = 0.01 * np.ones(n_features, dtype=float)
        w_p = w_p / sqrt((w_p ** 2).sum())
        raise Warning('Equal classes centroids!')

    else:
        # Normal vector of diff_vec
        w_p = diff_vec / sqrt((diff_vec ** 2).sum())

    theta_p = np.dot(-w_p, (centroids[idx_class_1, ] + centroids[idx_class_2, ]) / 2)

    # Weights of linear classifier
    weights[idx_class_1, ] = w_p
    weights[idx_class_2, ] = -w_p

    # Bias of linear classifier
    bias[idx_class_1, ] = theta_p
    bias[idx_class_2, ] = -theta_p

    # Return updated centroids, weights and bias of the linear classifier
    return centroids, weights, bias


class SGH(BaseEnsemble):
    """
    Self-Generating Hyperplanes (SGH).

    Generates a pool of classifiers which guarantees an Oracle accuracy rate of 100% over the training (input) set.
    That is, for each instance in the training set, there is at least one classifier in the pool able to correctly
    label it. The generated classifiers are always two-class hyperplanes.

    References
    ----------
    L. I. Kuncheva, A theoretical study on six classifier fusion strategies, IEEE Transactions on
    Pattern Analysis and Machine Intelligence 24 (2) (2002) 281-286.

    M. A. Souza, G. D. Cavalcanti, R. M. Cruz, R. Sabourin, On the characterization of the
    oracle for dynamic classifier selection, in: International Joint Conference on Neural Networks,
    IEEE, 2017, pp. 332-339.

    """
    def __init__(self):

        super(SGH, self).__init__(base_estimator=SGDClassifier,
                                  n_estimators=1)

        # Pool initially empty
        self.estimators_ = []

    def fit(self, X, y, included_samples=np.array([])):
        """
        Populates the SHG ensemble.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The training data.

        y : array of shape = [n_samples]
            class labels of each example in X.

        included_samples : array of shape = [n_samples]
            array of ones and zeros ('1','0'), indicating which samples in X are to be used for training.
            If all, leave blank.


        Returns
        -------
        self
        """
        check_X_y(X, y)
        return self._fit(X, y, included_samples)

    def _fit(self, X, y, included_samples):

        # Set base estimator as the Perceptron
        self.base_estimator_ = SGDClassifier(loss="perceptron",
                                             eta0=1.e-17,
                                             max_iter=1,
                                             learning_rate="constant",
                                             penalty=None)

        # If there is no indication of which instances to include in the training, include all
        if included_samples.sum() == 0:
            included_samples = np.ones((X.shape[0]), int)

        # Generate pool
        self._generate_pool(X, y, included_samples)

        return self

    def _generate_pool(self, X, y, curr_training_samples):
        """
        Generates the classifiers in the pool of classifiers ("estimators_") using the SGH method.

        In each iteration of the method, a hyperplane is placed in the midpoint between the centroids
        of the two most distant classes in the training data. Then, the newly generated classifier is
        tested over all samples and the ones it correctly labels are removed from the set. In the following
        iteration, a new hyperplane is created based on the classes of the remaining samples in the training set.
        The method stops when no sample remains in the training set.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The training data.

        y : array of shape = [n_samples]
            class labels of each example in X.

        curr_training_samples : array of shape = [n_samples]
            array of ones and zeros ('1','0'), indicating which samples in X are to be used for training.
            If all, leave blank.

        Returns
        -------
        self
        """
        n_samples, n_features = X.shape

        # Labels of the correct classifier for each training sample
        corr_classif_lab = np.zeros(n_samples, dtype=int)

        # Pool size
        n_perceptrons = 0
        n_err = 0
        max_err = 50

        # Problem's classes
        classes = np.unique(y)
        n_classes = classes.size

        # Centroids of each class
        centroids = np.zeros((n_classes, n_features), float)

        # While there are still misclassified samples
        while curr_training_samples.sum() > 0 and n_err < max_err:

            # Update centroids and obtain the classifier's coefficients
            centroids, weights, bias = _build_Perceptron(X, y, curr_training_samples, centroids)

            # Generate classifier
            curr_perc = self._make_estimator(append=False, random_state=None)

            curr_perc.classes_ = classes
            curr_perc.fit(X, y)

            curr_perc.coef_ = weights
            curr_perc.intercept_ = bias

            # Add classifier to pool
            self.estimators_.append(curr_perc)

            # Obtain set with instances that weren't correctly classified yet
            idx_curr_training_samples = np.where(curr_training_samples > 0)
            eval_X = X[idx_curr_training_samples[0]]
            eval_y = y[idx_curr_training_samples[0]]

            # Evaluate generated classifier over eval_X
            out_curr_perc = self.estimators_[n_perceptrons].predict(eval_X)

            # Identify correctly classified samples
            idx_correct_eval = (out_curr_perc == eval_y).nonzero()

            # Exclude correctly classified samples from current training set
            curr_training_samples[idx_curr_training_samples[0][idx_correct_eval[0]]] = 0

            # Set classifier label for the correctly classified instances
            corr_classif_lab[idx_curr_training_samples[0][idx_correct_eval[0]]] = n_perceptrons
            # Increase pool size
            n_perceptrons += 1
            n_err += 1

        # Update pool size
        self.n_estimators_ = n_perceptrons
        # Update classifier labels
        self.correct_classif_label_ = corr_classif_lab

        return self

    # TODO: implement hit rate function
