# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause
import numpy as np
import numpy.matlib as npm
from scipy.special import betainc
from scipy.stats import entropy

"""This file contains the implementation of several functions used to estimate
the competence level of a base classifiers based on posterior probabilities
predicted for each class.

Reference
----------
T.Woloszynski, M. Kurzynski, A probabilistic model of classifier competence for
dynamic ensemble selection,
Pattern Recognition 44 (2011) 2656–2668.

R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
selection: Recent advances and perspectives,”
Information Fusion, vol. 41, pp. 195 – 216, 2018.

B. Antosik, M. Kurzynski, New measures of classifier competence – heuristics
and application to the design of multiple classifier systems., in: Computer
recognition systems 4., 2011, pp. 197–206.
"""


def exponential_func(n_classes, support_correct):
    """Calculate the exponential function based on the support obtained by
    the base classifier for the correct class label.

    Parameters
    ----------
    n_classes : int
        The number of classes in the problem

    support_correct: array of shape = [n_samples]
        containing the supports obtained by the base classifier for the correct
        class

    Returns
    -------
    C_src : array of shape = [n_samples]
        Representing the classifier competences at each data point
    """
    support_correct[support_correct <= 0.0] = 0.0
    C_src = np.zeros(support_correct.size)

    # Special case where the support to the correct class is equal to one.
    support_one_indices = support_correct >= 1
    C_src[np.where(support_one_indices)[0]] = 1

    # Apply the competence formula when the support is less than one
    indices_not_one = np.where(~support_one_indices)[0]

    temp = (1.0 - ((n_classes - 1.0) * support_correct[indices_not_one]) / (
                1.0 - support_correct[indices_not_one]))
    C_src[indices_not_one] = (1.0 - (2 ** temp))

    return C_src


def log_func(n_classes, support_correct):
    """Calculate the logarithm in the support obtained by
    the base classifier.

    Parameters
    ----------
    n_classes : int
        The number of classes in the problem

    support_correct: array of shape = [n_samples]
        Containing the supports obtained by the base classifier for the correct
        class

    Returns
    -------
    C_src : array of shape = [n_samples]
            representing the classifier competences at each data point

    References
    ----------
    T.Woloszynski, M. Kurzynski, A measure of competence based on randomized
    reference classifier for dynamic ensemble selection, in: International
    Conference on Pattern Recognition (ICPR), 2010, pp. 4194–4197.
    """

    support_correct[support_correct > 1] = 1
    support_correct[support_correct < 0] = 0

    if n_classes == 2:
        C_src = (2 * support_correct) - 1
    else:
        temp = np.log(2) / np.log(n_classes)
        C_src = (2 * (support_correct ** temp)) - 1

    return C_src


def entropy_func(n_classes, supports, is_correct):
    """Calculate the entropy in the support obtained by
    the base classifier. The value of the source competence is inverse
    proportional to the normalized entropy of its supports vector and the sign
    of competence is simply determined  by the correct/incorrect classification

    Parameters
    ----------
    n_classes : int
        The number of classes in the problem

    supports: array of shape = [n_samples, n_classes]
        Containing the supports obtained by the base classifier for each class.

    is_correct: array of shape = [n_samples]
        Array with 1 whether the base classifier predicted the correct label
        and -1 otherwise

    Returns
    -------
    C_src : array of shape = [n_samples]
        Representing the classifier competences at each data point

    References
    ----------
    B. Antosik, M. Kurzynski, New measures of classifier competence –
    heuristics and application to the design of multiple classifier systems.,
    in: Computer recognition systems 4., 2011, pp. 197–206.
    """
    n_samples = is_correct.shape[0]
    if n_samples != supports.shape[0]:
        raise ValueError("The number of samples in X and y must be the same"
                         "n_samples X = {}, n_samples y = {} ".format(
                           n_samples, supports.shape[0]))

    supports[supports > 1.0] = 1.0
    supports[supports < 0.0] = 0.0

    C_src = np.zeros(n_samples)
    for index in range(n_samples):
        C_src[index] = (1.0 / np.log(n_classes)) * (
            entropy(supports[index, :]))
        C_src[index] += ((2 * is_correct[index]) - 1)
    return C_src


def ccprmod(supports, idx_correct_label, B=20):
    """Python implementation of the ccprmod.m (Classifier competence based on
    probabilistic modelling)
    function. Matlab code is available at:
    http://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/28391/versions/6/previews/ccprmod.m/index.html

    Parameters
    ----------
    supports: array of shape = [n_samples, n_classes]
        Containing the supports obtained by the base classifier for each class.

    idx_correct_label: array of shape = [n_samples]
                       containing the index of the correct class.

    B : int (Default = 20)
        number of points used in the calculation of the competence, higher
        values result in a more accurate estimation.

    Returns
    -------
    C_src : array of shape = [n_samples]
            representing the classifier competences at each data point

    Examples
    --------
    >>> supports = [[0.3, 0.6, 0.1],[1.0/3, 1.0/3, 1.0/3]]
    >>> idx_correct_label = [1,0]
    >>> ccprmod(supports,idx_correct_label)
    ans = [0.784953394056843, 0.332872292262951]

    References
    ----------
    T.Woloszynski, M. Kurzynski, A probabilistic model of classifier competence
    for dynamic ensemble selection,
    Pattern Recognition 44 (2011) 2656–2668.
    """
    if not isinstance(B, int):
        raise TypeError(
            'Parameter B should be an integer. '
            'Currently B is {0}'.format(type(B)))

    if B <= 0 or B is None:
        raise ValueError(
            'The parameter B should be higher than 0. '
            'Currently B is {0}'.format(B))

    supports = np.asarray(supports)
    idx_correct_label = np.array(idx_correct_label)
    supports[supports > 1] = 1

    N, C = supports.shape

    x = np.linspace(0, 1, B)
    x = np.matlib.repmat(x, N, C)

    a = npm.zeros(x.shape)

    for c in range(C):
        a[:, c * B:(c + 1) * B] = C * supports[:, c:c + 1]

    b = C - a

    # For extreme cases, with a or b equal to 0, add a small constant:
    eps = 1e-20
    a[a == 0] = eps
    b[b == 0] = eps
    betaincj = betainc(a, b, x)

    C_src = np.zeros(N)
    for n in range(N):
        t = range((idx_correct_label[n]) * B, (idx_correct_label[n] + 1) * B)
        bc = betaincj[n, t]
        bi = betaincj[n, list(set(range(0, (C * B))) - set(t))]
        bi = npm.transpose(npm.reshape(bi, (B, C - 1), order='F'))
        C_src[n] = np.sum(np.multiply((bc[0, 1:] - bc[0, 0:-1]),
                                      np.prod((bi[:, 0:-1] + bi[:, 1:]) / 2,
                                              0)))

    return C_src


def min_difference(supports, idx_correct_label):
    """The minimum difference between the supports obtained for the correct
    class and the vector of class supports. The value of the source competence
    is negative if the sample is misclassified and positive otherwise.

    Parameters
    ----------
    supports: array of shape = [n_samples, n_classes]
        Containing the supports obtained by the base classifier for each class

    idx_correct_label: array of shape = [n_samples]
        Containing the index of the correct class

    Returns
    -------
    C_src : array of shape = [n_samples]
        Representing the classifier competences at each data point

    References
    ----------
    B. Antosik, M. Kurzynski, New measures of classifier competence –
    heuristics and application to the design of multiple classifier systems.,
    in: Computer recognition systems 4., 2011, pp. 197–206.
    """
    n_samples = len(idx_correct_label)
    # Boolean mask for the correct class
    mask = np.zeros(supports.shape, dtype=np.bool)
    mask[np.arange(n_samples), idx_correct_label] = True
    # Get supports for the correct class
    supports_correct = supports[mask]
    # Get supports for the other classes
    supports_others = supports[~mask]

    if len(supports_others) == 0:
        # Corner case where there is a single class in y_train
        supports_others = np.zeros_like(supports_correct)

    difference = supports_correct.reshape(-1, 1) - supports_others.reshape(
        supports_correct.size, -1)
    C_src = np.sort(difference, axis=1)[:, 0]
    return C_src


def softmax(w, theta=1.0):
    """Takes an vector w of S N-element and returns a vectors where each column
    of the vector sums to 1, with elements exponentially proportional to the
    respective elements in N.

    Parameters
    ----------
    w : array of shape = [N,  M]

    theta : float (default = 1.0)
            used as a multiplier  prior to exponentiation.

    Returns
    -------
    dist : array of shape = [N, M]
        Which the sum of each row sums to 1 and the elements are exponentially
        proportional to the respective elements in N

    """
    w = np.atleast_2d(w)
    e = np.exp(np.array(w) / theta)
    dist = e / np.sum(e, axis=1).reshape(-1, 1)
    return dist
