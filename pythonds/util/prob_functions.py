# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

"""This file contains the implementation of several functions to estimate the competence level of a base classifiers
based on the vector of supports predicted for each class.

Reference
----------
Smith, M.R., Martinez, T. and Giraud-Carrier, C., 2014. An instance level analysis of data complexity.
Machine learning, 95(2), pp.225-256
"""
import numpy as np
import numpy.matlib as npm
from scipy.special import betainc


def exponential_func(n_classes, support_correct):
    """Calculate the exponential function based on the support obtained by
    the base classifier for the correct class label.
    Parameters
    ----------
    n_classes : int, Number of classes in the problem

    support_correct: np.array(dtype=Float), supports obtained by the base
    classifier for the correct class

    Returns
    -------
    C_src : Result of the exponential function calculated over all training samples
    """
    support_correct[support_correct > 1] = 1
    support_correct[support_correct < 0] = 0
    C_src = []
    for idx, support in enumerate(support_correct):
        if support == 1:
            temp = 1
        else:
            temp = -1 * ((n_classes - 1) * support)/(1 - support)

        C_src.append(1 - (2**temp))
    return np.array(C_src)


def log_func(n_classes, support_correct):
    """Calculate the logarithm in the support obtained by
    the base classifier.
    Parameters
    ----------
    n_classes : int, Number of classes in the problem

    support_correct: np.array(dtype=Float), supports obtained by the base
    classifier for the correct class

    Returns
    -------
    C_src : Result of the logarithmic function calculated over all training samples
    """
    support_correct[support_correct > 1] = 1
    support_correct[support_correct < 0] = 0
    C_src = 2 * ((support_correct ** (np.log(2) / np.log(n_classes))) - 1)

    return C_src


def entropy_func(n_classes, supports, is_correct):
    """Calculate the entropy in the support obtained by
    the base classifier.

    The value of the source competence is inverse proportional to
    the normalized entropy of its supports vector and the sign of competence is simply
    determined  by the correct/incorrect classification.
    Parameters
    ----------
    n_classes : int, Number of classes in the problem

    supports: np.array(dtype=Float), supports obtained by the base
    classifier for each class

    is_correct: np.array(dtype=int), array with 1 whether the base
    classifier predicted the correct label, -1 otherwise

    Returns
    -------
    C_src : Result of the entropy function calculated over all input samples
    """
    supports[supports > 1] = 1
    supports[supports < 0] = 0
    is_correct[is_correct == 0] = -1

    ent = 1 - ((1/np.log(n_classes)) * np.sum(np.multiply(supports, np.log(supports)), axis=1))
    C_src = np.multiply(ent, is_correct)

    return C_src


def ccprmod(supports, idx_correct_class, B=20):
    """
    Python implementation of the ccprmod.m (Classifier competence based on probabilistic modelling)
    function. Matlab code is available at:
http://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/28391/versions/6/previews/ccprmod.m/index.html

    Input:
    supports - NxC matrix of normalised C class supports produced by the classifier for N objects
    idx_correct_class - Nx1 vector of indices of the correct classes for N objects
    B - number of points used in the calculation of the competence, higher values result
         in a more accurate estimation (optional, default B=20)

    Output:
    cc - Nx1 vector of the classifier competences

    Example:
    supports = [[0.3, 0.6, 0.1],[1.0/3, 1.0/3, 1.0/3]]
    idx_correct_class = [1,0]

    ccprmod(supports,idx_correct_class)
     ans = 0.784953394056843
           0.332872292262951
    """
    supports = npm.mat(supports)
    idx_correct_class = np.array(idx_correct_class)
    supports[supports > 1] = 1
    # Close to zero support for any given class leads to NaN. So here we ae giving a small number instead to see if it
    # works
    supports[np.isclose(supports, 0)] = 0.1

    N, C = supports.shape

    x = np.matlib.linspace(0, 1, B)
    x = np.matlib.repmat(x, N, C)

    a = npm.zeros(x.shape)

    for c in range(C):
        a[:, c*B:(c+1)*B] = np.matlib.repmat(C*supports[:, c], 1, B)

    b = C - a
    betaincj = betainc(a, b, x)

    C_src = np.zeros(N)
    for n in range(N):
        t = range((idx_correct_class[n])*B, (idx_correct_class[n]+1)*B)
        bc = betaincj[n, t]
        bi = betaincj[n, list(set(range(0, (C*B))) - set(t))]
        bi = npm.transpose(npm.reshape(bi, (B, C-1), order='F'))
        C_src[n] = np.sum(np.multiply((bc[0, 1:] - bc[0, 0:-1]), np.prod((bi[:, 0:-1] + bi[:, 1:])/2, 0)))

    return C_src


def softmax(w, theta=1.0):
    """"
    takes an vector w of S N-element and returns a vectors where each column of the
    vector sums to 1, with elements exponentially proportional to the
    respective elements in N.
    Parameters
    ----------
    w : ndarray with N elements

    theta : float parameter, used as a multiplier  prior to exponentiation. Default = 1.

    Returns
    ----------
    dist: np.array which the sum of each column sums to 1 and the elements are exponentially propoertional to the
    respective elements in N

    """
    e = np.exp(np.array(w) / theta)
    dist = e / np.sum(e)
    return dist


def min_difference():
    competences = []
    return competences
