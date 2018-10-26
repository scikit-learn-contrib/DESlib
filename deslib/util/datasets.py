# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.utils.validation import check_random_state

"""
This file contains routines to generate 2D classification datasets
that can be used to test the performance of different machine learning
algorithms.

Datasets:

- P2 Dataset
- Circle and Square
- Banana
- Banana 2
- XOR

"""


def make_P2(size_classes, random_state=None):
    """Generate the P2 Dataset:

    The P2 is a two-class problem, presented by Valentini[1], in which each
    class is defined in multiple decision regions delimited by polynomial
    and trigonometric functions (E1, E2, E3 and E4):

    .. math:: \\begin{eqnarray}
        \\label{eq:problem1}
        E1(x) = sin(x) + 5 \\\\
        \\label{eq:problem2}
        E2(x) = (x - 2)^{2} + 1 \\\\
        \\label{eq:problem3}
        E3(x) = -0.1 \\cdot x^{2} + 0.6sin(4x) + 8 \\\\
        \\label{eq:problem4}
        E4(x) = \\frac{(x - 10)^{2}}{2} + 7.902
        \\end{eqnarray}

    Parameters
    ----------
    size_classes : list with the number of samples for each class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    returns
    -------
    X : array of shape = [size_classes, 2]
        The generated data points.

    y : array of shape = [size_classes]
        Class labels associated with each class.

    References
    ----------
    G. Valentini, An experimental bias-variance analysis of svm ensembles
    based on resampling techniques, IEEE Transactions on Systems, Man,
    and Cybernetics, Part B 35 (2005) 1252–1271.

    """
    rng = check_random_state(random_state)
    n_samples = sum(size_classes) * 6
    class_1 = np.zeros((size_classes[0], 2))
    class_2 = np.zeros((size_classes[1], 2))
    size_class1 = 0
    size_class2 = 0
    data = rng.rand(n_samples, 2)
    for x in data:
        if (size_class1 + size_class2) >= sum(size_classes) * 2:
            break

        # This if is very complex. Need to simplify that somehow
        if x[1] > (-0.1 * (x[0] * 10) ** 2 + 0.6 * np.sin(
                4 * x[0] * 10) + 8.) / 10. and x[1] > (
                (x[0] * 10 - 2) ** 2 + 1) / 10 or \
                x[1] < (2 * np.sin(x[0] * 10) + 5) / 10 and x[1] > (
                (x[0] * 10 - 2) ** 2 + 1) / 10 or \
                x[1] < (-0.1 * (x[0] * 10) ** 2 + 0.6 * np.sin(
                    4 * x[0] * 10) + 8) / 10 and x[1] < (
                (x[0] * 10 - 2) ** 2 + 1) / 10 and \
                x[1] > (2 * np.sin(x[0] * 10) + 5) / 10 or \
                x[1] > (-0.1 * (x[0] * 10) ** 2 + 0.6 * np.sin(
                    4 * x[0] * 10) + 8) / 10 and x[1] < (
                2 * np.sin(x[0] * 10) + 5) / 10 or \
                x[1] > (((x[0] * 10 - 10) ** 2) / 2 + 7.902) / 10.:

            if size_class1 < size_classes[0]:
                class_1[size_class1] = x
                size_class1 += 1
        elif size_class2 < size_classes[1]:
            class_2[size_class2] = x
            size_class2 += 1

    y = np.hstack((np.zeros(size_class1), np.ones(size_class2)))
    X = np.vstack((class_1, class_2))

    return X, y


def make_circle_square(size_classes, random_state=None):
    """Generate the circle square dataset.

    Parameters
    ----------
    size_classes : list with the number of samples for each class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    returns
    -------
    X : array of shape = [size_classes, 2]
        The generated data points.

    y : array of shape = [size_classes]
        Class labels associated with each class.

    References
    ----------
    P. Henniges, E. Granger, R. Sabourin, Factors of overtraining
    with fuzzy artmap neural networks, International Joint Conference
    on Neural Networks (2005) 1075–1080.

    """
    rng = check_random_state(random_state)
    n_samples = sum(size_classes) * 2
    class_1 = np.zeros((size_classes[0], 2))
    class_2 = np.zeros((size_classes[1], 2))
    size_class1 = 0
    size_class2 = 0
    data = rng.rand(n_samples * 10, 2)
    r = 0.398942
    for x in data:
        test_class = ((x[0] - 0.5) ** 2) + ((x[1] - 0.5) ** 2)
        if test_class < (r ** 2):
            if size_class1 < size_classes[0]:
                class_1[size_class1] = x
                size_class1 += 1

        elif size_class2 < size_classes[1]:
            class_2[size_class2] = x
            size_class2 += 1

        if size_class2 + size_class1 > n_samples:
            break

    y = np.hstack((np.zeros(size_class1), np.ones(size_class2)))
    X = np.vstack((class_1, class_2))

    return X, y


def make_banana(size_classes, na=0.1, random_state=None):
    """Generate the Banana dataset.

    Parameters
    ----------
    size_classes : list with the number of samples for each class.

    na : float (Default = 0.2),
        Noise amplitude. It must be < 1.0

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape = [size_classes, 2]
        The generated data points.

    y : array of shape = [size_classes]
        Class labels associated with each class.

    References
    ----------
    Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms.
    John Wiley & Sons, 2004.

    """
    rng = check_random_state(random_state)
    if not isinstance(na, float) or na > 1.:
        raise ValueError(
            'Parameter na must be a float lower than 1. na = {}'.format(na))

    t1 = np.transpose(-np.linspace(-np.pi / 4, np.pi, size_classes[0]))
    z1 = np.transpose((np.sin(t1), np.cos(t1)))
    class_1 = 1.5 * z1 + rng.randn(size_classes[0], 2) * na

    t2 = np.transpose(-np.linspace(-np.pi / 4, np.pi, size_classes[1]))
    z2 = np.transpose((np.sin(t2), np.cos(t2)))
    class_2 = z2 - rng.randn(size_classes[1], 2) * na
    y = np.hstack((np.zeros(size_classes[0]), np.ones(size_classes[1])))
    X = np.vstack((class_1, class_2))

    return X, y


def make_banana2(size_classes, sigma=1, random_state=None):
    """Generate the Banana dataset similar to the Matlab PRTools toolbox.

    Parameters
    ----------
    size_classes : list with the number of samples for each class.

    sigma : float (Default = 1),
        variance of the normal distribution

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape = [size_classes, 2]
        The generated data points.

    y : array of shape = [size_classes]
        Class labels associated with each class.

    References
    ----------
    R.P.W. Duin, P. Juszczak, D.de Ridder, P. Paclik, E. Pekalska, D.M.Tax,
    Prtools, a matlab toolbox for
    pattern recognition, 2004. URL 〈http://www.prtools.org〉.

    """
    rng = check_random_state(random_state)
    banana_size = 5
    region_class_1 = 0.125 * np.pi + rng.rand(
        size_classes[0]) * 1.25 * np.pi
    data_class_1 = banana_size * np.transpose(
        [np.sin(region_class_1), np.cos(region_class_1)]) + \
        rng.randn(size_classes[0], 2) * sigma

    region_class_2 = 0.375 * np.pi - rng.rand(
        size_classes[1]) * 1.25 * np.pi

    tmp = np.transpose(
        [np.sin(region_class_2), np.cos(region_class_2)]) * banana_size

    data_class_2 = (tmp + rng.randn(size_classes[1], 2) * sigma) + (
        np.ones((size_classes[1], 2)) * (-0.75 * banana_size))

    X = np.vstack((data_class_1, data_class_2))
    y = np.hstack((np.zeros(size_classes[0]), np.ones(size_classes[1])))

    return X, y


def make_xor(n_samples, random_state=None):
    """Generate the exclusive-or (XOR) dataset.

    Parameters
    ----------
    n_samples : int
                Number of generated data points.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape = [size_classes, 2]
        The generated data points.

    y : array of shape = [size_classes]
        Class labels associated with each class.

    """
    rng = check_random_state(random_state)
    X = rng.uniform(low=0, high=1, size=(n_samples, 2))
    y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5)

    return X, y
