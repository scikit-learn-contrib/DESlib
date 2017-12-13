# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

def _process_predictions(y, y_pred1, y_pred2):

    N00, N01, N10, N11 = 0.0, 0.0, 0.0, 0.0

    for index in range(len(y)):
        if y_pred1[index] == y[index] and y_pred2[index] == y[index]:
            N11 += 1.0
        elif y_pred1[index] == y[index] and y_pred2[index] != y[index]:
            N10 += 1.0
        elif y_pred1[index] != y[index] and y_pred2[index] == y[index]:
            N01 += 1.0
        else:
            N00 += 1.0

    return N00, N10, N01, N11


def double_fault(y, y_pred1, y_pred2):
    N00, N10, N01, N11 = _process_predictions(y, y_pred1, y_pred2)
    return N00/(N00 + N10 + N01 + N11)


def Q_statistic(y, y_pred1, y_pred2):
    N00, N10, N01, N11 = _process_predictions(y, y_pred1, y_pred2)
    return ((N11*N00) - (N01*N10)) / ((N11 * N00) + (N01 * N10))


def ratio_errors(y, y_pred1, y_pred2):
    N00, N10, N01, N11 = _process_predictions(y, y_pred1, y_pred2)
    return (N01 + N10)/N00

