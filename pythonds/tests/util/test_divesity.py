from pythonds.util.diversity import double_fault
import numpy as np

def test_double_fault():
    labels = np.array([0, 0, 0, 0, 1, 1, 1])
    pred1 = np.array([1, 0, 1, 0, 0, 0, 0])
    pred2 = np.array([1, 0, 0, 0, 1, 0, 0])

    actual = double_fault(labels,
                          pred1,
                          pred2)

    assert actual == 3./7  # three common errors out of 7 predictions