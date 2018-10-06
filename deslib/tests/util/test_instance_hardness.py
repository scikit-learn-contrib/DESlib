import numpy as np
import pytest

from deslib.tests.examples_test import setup_example1
from deslib.util.instance_hardness import hardness_region_competence


# -----------------------Test routines for the Instance Hardness calculation--------------------
@pytest.mark.parametrize('index, expected', [(0, 0.42),
                                             (1, 0.28),
                                             (2, 0.28)])
def test_instance_hardness_region(index, expected):
    _, y, neighbors, _, _, _ = setup_example1()
    k = 7
    neighbors = neighbors[index, :]
    IH = hardness_region_competence(neighbors, y, k)
    assert np.isclose(IH, expected, atol=0.01)


def test_instance_hardness_region_batch():
    expected = np.array([0.42, 0.28, 0.28])
    _, y, neighbors, _, _, _ = setup_example1()

    k = 7
    IH = hardness_region_competence(neighbors, y, k)
    assert np.allclose(IH, expected, atol=0.01)


def test_instance_hardness_region_all_same():
    y = setup_example1()[1]
    k = 7
    neighbors = np.array([0, 1, 2, 6, 7, 8, 13])
    IH = hardness_region_competence(neighbors, y, k)
    assert IH == 0.0
