import numpy as np
import pytest

from deslib.util.instance_hardness import hardness_region_competence


# ------Test routines for the Instance Hardness calculation------------------
@pytest.mark.parametrize('index, expected', [(0, 0.42),
                                             (1, 0.28),
                                             (2, 0.28)])
def test_instance_hardness_region(index,
                                  expected,
                                  example_estimate_competence):
    y, neighbors = example_estimate_competence[1:3]
    k = 7
    neighbors = neighbors[index, :]
    IH = hardness_region_competence(neighbors, y, k)
    assert np.isclose(IH, expected, atol=0.01)


def test_instance_hardness_region_batch(example_estimate_competence):
    expected = np.array([0.42, 0.28, 0.28])
    y, neighbors = example_estimate_competence[1:3]

    k = 7
    IH = hardness_region_competence(neighbors, y, k)
    assert np.allclose(IH, expected, atol=0.01)


def test_instance_hardness_region_all_same(example_estimate_competence):
    y = example_estimate_competence[1]
    k = 7
    neighbors = np.array([0, 1, 2, 6, 7, 8, 13])
    IH = hardness_region_competence(neighbors, y, k)
    assert IH == 0.0
