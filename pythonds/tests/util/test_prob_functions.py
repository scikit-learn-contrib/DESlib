# from pythonds.util.prob_functions import ccprmod, log_func, exponential_func, entropy_func, min_difference
# import numpy as np
# import math
# import pytest
#
# data_value_ccprmod = [([0.3, 0.6, 0.1], 1), ([1.0 / 3, 1.0 / 3, 1.0 / 3], 0)]
#
#
# @pytest.mark.parametrize("supports, idx_correct_class", data_value_ccprmod)
# def test_ccprmod_value(supports, idx_correct_class):
#         value = ccprmod(supports. idx_correct_class)
#         assert np.isclose(value, [784953394056843, 0.332872292262951])
#
#
# @pytest.mark.parametrize('B', [0, -1, math.nan, None])
# def test_valid_ccprmod_beta(supports, idx_correct_class, B):
#     with pytest.raises(ValueError):
#         ccprmod(supports, idx_correct_class, B)
#
#
# @pytest.mark.parametrize('supports', [[0.0, 0.0, 1.0], [0.1, 0.1, 0.8],
#                                       [np.nextafter(1.0, 0.0), np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0)]])
# def test_ccprmod_zero_support(support):
#     idx_correct_class = [0, 2, 1]
#     assert not np.isnan(ccprmod(support,idx_correct_class))

