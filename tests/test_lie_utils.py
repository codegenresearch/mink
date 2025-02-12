"""Tests for utility functions in utils.py."""

import numpy as np
from absl.testing import absltest

from mink.lie import utils


class TestUtilityFunctions(absltest.TestCase):
    def test_skew_raises_assertion_error_for_invalid_shape(self):
        with self.assertRaises(AssertionError):
            utils.skew(np.zeros((5,)))

    def test_skew_matrix_transpose_equals_negative(self):
        random_vector = np.random.randn(3)
        skew_matrix = utils.skew(random_vector)
        np.testing.assert_allclose(skew_matrix.T, -skew_matrix)


if __name__ == "__main__":
    absltest.main()