"""Tests utils.py."""

import numpy as np
from absl.testing import absltest

from mink.lie import utils


class TestUtils(absltest.TestCase):
    def test_skew_throws_assertion_error_if_shape_invalid(self):
        with self.assertRaises(AssertionError):
            utils.skew(np.zeros((5,)))

    def test_skew_equals_negative(self):
        random_vector: np.ndarray = np.random.randn(3)
        skew_matrix: np.ndarray = utils.skew(random_vector)
        np.testing.assert_allclose(skew_matrix.T, -skew_matrix)


if __name__ == "__main__":
    absltest.main()