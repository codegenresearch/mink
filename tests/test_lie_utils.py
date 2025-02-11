"""Tests for utility functions in mink/lie/utils.py."""

import numpy as np
from absl.testing import absltest

from mink.lie import utils


class TestUtils(absltest.TestCase):
    def test_skew_raises_error_for_invalid_shape(self):
        with self.assertRaises(AssertionError):
            utils.skew(np.zeros((5,)))

    def test_skew_matrix_transpose_is_negative(self):
        m = utils.skew(np.random.randn(3))
        np.testing.assert_allclose(m.T, -m)


if __name__ == "__main__":
    absltest.main()