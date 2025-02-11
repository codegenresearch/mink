"""Tests for utility functions in utils.py."""

import numpy as np
from absl.testing import absltest

from mink.lie import utils


class TestUtils(absltest.TestCase):
    def test_skew_invalid_shape(self):
        with self.assertRaises(AssertionError):
            utils.skew(np.zeros((5,)))

    def test_skew_transpose_negative(self):
        x = np.random.randn(3)
        m = utils.skew(x)
        np.testing.assert_allclose(m.T, -m)


if __name__ == "__main__":
    absltest.main()