"""Tests utils.py."""

import numpy as np
from absl.testing import absltest

from mink.lie import utils


class TestUtils(absltest.TestCase):
    def test_skew_throws_assertion_error_if_shape_invalid(self):
        with self.assertRaises(AssertionError):
            utils.skew(np.zeros((5,)))

    def test_skew_throws_assertion_error_if_input_not_1d(self):
        with self.assertRaises(AssertionError):
            utils.skew(np.zeros((3, 3)))

    def test_skew_throws_assertion_error_if_input_not_numeric(self):
        with self.assertRaises(AssertionError):
            utils.skew(np.array(['a', 'b', 'c']))

    def test_skew_equals_negative(self):
        m = utils.skew(np.random.randn(3))
        np.testing.assert_allclose(m.T, -m)

    def test_skew_returns_correct_skew_symmetric_matrix(self):
        v = np.array([1, 2, 3])
        expected = np.array([[0, -3, 2],
                             [3, 0, -1],
                             [-2, 1, 0]])
        np.testing.assert_allclose(utils.skew(v), expected)

    def test_skew_zero_vector_returns_zero_matrix(self):
        v = np.zeros(3)
        expected = np.zeros((3, 3))
        np.testing.assert_allclose(utils.skew(v), expected)


if __name__ == "__main__":
    absltest.main()


I have removed the stray comment that caused the `SyntaxError`. The test cases are now properly formatted and should run without syntax issues. I have also ensured that the test cases cover the essential functionality of the `skew` function while maintaining simplicity and reducing redundancy.