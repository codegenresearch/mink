"""Tests utils.py."""

import numpy as np
from absl.testing import absltest

from mink.lie import utils


class TestUtils(absltest.TestCase):
    def test_skew_throws_assertion_error_if_shape_invalid(self):
        with self.assertRaises(AssertionError):
            utils.skew(np.zeros((5,)))

    def test_skew_equals_negative(self):
        m = utils.skew(np.random.randn(3))
        np.testing.assert_allclose(m.T, -m)

    def test_skew_returns_correct_skew_symmetric_matrix(self):
        v = np.array([1, 2, 3])
        expected = np.array([[0, -3, 2],
                             [3, 0, -1],
                             [-2, 1, 0]])
        np.testing.assert_allclose(utils.skew(v), expected)

    def test_skew_with_zero_vector(self):
        v = np.array([0, 0, 0])
        expected = np.zeros((3, 3))
        np.testing.assert_allclose(utils.skew(v), expected)

    def test_skew_with_single_element_vector(self):
        v = np.array([1])
        with self.assertRaises(AssertionError):
            utils.skew(v)

    def test_skew_with_two_element_vector(self):
        v = np.array([1, 2])
        with self.assertRaises(AssertionError):
            utils.skew(v)

    def test_skew_with_four_element_vector(self):
        v = np.array([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            utils.skew(v)


if __name__ == "__main__":
    absltest.main()


This code snippet includes additional test cases to ensure robustness, checks for various input shapes, and ensures that comments are properly formatted. The `test_skew_returns_correct_skew_symmetric_matrix` test case verifies that the `skew` function returns the correct skew-symmetric matrix for a given input vector. Additional test cases check the behavior of the `skew` function with zero vectors and vectors of incorrect shapes, ensuring that an `AssertionError` is raised as expected.