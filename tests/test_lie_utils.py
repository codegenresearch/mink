"""Tests utils.py."""

import numpy as np
from absl.testing import absltest

from mink.lie import utils


class TestUtils(absltest.TestCase):
    def test_skew_throws_assertion_error_if_shape_invalid(self):
        with self.assertRaises(AssertionError):
            utils.skew(np.zeros((5,)))

    def test_skew_throws_assertion_error_if_not_1d_array(self):
        with self.assertRaises(AssertionError):
            utils.skew(np.zeros((3, 3)))

    def test_skew_throws_assertion_error_if_not_numeric(self):
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

    def test_unskew_throws_assertion_error_if_shape_invalid(self):
        with self.assertRaises(AssertionError):
            utils.unskew(np.zeros((5, 5)))

    def test_unskew_throws_assertion_error_if_not_square_matrix(self):
        with self.assertRaises(AssertionError):
            utils.unskew(np.zeros((3, 4)))

    def test_unskew_throws_assertion_error_if_not_skew_symmetric(self):
        with self.assertRaises(AssertionError):
            utils.unskew(np.array([[0, 1, 2],
                                    [3, 0, 4],
                                    [5, 6, 0]]))

    def test_unskew_returns_correct_vector(self):
        m = np.array([[0, -3, 2],
                      [3, 0, -1],
                      [-2, 1, 0]])
        np.testing.assert_allclose(utils.unskew(m), np.array([1, 2, 3]))

    def test_expmap_returns_identity_for_zero_tangent_vector(self):
        v = np.zeros(3)
        np.testing.assert_allclose(utils.expmap(v), np.eye(3))

    def test_expmap_throws_assertion_error_if_invalid_tangent_vector(self):
        with self.assertRaises(AssertionError):
            utils.expmap(np.zeros((3, 3)))

    def test_logmap_returns_zero_vector_for_identity_matrix(self):
        np.testing.assert_allclose(utils.logmap(np.eye(3)), np.zeros(3))

    def test_logmap_throws_assertion_error_if_invalid_rotation_matrix(self):
        with self.assertRaises(AssertionError):
            utils.logmap(np.zeros((3, 3)))


if __name__ == "__main__":
    absltest.main()