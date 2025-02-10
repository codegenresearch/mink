"""Tests utils.py."""

import numpy as np
from absl.testing import absltest

from mink.lie import utils


class TestUtils(absltest.TestCase):
    def test_skew_throws_assertion_error_if_shape_invalid(self):
        with self.assertRaises(AssertionError):
            utils.skew(np.zeros((5,)))  # Invalid shape, should raise AssertionError

    def test_skew_equals_negative(self):
        omega: np.ndarray = np.random.randn(3)
        m: np.ndarray = utils.skew(omega)
        np.testing.assert_allclose(m.T, -m, err_msg="Skew matrix is not skew-symmetric")

    def test_skew_returns_correct_skew_symmetric_matrix(self):
        omega: np.ndarray = np.array([1.0, 2.0, 3.0])
        expected_m: np.ndarray = np.array([[0.0, -3.0, 2.0],
                                          [3.0, 0.0, -1.0],
                                          [-2.0, 1.0, 0.0]])
        m: np.ndarray = utils.skew(omega)
        np.testing.assert_allclose(m, expected_m, err_msg="Skew matrix does not match expected output")


if __name__ == "__main__":
    absltest.main()