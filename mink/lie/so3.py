from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Type

import mujoco
import numpy as np

from .base import MatrixLieGroup
from .utils import get_epsilon, skew

_IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_INVERT_QUAT_SIGN = np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float64)


class RollPitchYaw(NamedTuple):
    """Represents roll, pitch, and yaw angles in radians."""

    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
class SO3(MatrixLieGroup):
    """Special orthogonal group for 3D rotations.\n\n    Internal parameterization is (qw, qx, qy, qz). Tangent parameterization is\n    (omega_x, omega_y, omega_z).\n    """

    wxyz: np.ndarray
    matrix_dim: int = 3
    parameters_dim: int = 4
    tangent_dim: int = 3
    space_dim: int = 3

    def __post_init__(self) -> None:
        """Validates that the wxyz array has the correct shape."""
        if self.wxyz.shape != (self.parameters_dim,):
            raise ValueError(f"Expected wxyz to be a length 4 vector but got {self.wxyz.shape[0]}.")

    def __repr__(self) -> str:
        """Provides a string representation of the SO3 instance."""
        wxyz = np.round(self.wxyz, 5)
        return f"{self.__class__.__name__}(wxyz={wxyz})"

    def parameters(self) -> np.ndarray:
        """Returns the internal parameterization of the rotation."""
        return self.wxyz

    def copy(self) -> SO3:
        """Returns a copy of the SO3 instance."""
        return SO3(wxyz=self.wxyz.copy())

    @classmethod
    def from_x_radians(cls: Type['SO3'], theta: float) -> SO3:
        """Creates an SO3 instance from a rotation around the x-axis."""
        return SO3.exp(np.array([theta, 0.0, 0.0], dtype=np.float64))

    @classmethod
    def from_y_radians(cls: Type['SO3'], theta: float) -> SO3:
        """Creates an SO3 instance from a rotation around the y-axis."""
        return SO3.exp(np.array([0.0, theta, 0.0], dtype=np.float64))

    @classmethod
    def from_z_radians(cls: Type['SO3'], theta: float) -> SO3:
        """Creates an SO3 instance from a rotation around the z-axis."""
        return SO3.exp(np.array([0.0, 0.0, theta], dtype=np.float64))

    @classmethod
    def from_rpy_radians(
        cls: Type['SO3'],
        roll: float,
        pitch: float,
        yaw: float,
    ) -> SO3:
        """Creates an SO3 instance from roll, pitch, and yaw angles."""
        return (
            SO3.from_z_radians(yaw)
            @ SO3.from_y_radians(pitch)
            @ SO3.from_x_radians(roll)
        )

    @classmethod
    def from_matrix(cls: Type['SO3'], matrix: np.ndarray) -> SO3:
        """Creates an SO3 instance from a rotation matrix."""
        if matrix.shape != (SO3.matrix_dim, SO3.matrix_dim):
            raise ValueError(f"Expected a 3x3 matrix but got {matrix.shape}.")
        wxyz = np.zeros(SO3.parameters_dim, dtype=np.float64)
        mujoco.mju_mat2Quat(wxyz, matrix.ravel())
        return SO3(wxyz=wxyz)

    @classmethod
    def identity(cls: Type['SO3']) -> SO3:
        """Creates an identity SO3 instance."""
        return SO3(wxyz=_IDENTITY_WXYZ)

    @classmethod
    def sample_uniform(cls: Type['SO3']) -> SO3:
        """Samples a random SO3 instance uniformly."""
        # Ref: https://lavalle.pl/planning/node198.html
        u1, u2, u3 = np.random.uniform(
            low=np.zeros(shape=(3,)),
            high=np.array([1.0, 2.0 * np.pi, 2.0 * np.pi]),
        )
        a = np.sqrt(1.0 - u1)
        b = np.sqrt(u1)
        wxyz = np.array(
            [
                a * np.sin(u2),
                a * np.cos(u2),
                b * np.sin(u3),
                b * np.cos(u3),
            ],
            dtype=np.float64,
        )
        return SO3(wxyz=wxyz)

    def as_matrix(self) -> np.ndarray:
        """Converts the SO3 instance to a rotation matrix."""
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, self.wxyz)
        return mat.reshape(3, 3)

    def compute_roll_radians(self) -> float:
        """Computes the roll angle in radians."""
        q0, q1, q2, q3 = self.wxyz
        return np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))

    def compute_pitch_radians(self) -> float:
        """Computes the pitch angle in radians."""
        q0, q1, q2, q3 = self.wxyz
        return np.arcsin(2 * (q0 * q2 - q3 * q1))

    def compute_yaw_radians(self) -> float:
        """Computes the yaw angle in radians."""
        q0, q1, q2, q3 = self.wxyz
        return np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

    def as_rpy_radians(self) -> RollPitchYaw:
        """Converts the SO3 instance to roll, pitch, and yaw angles."""
        return RollPitchYaw(
            roll=self.compute_roll_radians(),
            pitch=self.compute_pitch_radians(),
            yaw=self.compute_yaw_radians(),
        )

    def inverse(self) -> SO3:
        """Computes the inverse of the SO3 instance."""
        return SO3(wxyz=self.wxyz * _INVERT_QUAT_SIGN)

    def normalize(self) -> SO3:
        """Normalizes the quaternion to ensure it is a unit quaternion."""
        return SO3(wxyz=self.wxyz / np.linalg.norm(self.wxyz))

    def apply(self, target: np.ndarray) -> np.ndarray:
        """Applies the rotation to a target vector."""
        if target.shape != (SO3.space_dim,):
            raise ValueError(f"Expected a 3-dimensional vector but got {target.shape}.")
        padded_target = np.concatenate([np.zeros(1, dtype=np.float64), target])
        result = (self @ SO3(wxyz=padded_target) @ self.inverse()).wxyz[1:]
        return result

    def multiply(self, other: SO3) -> SO3:
        """Multiplies the current SO3 instance with another SO3 instance."""
        res = np.empty(self.parameters_dim, dtype=np.float64)
        mujoco.mju_mulQuat(res, self.wxyz, other.wxyz)
        return SO3(wxyz=res)

    @classmethod
    def exp(cls: Type['SO3'], tangent: np.ndarray) -> SO3:
        """Exponentiates a tangent vector to obtain an SO3 instance."""
        if tangent.shape != (SO3.tangent_dim,):
            raise ValueError(f"Expected a 3-dimensional tangent vector but got {tangent.shape}.")
        theta_squared = tangent @ tangent
        theta_pow_4 = theta_squared * theta_squared
        use_taylor = theta_squared < get_epsilon(tangent.dtype)
        safe_theta = 1.0 if use_taylor else np.sqrt(theta_squared)
        safe_half_theta = 0.5 * safe_theta
        if use_taylor:
            real = 1.0 - theta_squared / 8.0 + theta_pow_4 / 384.0
            imaginary = 0.5 - theta_squared / 48.0 + theta_pow_4 / 3840.0
        else:
            real = np.cos(safe_half_theta)
            imaginary = np.sin(safe_half_theta) / safe_theta
        wxyz = np.concatenate([np.array([real]), imaginary * tangent])
        return SO3(wxyz=wxyz)

    def log(self) -> np.ndarray:
        """Computes the logarithm of the SO3 instance to obtain a tangent vector."""
        w = self.wxyz[0]
        norm_sq = self.wxyz[1:] @ self.wxyz[1:]
        use_taylor = norm_sq < get_epsilon(norm_sq.dtype)
        norm_safe = 1.0 if use_taylor else np.sqrt(norm_sq)
        w_safe = w if use_taylor else 1.0
        atan_n_over_w = np.arctan2(-norm_safe if w < 0 else norm_safe, abs(w))
        if use_taylor:
            atan_factor = 2.0 / w_safe - 2.0 / 3.0 * norm_sq / w_safe**3
        else:
            if abs(w) < get_epsilon(w.dtype):
                scl = 1.0 if w > 0.0 else -1.0
                atan_factor = scl * np.pi / norm_safe
            else:
                atan_factor = 2.0 * atan_n_over_w / norm_safe
        return atan_factor * self.wxyz[1:]

    def adjoint(self) -> np.ndarray:
        """Computes the adjoint representation of the SO3 instance."""
        return self.as_matrix()

    @classmethod
    def ljac(cls: Type['SO3'], other: np.ndarray) -> np.ndarray:
        """Computes the left Jacobian of the SO3 instance."""
        theta = np.sqrt(other @ other)
        use_taylor = theta < get_epsilon(theta.dtype)
        if use_taylor:
            t2 = theta**2
            A = (1.0 / 2.0) * (1.0 - t2 / 12.0 * (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0)))
            B = (1.0 / 6.0) * (1.0 - t2 / 20.0 * (1.0 - t2 / 42.0 * (1.0 - t2 / 72.0)))
        else:
            A = (1 - np.cos(theta)) / (theta**2)
            B = (theta - np.sin(theta)) / (theta**3)
        skew_other = skew(other)
        return np.eye(3) + A * skew_other + B * (skew_other @ skew_other)

    @classmethod
    def ljacinv(cls: Type['SO3'], other: np.ndarray) -> np.ndarray:
        """Computes the inverse of the left Jacobian of the SO3 instance."""
        theta = np.sqrt(other @ other)
        use_taylor = theta < get_epsilon(theta.dtype)
        if use_taylor:
            t2 = theta**2
            A = (1.0 / 12.0) * (1.0 + t2 / 60.0 * (1.0 + t2 / 42.0 * (1.0 + t2 / 40.0)))
        else:
            A = (1.0 / theta**2) * (
                1.0 - (theta * np.sin(theta) / (2.0 * (1.0 - np.cos(theta))))
            )
        skew_other = skew(other)
        return np.eye(3) - 0.5 * skew_other + A * (skew_other @ skew_other)