"""Tests for frame_task.py."""

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import SE3, SO3, Configuration
from mink.tasks import FrameTask, TargetNotSet, TaskDefinitionError


class TestFrameTask(absltest.TestCase):
    """Test consistency of the frame task."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_robot_description("g1_mj_description")

    def setUp(self) -> None:
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("stand")

    def test_cost_correctly_broadcast(self) -> None:
        task = FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=5.0,
        )
        np.testing.assert_array_equal(task.cost, np.array([1, 1, 1, 5, 5, 5]))

        task = FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=[1.0, 2.0, 3.0],
            orientation_cost=[5.0, 6.0, 7.0],
        )
        np.testing.assert_array_equal(task.cost, np.array([1, 2, 3, 5, 6, 7]))

    def test_task_raises_error_if_cost_dim_invalid(self) -> None:
        with self.assertRaises(TaskDefinitionError):
            FrameTask(
                frame_name="pelvis",
                frame_type="body",
                position_cost=[1.0, 2.0],  # type: list[float]
                orientation_cost=2.0,
            )
        with self.assertRaises(TaskDefinitionError):
            FrameTask(
                frame_name="pelvis",
                frame_type="body",
                position_cost=7.0,
                orientation_cost=[2.0, 5.0],  # type: list[float]
            )

    def test_task_raises_error_if_cost_negative(self) -> None:
        with self.assertRaises(TaskDefinitionError):
            FrameTask(
                frame_name="pelvis",
                frame_type="body",
                position_cost=1.0,
                orientation_cost=-1.0,
            )
        with self.assertRaises(TaskDefinitionError):
            FrameTask(
                frame_name="pelvis",
                frame_type="body",
                position_cost=[-1.0, 1.5],  # type: list[float]
                orientation_cost=[1, 2, 3],  # type: list[int]
            )

    def test_error_without_target(self) -> None:
        task = FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        with self.assertRaises(TargetNotSet):
            task.compute_error(self.configuration)

    def test_jacobian_without_target(self) -> None:
        task = FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        with self.assertRaises(TargetNotSet):
            task.compute_jacobian(self.configuration)

    def test_set_target_from_configuration(self) -> None:
        task = FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        task.set_target_from_configuration(self.configuration)
        pose = self.configuration.get_transform_frame_to_world("pelvis", "body")
        np.testing.assert_array_equal(
            task.transform_target_to_world.translation(), pose.translation()
        )
        np.testing.assert_array_equal(
            task.transform_target_to_world.rotation().wxyz, pose.rotation().wxyz
        )

    def test_target_is_a_copy(self) -> None:
        task = FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        target = self.configuration.get_transform_frame_to_world("pelvis", "body")
        task.set_target(target)
        y = target.translation()[1]
        target.translation()[1] += 12.0
        self.assertAlmostEqual(task.transform_target_to_world.translation()[1], y)
        self.assertNotAlmostEqual(
            task.transform_target_to_world.translation()[1],
            target.translation()[1],
        )

    def test_zero_error_when_target_at_body(self) -> None:
        task = FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        task.set_target_from_configuration(self.configuration)
        error = task.compute_error(self.configuration)
        np.testing.assert_allclose(error, np.zeros(6))

    def test_unit_cost_qp_objective(self) -> None:
        """Unit cost means the QP objective is exactly (J^T J, -e^T J)."""
        task = FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=0.0,
        )
        transform_target_to_frame = SE3.from_rotation_and_translation(
            rotation=SO3.identity(),
            translation=np.array([0.0, 0.01, 0.0]),
        )
        target = (
            self.configuration.get_transform_frame_to_world("pelvis", "body")
            @ transform_target_to_frame
        )
        task.set_target(target)
        J = task.compute_jacobian(self.configuration)
        e = task.compute_error(self.configuration)
        H, c = task.compute_qp_objective(self.configuration)
        np.testing.assert_allclose(H, J.T @ J)
        np.testing.assert_allclose(c, e.T @ J)

    def test_lm_damping_has_no_effect_at_target(self) -> None:
        """Levenberg-Marquardt damping has no effect when the error is zero."""
        task = FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        task.set_target_from_configuration(self.configuration)
        task.lm_damping = 1e-8
        H_1, c_1 = task.compute_qp_objective(self.configuration)
        task.lm_damping = 1e-4
        H_2, c_2 = task.compute_qp_objective(self.configuration)
        self.assertTrue(np.allclose(H_1, H_2))
        self.assertTrue(np.allclose(c_1, c_2))

    def test_zero_cost_same_as_disabling_task(self) -> None:
        task = FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=0.0,
        )
        task.set_target_from_configuration(self.configuration)
        objective = task.compute_qp_objective(self.configuration)
        x = np.random.random(self.configuration.nv)
        cost = objective.value(x)
        self.assertAlmostEqual(cost, 0.0)

    def test_additional_configurations(self) -> None:
        # Test with different frame types and names
        frame_names = ["base", "end_effector"]
        frame_types = ["body", "geometric"]
        for frame_name in frame_names:
            for frame_type in frame_types:
                task = FrameTask(
                    frame_name=frame_name,
                    frame_type=frame_type,
                    position_cost=1.0,
                    orientation_cost=1.0,
                )
                task.set_target_from_configuration(self.configuration)
                error = task.compute_error(self.configuration)
                np.testing.assert_allclose(error, np.zeros(6))
                J = task.compute_jacobian(self.configuration)
                self.assertIsNotNone(J)


if __name__ == "__main__":
    absltest.main()