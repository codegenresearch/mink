from pathlib import Path
from typing import Optional, Sequence

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "scene.xml"

# Single arm joint names.
_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]

# Single arm velocity limits, taken from:
# https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro
_VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}


def compensate_gravity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    subtree_ids: Sequence[int],
    qfrc_applied: Optional[np.ndarray] = None,
) -> None:
    """Compute forces to counteract gravity for the given subtrees.\n\n    Args:\n        model: Mujoco model.\n        data: Mujoco data.\n        subtree_ids: List of subtree ids. A subtree is defined as the kinematic tree\n            starting at the body and including all its descendants. Gravity\n            compensation forces will be applied to all bodies in the subtree.\n        qfrc_applied: Optional array to store the computed forces. If not provided,\n            the applied forces in `data` are used.\n    """
    qfrc_applied = data.qfrc_applied if qfrc_applied is None else qfrc_applied
    qfrc_applied[:] = 0.0  # Don't accumulate from previous calls.\n    jac = np.empty((3, model.nv))\n    for subtree_id in subtree_ids:\n        total_mass = model.body_subtreemass[subtree_id]\n        mujoco.mj_jacSubtreeCom(model, data, jac, subtree_id)\n        qfrc_applied[:] -= model.opt.gravity * total_mass @ jac\n\n\nif __name__ == "__main__":\n    model = mujoco.MjModel.from_xml_path(str(_XML))\n    data = mujoco.MjData(model)\n\n    # Bodies for which to apply gravity compensation.\n    left_subtree_id = model.body("left/base_link").id\n    right_subtree_id = model.body("right/base_link").id\n\n    # Get the dof and actuator ids for the joints we wish to control.\n    joint_names: list[str] = []\n    velocity_limits: dict[str, float] = {}\n    for prefix in ["left", "right"]:\n        for n in _JOINT_NAMES:\n            name = f"{prefix}/{n}"\n            joint_names.append(name)\n            velocity_limits[name] = _VELOCITY_LIMITS[n]\n    dof_ids = np.array([model.joint(name).id for name in joint_names])\n    actuator_ids = np.array([model.actuator(name).id for name in joint_names])\n\n    configuration = mink.Configuration(model)\n\n    tasks = [\n        l_ee_task := mink.FrameTask(\n            frame_name="left/gripper",\n            frame_type="site",\n            position_cost=1.0,\n            orientation_cost=1.0,\n            lm_damping=1.0,\n        ),\n        r_ee_task := mink.FrameTask(\n            frame_name="right/gripper",\n            frame_type="site",\n            position_cost=1.0,\n            orientation_cost=1.0,\n            lm_damping=1.0,\n        ),\n        posture_task := mink.PostureTask(model, cost=1e-4),\n    ]\n\n    # Enable collision avoidance between the following geoms.\n    l_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)\n    r_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)\n    l_geoms = mink.get_subtree_geom_ids(model, model.body("left/upper_arm_link").id)\n    r_geoms = mink.get_subtree_geom_ids(model, model.body("right/upper_arm_link").id)\n    frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)\n    collision_pairs = [\n        (l_wrist_geoms, r_wrist_geoms),\n        (l_geoms + r_geoms, frame_geoms + ["table"]),\n    ]\n    collision_avoidance_limit = mink.CollisionAvoidanceLimit(\n        model=model,\n        geom_pairs=collision_pairs,  # type: ignore\n        minimum_distance_from_collisions=0.05,\n        collision_detection_distance=0.1,\n    )\n\n    limits = [\n        mink.ConfigurationLimit(model=model),\n        mink.VelocityLimit(model, velocity_limits),\n        collision_avoidance_limit,\n    ]\n\n    l_mid = model.body("left/target").mocapid[0]\n    r_mid = model.body("right/target").mocapid[0]\n    solver = "quadprog"\n    pos_threshold = 5e-3\n    ori_threshold = 5e-3\n    max_iters = 5\n\n    with mujoco.viewer.launch_passive(\n        model=model, data=data, show_left_ui=False, show_right_ui=False\n    ) as viewer:\n        mujoco.mjv_defaultFreeCamera(model, viewer.cam)\n\n        # Initialize to the home keyframe.\n        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)\n        configuration.update(data.qpos)\n        mujoco.mj_forward(model, data)\n        posture_task.set_target_from_configuration(configuration)\n\n        # Initialize mocap targets at the end-effector site.\n        mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")\n        mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")\n\n        rate = RateLimiter(frequency=200.0)\n        while viewer.is_running():\n            # Update task targets.\n            l_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))\n            r_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))\n\n            # Compute velocity and integrate into the next configuration.\n            for i in range(max_iters):\n                vel = mink.solve_ik(\n                    configuration,\n                    tasks,\n                    rate.dt,\n                    solver,\n                    limits=limits,\n                    damping=1e-5,\n                )\n                configuration.integrate_inplace(vel, rate.dt)\n\n                l_err = l_ee_task.compute_error(configuration)\n                l_pos_achieved = np.linalg.norm(l_err[:3]) <= pos_threshold\n                l_ori_achieved = np.linalg.norm(l_err[3:]) <= ori_threshold\n                r_err = l_ee_task.compute_error(configuration)\n                r_pos_achieved = np.linalg.norm(r_err[:3]) <= pos_threshold\n                r_ori_achieved = np.linalg.norm(r_err[3:]) <= ori_threshold\n                if (\n                    l_pos_achieved\n                    and l_ori_achieved\n                    and r_pos_achieved\n                    and r_ori_achieved\n                ):\n                    break\n\n            data.ctrl[actuator_ids] = configuration.q[dof_ids]\n            compensate_gravity(model, data, [left_subtree_id, right_subtree_id])\n            mujoco.mj_step(model, data)\n\n            # Visualize at fixed FPS.\n            viewer.sync()\n            rate.sleep()