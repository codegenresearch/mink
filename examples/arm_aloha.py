from pathlib import Path

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

# Single arm velocity limits.
_VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}

def setup_tasks(model):
    return [
        mink.FrameTask(
            frame_name="left/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        mink.FrameTask(
            frame_name="right/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        mink.PostureTask(model, cost=1e-4),
    ]

def setup_limits(model, joint_names, velocity_limits):
    l_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)
    r_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)
    l_geoms = mink.get_subtree_geom_ids(model, model.body("left/upper_arm_link").id)
    r_geoms = mink.get_subtree_geom_ids(model, model.body("right/upper_arm_link").id)
    frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)
    collision_pairs = [
        (l_wrist_geoms, r_wrist_geoms),
        (l_geoms + r_geoms, frame_geoms + ["table"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )

    return [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, velocity_limits),
        collision_avoidance_limit,
    ]

def setup_dof_and_actuator_ids(model, joint_names):
    return (
        np.array([model.joint(name).id for name in joint_names]),
        np.array([model.actuator(name).id for name in joint_names]),
    )

def initialize_mocap_targets(model, data):
    mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
    mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")

def main():
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    joint_names = [f"{prefix}/{n}" for prefix in ["left", "right"] for n in _JOINT_NAMES]
    velocity_limits = {name: _VELOCITY_LIMITS[name.split('/')[-1]] for name in joint_names}

    dof_ids, actuator_ids = setup_dof_and_actuator_ids(model, joint_names)
    configuration = mink.Configuration(model)
    tasks = setup_tasks(model)
    limits = setup_limits(model, joint_names, velocity_limits)

    l_mid = model.body("left/target").mocapid[0]
    r_mid = model.body("right/target").mocapid[0]
    solver = "quadprog"
    pos_threshold = 1e-2
    ori_threshold = 1e-2
    max_iters = 2

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        tasks[-1].set_target_from_configuration(configuration)
        initialize_mocap_targets(model, data)

        rate = RateLimiter(frequency=200.0)
        while viewer.is_running():
            tasks[0].set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
            tasks[1].set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

            for _ in range(max_iters):
                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    rate.dt,
                    solver,
                    limits=limits,
                    damping=1e-5,
                )
                configuration.integrate_inplace(vel, rate.dt)

                if all(np.linalg.norm(err[:3]) <= pos_threshold and np.linalg.norm(err[3:]) <= ori_threshold
                       for err in (tasks[0].compute_error(configuration), tasks[1].compute_error(configuration))):
                    break

            data.ctrl[actuator_ids] = configuration.q[dof_ids]
            mujoco.mj_step(model, data)
            viewer.sync()
            rate.sleep()

if __name__ == "__main__":
    main()