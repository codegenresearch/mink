from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink
from typing import Optional, Sequence

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

def get_subtree_body_ids(model: mujoco.MjModel, body_name: str) -> list[int]:
    """Retrieve all body IDs in the subtree starting from the given body name."""
    body_id = model.body(body_name).id
    subtree_ids = [body_id]
    for i in range(body_id + 1, model.nbody):
        if model.body_parentid[i] in subtree_ids:
            subtree_ids.append(i)
    return subtree_ids

def get_subtree_geom_ids(model: mujoco.MjModel, body_name: str) -> list[int]:
    """Retrieve all geom IDs in the subtree starting from the given body name."""
    body_ids = get_subtree_body_ids(model, body_name)
    geom_ids = []
    for body_id in body_ids:
        geom_ids.extend(model.body_geomadr[body_id] + np.arange(model.body_geomnum[body_id]))
    return geom_ids

def compensate_gravity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    subtree_ids: Sequence[int]
) -> None:
    """
    Compute and apply forces to counteract gravity for the specified subtree.

    Args:
    - model (mujoco.MjModel): The MuJoCo model.
    - data (mujoco.MjData): The MuJoCo data.
    - subtree_ids (Sequence[int]): List of body IDs in the subtree.
    """
    # Initialize Jacobian and COM position arrays
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    com_pos = np.zeros(3)

    # Compute the Jacobian and COM position for the subtree
    mujoco.mj_jacSubtreeCom(model, data, jacp, jacr, com_pos, subtree_ids[0])

    # Compute the gravity compensation force
    gravity_compensation = -model.opt.gravity[2] * data.mass[subtree_ids[0]] * jacp[:, :3]

    # Apply the gravity compensation force to qfrc_applied
    data.qfrc_applied += gravity_compensation.flatten()

def test_get_subtree_body_ids():
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    left_wrist_body_ids = get_subtree_body_ids(model, "left/wrist_link")
    right_wrist_body_ids = get_subtree_body_ids(model, "right/wrist_link")
    assert len(left_wrist_body_ids) > 0
    assert len(right_wrist_body_ids) > 0
    assert set(left_wrist_body_ids).isdisjoint(right_wrist_body_ids)

def test_get_subtree_geom_ids():
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    left_wrist_geom_ids = get_subtree_geom_ids(model, "left/wrist_link")
    right_wrist_geom_ids = get_subtree_geom_ids(model, "right/wrist_link")
    assert len(left_wrist_geom_ids) > 0
    assert len(right_wrist_geom_ids) > 0
    assert set(left_wrist_geom_ids).isdisjoint(right_wrist_geom_ids)

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Get the dof and actuator ids for the joints we wish to control.
    joint_names: list[str] = []
    velocity_limits: dict[str, float] = {}
    for prefix in ["left", "right"]:
        for n in _JOINT_NAMES:
            name = f"{prefix}/{n}"
            joint_names.append(name)
            velocity_limits[name] = _VELOCITY_LIMITS[n]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    configuration = mink.Configuration(model)

    tasks = [
        l_ee_task := mink.FrameTask(
            frame_name="left/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        r_ee_task := mink.FrameTask(
            frame_name="right/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=1e-4),
    ]

    # Enable collision avoidance between the following geoms:
    # geoms starting at subtree "right wrist" - "table",
    # geoms starting at subtree "left wrist"  - "table",
    # geoms starting at subtree "right wrist" - geoms starting at subtree "left wrist".
    l_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)
    r_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)
    l_geoms = mink.get_subtree_geom_ids(model, model.body("left/upper_arm_link").id)
    r_geoms = mink.get_subtree_geom_ids(model, model.body("right/upper_arm_link").id)
    frame_geoms = mink.get_subtree_geom_ids(model, model.body("metal_frame").id)
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

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, velocity_limits),
        collision_avoidance_limit,
    ]

    l_mid = model.body("left/target").mocapid[0]
    r_mid = model.body("right/target").mocapid[0]
    solver = "quadprog"
    pos_threshold = 5e-3
    ori_threshold = 5e-3
    max_iters = 5

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        posture_task.set_target_from_configuration(configuration)

        # Initialize mocap targets at the end-effector site.
        mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
        mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")

        rate = RateLimiter(frequency=200.0)
        while viewer.is_running():
            # Update task targets.
            l_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
            r_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    rate.dt,
                    solver,
                    limits=limits,
                    damping=1e-5,
                )
                configuration.integrate_inplace(vel, rate.dt)

                l_err = l_ee_task.compute_error(configuration)
                l_pos_achieved = np.linalg.norm(l_err[:3]) <= pos_threshold
                l_ori_achieved = np.linalg.norm(l_err[3:]) <= ori_threshold
                r_err = r_ee_task.compute_error(configuration)
                r_pos_achieved = np.linalg.norm(r_err[:3]) <= pos_threshold
                r_ori_achieved = np.linalg.norm(r_err[3:]) <= ori_threshold
                if (
                    l_pos_achieved
                    and l_ori_achieved
                    and r_pos_achieved
                    and r_ori_achieved
                ):
                    break

            data.ctrl[actuator_ids] = configuration.q[dof_ids]

            # Apply gravity compensation
            left_subtree_ids = get_subtree_body_ids(model, "left/base_link")
            right_subtree_ids = get_subtree_body_ids(model, "right/base_link")
            compensate_gravity(model, data, left_subtree_ids)
            compensate_gravity(model, data, right_subtree_ids)

            mujoco.mj_step(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()