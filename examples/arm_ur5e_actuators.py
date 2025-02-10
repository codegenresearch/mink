from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "universal_robots_ur5e" / "scene.xml"

def setup_collision_avoidance(model):
    wrist_3_geoms = mink.get_body_geom_ids(model, model.body("wrist_3_link").id)
    collision_pairs = [(wrist_3_geoms, ["floor", "wall"])]
    return mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs)

def setup_velocity_limit(model):
    max_velocities = {
        "shoulder_pan": np.pi,
        "shoulder_lift": np.pi,
        "elbow": np.pi,
        "wrist_1": np.pi,
        "wrist_2": np.pi,
        "wrist_3": np.pi,
    }
    return mink.VelocityLimit(model, max_velocities)

def setup_tasks(model):
    return [
        mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

def setup_limits(model, configuration):
    collision_avoidance_limit = setup_collision_avoidance(model)
    velocity_limit = setup_velocity_limit(model)
    return [
        mink.ConfigurationLimit(model=configuration.model),
        collision_avoidance_limit,
        velocity_limit,
    ]

def main():
    model = mujoco.MjModel.from_xml_path(_XML_PATH.as_posix())
    data = mujoco.MjData(model)

    configuration = mink.Configuration(model)
    tasks = setup_tasks(model)
    limits = setup_limits(model, configuration)

    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    mid = model.body("target").mocapid[0]

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate = RateLimiter(frequency=500.0)
        while viewer.is_running():
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            tasks[0].set_target(T_wt)

            for _ in range(max_iters):
                vel = mink.solve_ik(
                    configuration, tasks, rate.dt, solver, damping=1e-3, limits=limits
                )
                configuration.integrate_inplace(vel, rate.dt)
                err = tasks[0].compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    break

            data.ctrl = configuration.q
            mujoco.mj_step(model, data)

            viewer.sync()
            rate.sleep()

if __name__ == "__main__":
    main()