from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "hello_robot_stretch_3" / "scene.xml"

def construct_model(xml_path):
    return mujoco.MjModel.from_xml_path(xml_path.as_posix())

def initialize_tasks(model):
    return [
        mink.FrameTask(
            frame_name="base_link",
            frame_type="body",
            position_cost=0.1,
            orientation_cost=1.0,
        ),
        mink.FrameTask(
            frame_name="link_grasp_center",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1e-4,
        ),
    ]

def main():
    model = construct_model(_XML_PATH)
    configuration = mink.Configuration(model)
    tasks = initialize_tasks(model)

    base_target_mocapid = model.body("base_target").mocapid[0]
    solver = "quadprog"
    circle_radius = 0.5

    with mujoco.viewer.launch_passive(
        model=model, data=configuration.data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")
        base_task = tasks[0]
        base_task.set_target_from_configuration(configuration)
        assert base_task.transform_target_to_world is not None

        fingertip_task = tasks[1]
        transform_fingertip_target_to_world = (
            configuration.get_transform_frame_to_world("link_grasp_center", "site")
        )
        center_translation = transform_fingertip_target_to_world.translation()[:2]
        fingertip_task.set_target(transform_fingertip_target_to_world)
        mink.move_mocap_to_frame(model, configuration.data, "EE_target", "link_grasp_center", "site")

        rate_limiter = RateLimiter(frequency=100.0, warn=True)
        t = 0.0

        while viewer.is_running():
            # Update task targets
            u = np.array([np.cos(t / 2), np.sin(t / 2)])
            T = base_task.transform_target_to_world.copy()
            translation = T.translation()
            translation[:2] = center_translation + circle_radius * u
            configuration.data.mocap_pos[base_target_mocapid] = translation
            configuration.data.mocap_quat[base_target_mocapid] = mink.SO3.from_rpy_radians(
                0.0, 0.0, 0.5 * np.pi * t
            ).wxyz
            base_task.set_target(mink.SE3.from_mocap_id(configuration.data, base_target_mocapid))

            # Compute velocity and integrate into the next configuration.
            velocity = mink.solve_ik(configuration, tasks, rate_limiter.dt, solver, 1e-3)
            configuration.integrate_inplace(velocity, rate_limiter.dt)
            mujoco.mj_camlight(model, configuration.data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate_limiter.sleep()
            t += rate_limiter.dt

if __name__ == "__main__":
    main()