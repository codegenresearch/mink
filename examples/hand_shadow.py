from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "shadow_hand" / "scene_left.xml"

def initialize_tasks(model, fingers):
    finger_tasks = [
        mink.FrameTask(
            frame_name=finger,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        for finger in fingers
    ]
    posture_task = mink.PostureTask(model, cost=1e-2)
    return [posture_task, *finger_tasks]

def initialize_mocap_positions(model, data, fingers):
    for finger in fingers:
        mink.move_mocap_to_frame(model, data, f"{finger}_target", finger, "site")

def update_task_targets(model, data, fingers, finger_tasks):
    for finger, task in zip(fingers, finger_tasks):
        task.set_target(mink.SE3.from_mocap_name(model, data, f"{finger}_target"))

def main():
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    configuration = mink.Configuration(model)
    fingers = ["thumb", "first", "middle", "ring", "little"]
    tasks = initialize_tasks(model, fingers)
    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        configuration.update_from_keyframe("grasp hard")
        posture_task = tasks[0]
        posture_task.set_target_from_configuration(configuration)
        initialize_mocap_positions(model, data, fingers)

        rate = RateLimiter(frequency=500.0)  # Adjusted frequency for better performance
        while viewer.is_running():
            update_task_targets(model, data, fingers, tasks[1:])
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-5)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()

if __name__ == "__main__":
    main()