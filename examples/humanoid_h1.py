from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "unitree_h1" / "scene.xml"

def create_frame_task(frame_name, frame_type, position_cost, orientation_cost, lm_damping=None):
    return mink.FrameTask(
        frame_name=frame_name,
        frame_type=frame_type,
        position_cost=position_cost,
        orientation_cost=orientation_cost,
        lm_damping=lm_damping,
    )

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML_PATH.as_posix())
    configuration = mink.Configuration(model)

    FEET = ["right_foot", "left_foot"]
    HANDS = ["right_wrist", "left_wrist"]

    tasks = [
        create_frame_task("pelvis", "body", 0.0, 10.0),
        mink.PostureTask(model, cost=1.0),
        mink.ComTask(cost=200.0),
    ]

    feet_tasks = [create_frame_task(foot, "site", 200.0, 10.0, lm_damping=1.0) for foot in FEET]
    tasks.extend(feet_tasks)

    hand_tasks = [create_frame_task(hand, "site", 200.0, 0.0, lm_damping=1.0) for hand in HANDS]
    tasks.extend(hand_tasks)

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in FEET]
    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in HANDS]

    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("stand")
        tasks[1].set_target_from_configuration(configuration)  # posture_task
        tasks[0].set_target_from_configuration(configuration)  # pelvis_orientation_task

        # Initialize mocap bodies at their respective sites.
        for hand, foot in zip(HANDS, FEET):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
        data.mocap_pos[com_mid] = data.subtree_com[1]

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task targets.
            tasks[2].set_target(data.mocap_pos[com_mid])  # com_task
            for i, (hand_task, foot_task) in enumerate(zip(hand_tasks, feet_tasks)):
                foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
                hand_task.set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))

            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()