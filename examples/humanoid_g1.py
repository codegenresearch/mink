from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

HERE = Path(__file__).parent
MODEL_XML_PATH = HERE / "unitree_g1" / "scene.xml"

def create_frame_task(frame_name, frame_type, position_cost, orientation_cost, lm_damping=None):
    return mink.FrameTask(
        frame_name=frame_name,
        frame_type=frame_type,
        position_cost=position_cost,
        orientation_cost=orientation_cost,
        lm_damping=lm_damping,
    )

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH.as_posix())

    configuration = mink.Configuration(model)

    feet = ["right_foot", "left_foot"]
    hands = ["right_palm", "left_palm"]

    tasks = [
        create_frame_task("pelvis", "body", 0.0, 10.0),
        mink.PostureTask(model, cost=1.0),
        mink.ComTask(cost=200.0),
    ]

    feet_tasks = [create_frame_task(foot, "site", 200.0, 10.0, lm_damping=1.0) for foot in feet]
    tasks.extend(feet_tasks)

    hand_tasks = [create_frame_task(hand, "site", 200.0, 0.0, lm_damping=1.0) for hand in hands]
    tasks.extend(hand_tasks)

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]

    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("stand")
        mink.PostureTask.set_target_from_configuration(tasks[1], configuration)
        mink.FrameTask.set_target_from_configuration(tasks[0], configuration)

        # Initialize mocap bodies at their respective sites.
        for hand, foot in zip(hands, feet):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
        data.mocap_pos[com_mid] = data.subtree_com[1]

        rate_limiter = RateLimiter(frequency=200.0)
        while viewer.is_running():
            # Update task targets.
            tasks[2].set_target(data.mocap_pos[com_mid])
            for i, (hand_task, foot_task) in enumerate(zip(hand_tasks, feet_tasks)):
                foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
                hand_task.set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))

            velocity = mink.solve_ik(configuration, tasks, rate_limiter.dt, solver, 1e-1)
            configuration.integrate_inplace(velocity, rate_limiter.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate_limiter.sleep()