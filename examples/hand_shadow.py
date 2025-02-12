from pathlib import Path
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink

_HERE = Path(__file__).parent
_XML = _HERE / "shadow_hand" / "scene_left.xml"

def initialize_model_and_tasks(model_path):
    model = mujoco.MjModel.from_xml_path(model_path.as_posix())
    configuration = mink.Configuration(model)
    
    posture_task = mink.PostureTask(model, cost=1e-2)
    
    fingers = ["thumb", "first", "middle", "ring", "little"]
    finger_tasks = [
        mink.FrameTask(
            frame_name=finger,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        ) for finger in fingers
    ]
    
    tasks = [posture_task, *finger_tasks]
    return model, configuration, tasks

def setup_viewer(model, data):
    viewer = mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False)
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
    return viewer

def initialize_targets(configuration, fingers, finger_tasks, model, data):
    configuration.update_from_keyframe("grasp hard")
    posture_task.set_target_from_configuration(configuration)
    for finger, task in zip(fingers, finger_tasks):
        mink.move_mocap_to_frame(model, data, f"{finger}_target", finger, "site")
        task.set_target(mink.SE3.from_mocap_name(model, data, f"{finger}_target"))

def main():
    model_path = _HERE / "shadow_hand" / "scene_left.xml"
    model, configuration, tasks = initialize_model_and_tasks(model_path)
    data = configuration.data
    solver = "quadprog"
    
    viewer = setup_viewer(model, data)
    fingers = ["thumb", "first", "middle", "ring", "little"]
    finger_tasks = tasks[1:]  # Skip the posture task
    
    initialize_targets(configuration, fingers, finger_tasks, model, data)
    
    rate_limiter = RateLimiter(frequency=500.0)  # Adjusted frequency for better performance
    dt = rate_limiter.dt
    t = 0
    
    while viewer.is_running():
        # Update task targets
        for finger, task in zip(fingers, finger_tasks):
            task.set_target(mink.SE3.from_mocap_name(model, data, f"{finger}_target"))
        
        # Solve inverse kinematics and integrate
        velocity = mink.solve_ik(configuration, tasks, dt, solver, 1e-5)
        configuration.integrate_inplace(velocity, dt)
        mujoco.mj_camlight(model, data)
        
        # Visualize at fixed FPS
        viewer.sync()
        rate_limiter.sleep()
        t += dt

if __name__ == "__main__":
    main()