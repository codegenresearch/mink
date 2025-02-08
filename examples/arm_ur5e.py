from pathlib import Path\n\nimport mujoco\nimport mujoco.viewer\nimport numpy as np\nfrom loop_rate_limiters import RateLimiter\n\nimport mink\n\n_HERE = Path(__file__).parent\n_XML = _HERE / "universal_robots_ur5e" / "scene.xml"\n\n\nif __name__ == "__main__":\n    model = mujoco.MjModel.from_xml_path(_XML.as_posix())\n\n    configuration = mink.Configuration(model)\n\n    tasks = [\n        end_effector_task := mink.FrameTask(\n            frame_name="attachment_site",\n            frame_type="site",\n            position_cost=1.0,\n            orientation_cost=1.0,\n            lm_damping=1.0,\n        ),\n    ]\n\n    # Enable collision avoidance between the following geoms:\n    wrist_3_geoms = mink.get_body_geom_ids(model, model.body("wrist_3_link").id)\n    collision_pairs = [\n        (wrist_3_geoms, ["floor", "wall"]),\n    ]\n\n    limits = [\n        mink.ConfigurationLimit(model=model),\n        mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),\n    ]\n\n    max_velocities = {\n        "shoulder_pan": np.pi,\n        "shoulder_lift": np.pi,\n        "elbow": np.pi,\n        "wrist_1": np.pi,\n        "wrist_2": np.pi,\n        "wrist_3": np.pi,\n    }\n    velocity_limit = mink.VelocityLimit(model, max_velocities)\n    limits.append(velocity_limit)\n\n    mid = model.body("target").mocapid[0]\n    model = configuration.model\n    data = configuration.data\n    solver = "quadprog"\n\n    with mujoco.viewer.launch_passive(\n        model=model, data=data, show_left_ui=False, show_right_ui=False\n    ) as viewer:\n        mujoco.mjv_defaultFreeCamera(model, viewer.cam)\n\n        # Initialize to the home keyframe.\n        configuration.update_from_keyframe("home")\n\n        # Initialize the mocap target at the end-effector site.\n        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")\n\n        rate = RateLimiter(frequency=200.0, warn=False)\n        while viewer.is_running():\n            # Update task target.\n            T_wt = mink.SE3.from_mocap_name(model, data, "target")\n            end_effector_task.set_target(T_wt)\n\n            # Compute velocity and integrate into the next configuration.\n            vel = mink.solve_ik(\n                configuration, tasks, rate.dt, solver, damping=1e-3, limits=limits\n            )\n            configuration.integrate_inplace(vel, rate.dt)\n            mujoco.mj_camlight(model, data)\n\n            # Note the below are optional: they are used to visualize the output of the\n            # fromto sensor which is used by the collision avoidance constraint.\n            mujoco.mj_fwdPosition(model, data)\n            mujoco.mj_sensorPos(model, data)\n\n            # Visualize at fixed FPS.\n            viewer.sync()\n            rate.sleep()\n