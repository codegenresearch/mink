from pathlib import Path\n\nimport mujoco\nimport mujoco.viewer\nfrom loop_rate_limiters import RateLimiter\n\nimport mink\n\n_HERE = Path(__file__).parent\n_XML = _HERE / "unitree_h1" / "scene.xml"\n\n\nif __name__ == "__main__":\n    model = mujoco.MjModel.from_xml_path(_XML.as_posix())\n\n    configuration = mink.Configuration(model)\n\n    feet = ["right_foot", "left_foot"]\n    hands = ["right_wrist", "left_wrist"]\n\n    tasks = [\n        mink.FrameTask(\n            frame_name="pelvis",\n            frame_type="body",\n            position_cost=0.0,\n            orientation_cost=10.0,\n        ),\n        mink.PostureTask(model, cost=1.0),\n        mink.ComTask(cost=200.0),\n    ]\n\n    # Create tasks for feet\n    for foot in feet:\n        foot_task = mink.FrameTask(\n            frame_name=foot,\n            frame_type="site",\n            position_cost=200.0,\n            orientation_cost=10.0,\n            lm_damping=1.0,\n        )\n        tasks.append(foot_task)\n\n    # Create tasks for hands\n    for hand in hands:\n        hand_task = mink.FrameTask(\n            frame_name=hand,\n            frame_type="site",\n            position_cost=200.0,\n            orientation_cost=0.0,\n            lm_damping=1.0,\n        )\n        tasks.append(hand_task)\n\n    com_mid = model.body("com_target").mocapid[0]\n    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]\n    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]\n\n    model = configuration.model\n    data = configuration.data\n    solver = "quadprog"\n\n    with mujoco.viewer.launch_passive(\n        model=model, data=data, show_left_ui=False, show_right_ui=False\n    ) as viewer:\n        mujoco.mjv_defaultFreeCamera(model, viewer.cam)\n\n        # Initialize to the home keyframe\n        configuration.update_from_keyframe("stand")\n        posture_task.set_target_from_configuration(configuration)\n        pelvis_orientation_task.set_target_from_configuration(configuration)\n\n        # Initialize mocap bodies at their respective sites\n        for hand, foot in zip(hands, feet):\n            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")\n            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")\n        data.mocap_pos[com_mid] = data.subtree_com[1]\n\n        rate = RateLimiter(frequency=200.0, warn=False)\n        while viewer.is_running():\n            # Update task targets\n            com_task.set_target(data.mocap_pos[com_mid])\n            for i, (hand_task, foot_task) in enumerate(zip(hand_tasks, feet_tasks)):\n                foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))\n                hand_task.set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))\n\n            # Solve inverse kinematics and integrate\n            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1)\n            configuration.integrate_inplace(vel, rate.dt)\n            mujoco.mj_camlight(model, data)\n\n            # Visualize at fixed FPS\n            viewer.sync()\n            rate.sleep()\n