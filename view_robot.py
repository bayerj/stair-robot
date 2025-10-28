#!/usr/bin/env python3

import mujoco
import mujoco.viewer
import time

def main():
    # Load the model
    model = mujoco.MjModel.from_xml_path("hexagonal_robot.xml")
    data = mujoco.MjData(model)
    
    # Set initial joint positions to make the robot stand
    # Set knee joints to bend backward to support the body
    for leg in ["northeast", "east", "southeast", "southwest", "west", "northwest"]:
        knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_knee_pitch")
        if knee_id >= 0:
            data.qpos[model.jnt_qposadr[knee_id]] = -0.525  # Bend knees backward (75% of original)
    
    # Initialize the visualization
    mujoco.mj_forward(model, data)
    
    print("Model loaded successfully. Robot has:")
    print(f"- {model.njnt} joints")
    print(f"- {model.nbody} bodies")
    print("Joint names:")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name:
            print(f"  {joint_name}")
    
    print("\nLaunching MuJoCo viewer...")
    
    # Launch the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Run the simulation
        while viewer.is_running():
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Sync the viewer
            viewer.sync()

if __name__ == "__main__":
    main()