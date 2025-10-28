#!/usr/bin/env python3

import mujoco
import numpy as np
from seher.systems.mujoco_playground import MujocoPlaygroundMDP
import os
import jax
import jax.numpy as jnp

class HexapodEnv:
    """Hexapod robot environment using seher MujocoPlaygroundMDP"""
    
    def __init__(self, xml_path="hexagonal_robot.xml", render_mode="human"):
        # Get absolute path to XML file
        if not os.path.isabs(xml_path):
            xml_path = os.path.join(os.path.dirname(__file__), xml_path)
        
        # Initialize the MuJoCo playground
        self.mdp = MujocoPlaygroundMDP(xml_path)
        self.model = self.mdp.model
        self.data = self.mdp.data
        
        # Get joint names and indices for control
        self.leg_names = ["northeast", "east", "southeast", "southwest", "west", "northwest"]
        self.joint_names = []
        self.actuator_names = []
        
        for leg in self.leg_names:
            for joint_type in ["yaw", "hip_pitch", "knee_pitch"]:
                joint_name = f"{leg}_{joint_type}"
                motor_name = f"{leg}_{joint_type}_motor"
                self.joint_names.append(joint_name)
                self.actuator_names.append(motor_name)
        
        self.n_joints = len(self.joint_names)
        print(f"Hexapod environment initialized with {self.n_joints} actuated joints")
        
        # Initialize viewer if needed
        self.viewer = None
        self.render_mode = render_mode
        
    def reset(self, seed=None):
        """Reset the environment"""
        # Reset to initial state
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial joint positions to make robot stand
        for leg in self.leg_names:
            knee_joint_name = f"{leg}_knee_pitch"
            try:
                knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, knee_joint_name)
                if knee_id >= 0:
                    self.data.qpos[self.model.jnt_qposadr[knee_id]] = -0.525
            except:
                pass
        
        # Forward dynamics to update state
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs()
    
    def _get_obs(self):
        """Get observation from the environment"""
        # Base position and orientation
        base_pos = self.data.qpos[:3].copy()  # x, y, z of base
        base_quat = self.data.qpos[3:7].copy()  # quaternion of base
        base_vel = self.data.qvel[:6].copy()  # linear and angular velocity of base
        
        # Joint positions and velocities
        joint_pos = []
        joint_vel = []
        
        for joint_name in self.joint_names:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0:
                    joint_pos.append(self.data.qpos[self.model.jnt_qposadr[joint_id]])
                    joint_vel.append(self.data.qvel[self.model.jnt_dofadr[joint_id]])
                else:
                    joint_pos.append(0.0)
                    joint_vel.append(0.0)
            except:
                joint_pos.append(0.0)
                joint_vel.append(0.0)
        
        # Concatenate all observations
        obs = np.concatenate([
            base_pos,      # 3
            base_quat,     # 4  
            base_vel,      # 6
            joint_pos,     # 18
            joint_vel      # 18
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action):
        """Step the environment with the given action"""
        # Clip actions to reasonable range
        action = np.clip(action, -1.0, 1.0)
        
        # Apply actions to actuators
        for i, actuator_name in enumerate(self.actuator_names):
            try:
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                if actuator_id >= 0 and i < len(action):
                    self.data.ctrl[actuator_id] = action[i]
            except:
                pass
        
        # Step the simulation (5 steps)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        # Get observation
        obs = self._get_obs()
        
        # Calculate reward (simple forward motion reward)
        base_vel_x = self.data.qvel[0]  # forward velocity
        base_pos_z = self.data.qpos[2]  # height
        
        # Reward forward motion and staying upright
        reward = base_vel_x * 10.0  # reward forward motion
        reward += max(0, base_pos_z - 0.1) * 5.0  # reward staying above ground
        
        # Penalty for excessive action
        action_penalty = np.sum(np.square(action)) * 0.1
        reward -= action_penalty
        
        # Check if episode is done (robot fell over)
        done = base_pos_z < 0.05  # robot is too low
        
        # Info dictionary
        info = {
            "base_height": base_pos_z,
            "forward_velocity": base_vel_x,
            "reward_components": {
                "forward": base_vel_x * 10.0,
                "height": max(0, base_pos_z - 0.1) * 5.0,
                "action_penalty": -action_penalty,
            }
        }
        
        return obs, reward, done, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            if self.viewer is None:
                try:
                    import mujoco.viewer
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                except Exception as e:
                    print(f"Could not launch viewer: {e}")
                    return
            
            if self.viewer is not None:
                self.viewer.sync()


def simple_walking_pattern(step, n_joints):
    """Generate a simple sinusoidal walking pattern"""
    # Create different phase offsets for different legs
    phases = []
    for i in range(6):  # 6 legs
        leg_phase = i * np.pi / 3  # 60 degree phase offset between legs
        phases.extend([leg_phase, leg_phase + np.pi/4, leg_phase + np.pi/2])  # yaw, hip, knee
    
    # Generate actions
    actions = []
    for i in range(n_joints):
        phase = phases[i % len(phases)]
        freq = 2.0  # Hz
        amplitude = 0.3
        action = amplitude * np.sin(2 * np.pi * freq * step * 0.002 + phase)  # 0.002 is timestep
        actions.append(action)
    
    return np.array(actions)


def main():
    """Test the hexapod environment"""
    env = HexapodEnv()
    
    print(f"Observation space shape: {env._get_obs().shape}")
    print(f"Action space size: {len(env.actuator_names)}")
    
    # Reset environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Run a walking test
    print("Testing walking pattern...")
    for step in range(1000):
        # Use simple walking pattern
        action = simple_walking_pattern(step, len(env.actuator_names))
        
        obs, reward, done, info = env.step(action)
        
        if step % 100 == 0:
            print(f"Step {step}: reward={reward:.3f}, height={info['base_height']:.3f}, "
                  f"vel_x={info['forward_velocity']:.3f}")
        
        if done:
            print(f"Episode ended at step {step}")
            obs = env.reset()
    
    print("Test completed!")


if __name__ == "__main__":
    main()