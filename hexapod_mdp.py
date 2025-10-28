#!/usr/bin/env python3

import mujoco
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from seher.types import JaxRandomKey
import os


@dataclass
class HexapodState:
    """Complete state representation for the hexapod robot - this IS the full MuJoCo state"""
    qpos: jnp.ndarray  # Full position state vector (nq,)
    qvel: jnp.ndarray  # Full velocity state vector (nv,)


@dataclass
class HexapodMDP:
    """Pure functional Hexapod robot MDP implementation for seher"""
    
    xml_path: str
    dt: float
    n_ctrl: int
    control_min: jnp.ndarray
    control_max: jnp.ndarray
    discount: float = 0.99
    
    @classmethod
    def create(cls, xml_path: str, dt: float = 0.01):
        """Create HexapodMDP by reading control info from MuJoCo model"""
        # Get absolute path to XML file
        if not os.path.isabs(xml_path):
            xml_path = os.path.join(os.path.dirname(__file__), xml_path)
        
        # Load model to read control information
        model = mujoco.MjModel.from_xml_path(xml_path)
        
        # Get number of actuators from model
        n_ctrl = model.nu
        
        # Read control bounds from model
        if model.actuator_ctrlrange is not None and model.actuator_ctrlrange.size > 0:
            ctrl_min = jnp.array(model.actuator_ctrlrange[:, 0])
            ctrl_max = jnp.array(model.actuator_ctrlrange[:, 1])
        else:
            # Default bounds if not specified
            ctrl_min = jnp.full(n_ctrl, -1.0)
            ctrl_max = jnp.full(n_ctrl, 1.0)
        
        return cls(
            xml_path=xml_path,
            dt=dt,
            n_ctrl=n_ctrl,
            control_min=ctrl_min,
            control_max=ctrl_max
        )
    
    def empty_control(self) -> jnp.ndarray:
        """Return an empty control vector"""
        return jnp.zeros(self.n_ctrl)
    
    def init(self, key: JaxRandomKey) -> HexapodState:
        """Initialize the robot in a standing position"""
        model = mujoco.MjModel.from_xml_path(self.xml_path)
        data = mujoco.MjData(model)
        
        # Reset to default state
        mujoco.mj_resetData(model, data)
        
        # Set initial joint positions for standing pose
        leg_names = ["northeast", "east", "southeast", "southwest", "west", "northwest"]
        for leg in leg_names:
            knee_joint_name = f"{leg}_knee_pitch"
            try:
                knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, knee_joint_name)
                if knee_id >= 0:
                    data.qpos[model.jnt_qposadr[knee_id]] = -0.525
            except:
                pass
        
        # Forward kinematics to get consistent state
        mujoco.mj_forward(model, data)
        
        # Add small random noise to initial state
        noise_scale = 0.01
        qpos_noise = jax.random.normal(key, shape=(model.nq,)) * noise_scale
        qvel_noise = jax.random.normal(key, shape=(model.nv,)) * noise_scale * 0.1
        
        qpos = jnp.array(data.qpos) + qpos_noise
        qvel = jnp.array(data.qvel) + qvel_noise
        
        return HexapodState(qpos=qpos, qvel=qvel)
    
    def transit(self, state: HexapodState, control: jnp.ndarray, key: JaxRandomKey) -> HexapodState:
        """Simulate one step forward - pure function"""
        model = mujoco.MjModel.from_xml_path(self.xml_path)
        data = mujoco.MjData(model)
        
        # Set state
        data.qpos[:] = np.array(state.qpos)
        data.qvel[:] = np.array(state.qvel)
        
        # Apply control directly to all actuators
        control = jnp.clip(control, self.control_min, self.control_max)
        data.ctrl[:] = np.array(control)
        
        # Simulate multiple steps
        steps_per_control = int(self.dt / model.opt.timestep)
        for _ in range(steps_per_control):
            mujoco.mj_step(model, data)
        
        return HexapodState(
            qpos=jnp.array(data.qpos.copy()),
            qvel=jnp.array(data.qvel.copy())
        )
    
    def cost(self, state: HexapodState, control: jnp.ndarray, key: JaxRandomKey) -> float:
        """Calculate cost (negative reward) - pure function"""
        # Extract relevant state information
        base_pos = state.qpos[:3]  # x, y, z position of base
        base_quat = state.qpos[3:7]  # quaternion orientation
        base_vel = state.qvel[:6]  # linear and angular velocity
        
        # Forward velocity reward (negative cost)
        forward_velocity = base_vel[0]
        velocity_reward = forward_velocity * 10.0
        
        # Height reward (staying upright)
        height = base_pos[2]
        height_reward = jnp.maximum(0, height - 0.1) * 5.0
        
        # Action penalty
        action_penalty = jnp.sum(jnp.square(control)) * 0.1
        
        # Stability penalty (approximate roll and pitch from quaternion)
        qw, qx, qy, qz = base_quat
        # Approximate small angle roll/pitch
        roll_approx = 2 * qx  # approximate for small angles
        pitch_approx = 2 * qy  # approximate for small angles
        stability_penalty = (jnp.abs(roll_approx) + jnp.abs(pitch_approx)) * 5.0
        
        # Terminal cost if robot falls
        fall_penalty = jnp.where(height < 0.05, 100.0, 0.0)
        
        # Return negative reward as cost
        total_reward = velocity_reward + height_reward - action_penalty - stability_penalty - fall_penalty
        return -total_reward


def simple_walking_pattern(step: int, n_ctrl: int) -> jnp.ndarray:
    """Generate a simple sinusoidal walking pattern"""
    # Create different phase offsets for different actuators
    phases = []
    for i in range(6):  # 6 legs
        leg_phase = i * jnp.pi / 3  # 60 degree phase offset between legs
        phases.extend([leg_phase, leg_phase + jnp.pi/4, leg_phase + jnp.pi/2])  # yaw, hip, knee
    
    # Generate actions
    actions = []
    for i in range(n_ctrl):
        phase = phases[i % len(phases)]
        freq = 2.0  # Hz
        amplitude = 0.3
        action = amplitude * jnp.sin(2 * jnp.pi * freq * step * 0.01 + phase)  # 0.01 is dt
        actions.append(action)
    
    return jnp.array(actions)


def main():
    """Test the hexapod MDP"""
    mdp = HexapodMDP.create("hexagonal_robot.xml")
    key = jax.random.PRNGKey(42)
    
    print("Testing pure functional Hexapod MDP...")
    print(f"MDP is pytree: {jax.tree_util.tree_structure(mdp)}")
    
    # Print control information read from model
    print(f"Number of actuators: {mdp.n_ctrl}")
    print(f"Control bounds min: {mdp.control_min}")
    print(f"Control bounds max: {mdp.control_max}")
    
    # Initialize
    key, subkey = jax.random.split(key)
    state = mdp.init(subkey)
    
    print(f"Initial state - height: {state.qpos[2]:.3f}")
    print(f"State qpos shape: {state.qpos.shape}")
    print(f"State qvel shape: {state.qvel.shape}")
    
    # Run simulation
    total_cost = 0.0
    for step in range(100):  # Shorter test since model loading is expensive
        # Generate walking pattern
        control = simple_walking_pattern(step, mdp.n_ctrl)
        
        # Calculate cost
        key, subkey = jax.random.split(key)
        cost = mdp.cost(state, control, subkey)
        
        # Step forward
        key, subkey = jax.random.split(key)
        state = mdp.transit(state, control, subkey)
        
        total_cost += cost
        
        if step % 20 == 0:
            height = state.qpos[2]
            vel_x = state.qvel[0]
            print(f"Step {step}: cost={cost:.3f}, height={height:.3f}, vel_x={vel_x:.3f}")
            
            # Check if robot fell
            if height < 0.05:
                print(f"Robot fell at step {step}")
                break
    
    print(f"Total cost over {step+1} steps: {total_cost:.3f}")
    print("Test completed!")


if __name__ == "__main__":
    main()