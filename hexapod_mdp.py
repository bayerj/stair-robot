#!/usr/bin/env python3

import mujoco
import numpy as np
import jax
import jax.numpy as jnp
from flax.struct import dataclass, field
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
    
    xml_path: str = field(pytree_node=False)
    dt: float = field(pytree_node=False)
    n_ctrl: int = field(pytree_node=False)
    control_min: jnp.ndarray = field(pytree_node=False)
    control_max: jnp.ndarray = field(pytree_node=False)
    discount: float = field(default=0.99, pytree_node=False)
    
    def __hash__(self):
        """Make HexapodMDP hashable for JAX compilation."""
        # Hash based on static configuration, not the JAX arrays
        return hash((self.xml_path, self.dt, self.n_ctrl, self.discount))
    
    @classmethod
    def create(cls, xml_path: str, dt: float = None):
        """Create HexapodMDP by reading control info from MuJoCo model"""
        # Get absolute path to XML file
        if not os.path.isabs(xml_path):
            xml_path = os.path.join(os.path.dirname(__file__), xml_path)
        
        # Load model to read control information
        model = mujoco.MjModel.from_xml_path(xml_path)
        
        # Use MuJoCo timestep if dt not specified
        if dt is None:
            dt = model.opt.timestep
        
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
        # This function should be JIT-compiled, so we use a host callback for MuJoCo simulation
        def _mujoco_step(qpos, qvel, ctrl):
            # Load model and data
            model = mujoco.MjModel.from_xml_path(self.xml_path)
            data = mujoco.MjData(model)
            
            # Set state (convert from JAX arrays to numpy)
            data.qpos[:] = np.asarray(qpos)
            data.qvel[:] = np.asarray(qvel)
            data.ctrl[:] = np.asarray(ctrl)
            
            # Simulate multiple steps
            steps_per_control = int(self.dt / model.opt.timestep)
            for _ in range(steps_per_control):
                mujoco.mj_step(model, data)
            
            # Return new state as numpy arrays
            return np.array(data.qpos.copy()), np.array(data.qvel.copy())
        
        # Clip control within bounds
        control = jnp.clip(control, self.control_min, self.control_max)
        
        # Use JAX host callback to call MuJoCo simulation
        result_shape = (jax.ShapeDtypeStruct(state.qpos.shape, state.qpos.dtype),
                       jax.ShapeDtypeStruct(state.qvel.shape, state.qvel.dtype))
        
        new_qpos, new_qvel = jax.experimental.io_callback(
            _mujoco_step,
            result_shape,
            state.qpos, state.qvel, control
        )
        
        return HexapodState(qpos=new_qpos, qvel=new_qvel)
    
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