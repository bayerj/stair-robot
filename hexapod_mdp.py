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
    cost_coefficients: dict = field(default_factory=dict, pytree_node=False)
    
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
        
        # Default cost coefficients
        default_coefficients = {
            'forward_velocity': 10.0,
            'height': 5.0,
            'action_penalty': 0.1,
            'stability': 5.0,
            'fall_penalty': 100.0,
            'joint_deviation': 1.0,
            'orientation': 2.0,
        }
        
        return cls(
            xml_path=xml_path,
            dt=dt,
            n_ctrl=n_ctrl,
            control_min=ctrl_min,
            control_max=ctrl_max,
            cost_coefficients=default_coefficients
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
    
    def _cost_forward_velocity(self, state: HexapodState, control: jnp.ndarray, key: JaxRandomKey) -> float:
        """Forward velocity reward component"""
        base_vel = state.qvel[:6]
        # Y-axis is the long direction of the hexagonal base (size 0.2 x 0.4 x 0.05)
        forward_velocity = base_vel[1]
        return -forward_velocity  # Negative because we want to maximize forward velocity
    
    def _cost_height(self, state: HexapodState, control: jnp.ndarray, key: JaxRandomKey) -> float:
        """Height maintenance reward component"""
        base_pos = state.qpos[:3]
        height = base_pos[2]
        return -jnp.maximum(0, height - 0.1)  # Negative because we want to maximize height above 0.1
    
    def _cost_action_penalty(self, state: HexapodState, control: jnp.ndarray, key: JaxRandomKey) -> float:
        """Action penalty for deviation from default/rest joint target positions"""
        # Create target control values (target joint angles) - same as joint deviation penalty
        # For a 6-legged hexapod with 3 actuators per leg = 18 total actuators
        # Each leg has: yaw, hip_pitch, knee_pitch actuators (in that order)
        # Knee actuators are at indices: 2, 5, 8, 11, 14, 17 (every 3rd starting from 2)
        # Most targets are 0.0, but knees should be at -0.525 for standing pose

        # Build target array assuming we have 18 actuators (pad if needed)
        # This is JAX-trace-safe as it doesn't check runtime shapes
        target_controls = jnp.array([
            0.0, 0.0, -0.525,  # northeast leg: yaw, hip, knee
            0.0, 0.0, -0.525,  # east leg
            0.0, 0.0, -0.525,  # southeast leg
            0.0, 0.0, -0.525,  # southwest leg
            0.0, 0.0, -0.525,  # west leg
            0.0, 0.0, -0.525,  # northwest leg
        ])

        # Slice to match control size (handles cases with fewer actuators)
        target_controls = target_controls[:control.size]

        # Calculate squared deviation from target control values
        return jnp.sum(jnp.square(control - target_controls))
    
    def _cost_stability(self, state: HexapodState, control: jnp.ndarray, key: JaxRandomKey) -> float:
        """Stability penalty (roll and pitch)"""
        base_quat = state.qpos[3:7]
        qw, qx, qy, qz = base_quat
        # Approximate small angle roll/pitch
        roll_approx = 2 * qx
        pitch_approx = 2 * qy
        return jnp.abs(roll_approx) + jnp.abs(pitch_approx)
    
    def _cost_fall_penalty(self, state: HexapodState, control: jnp.ndarray, key: JaxRandomKey) -> float:
        """Terminal penalty if robot falls"""
        base_pos = state.qpos[:3]
        height = base_pos[2]
        return jnp.where(height < 0.05, 1.0, 0.0)  # Binary penalty
    
    def _cost_joint_deviation(self, state: HexapodState, control: jnp.ndarray, key: JaxRandomKey) -> float:
        """Penalty for deviation from default/rest joint angles"""
        # The default joint configuration for standing pose:
        # - Base pose (7): [0, 0, 0.3, 1, 0, 0, 0] (position + quaternion) - skip these
        # - Most joints: 0.0 radians (default)
        # - Knee joints: -0.525 radians (for standing pose)

        # Joint positions start after the base (skip first 7 elements: 3 pos + 4 quat)
        joint_qpos = state.qpos[7:]  # All joint angles

        # Build target array assuming we have 18 joints (6 legs × 3 joints/leg)
        # Each leg has: yaw, hip_pitch, knee_pitch
        # This is JAX-trace-safe as it doesn't check runtime shapes
        target_angles = jnp.array([
            0.0, 0.0, -0.525,  # northeast leg
            0.0, 0.0, -0.525,  # east leg
            0.0, 0.0, -0.525,  # southeast leg
            0.0, 0.0, -0.525,  # southwest leg
            0.0, 0.0, -0.525,  # west leg
            0.0, 0.0, -0.525,  # northwest leg
        ])

        # Slice to match joint_qpos size (handles cases with fewer joints)
        target_angles = target_angles[:joint_qpos.size]

        # Calculate squared deviation from target angles
        deviation = jnp.sum(jnp.square(joint_qpos - target_angles))
        return deviation
    
    def _cost_orientation(self, state: HexapodState, control: jnp.ndarray, key: JaxRandomKey) -> float:
        """Penalty for deviation from upright orientation (pitch and roll only, allow yaw)"""
        # Base quaternion: qpos[3:7] = [qw, qx, qy, qz]
        base_quat = state.qpos[3:7]
        qw, qx, qy, qz = base_quat
        
        # The default/target orientation is upright: [1, 0, 0, 0] (no rotation)
        # We want to penalize pitch and roll but allow yaw rotation
        
        # Method 1: Direct penalty on qx and qy components
        # For small angles, qx ≈ roll/2 and qy ≈ pitch/2
        # So penalizing qx^2 + qy^2 penalizes roll and pitch while allowing qz (yaw)
        pitch_roll_penalty = jnp.square(qx) + jnp.square(qy)
        
        # Alternative method using rotation matrices (more accurate for larger angles):
        # Convert quaternion to rotation matrix and extract pitch/roll
        # But for efficiency and simplicity, the direct method above works well
        
        return pitch_roll_penalty
    
    def cost(self, state: HexapodState, control: jnp.ndarray, key: JaxRandomKey) -> float:
        """Calculate total cost by combining all subcost functions with their coefficients"""
        total_cost = 0.0
        
        # Dynamically find all _cost_* methods and apply them
        for name, coeff in self.cost_coefficients.items():
            cost_method_name = f"_cost_{name}"
            if hasattr(self, cost_method_name):
                cost_method = getattr(self, cost_method_name)
                subcost = cost_method(state, control, key)
                total_cost += coeff * subcost
            else:
                # Warn about missing cost function but don't fail
                pass
        
        return total_cost
    
    def get_cost_breakdown(self, state: HexapodState, control: jnp.ndarray, key: JaxRandomKey) -> dict:
        """Return breakdown of individual cost components for debugging/tuning"""
        breakdown = {}
        
        for name, coeff in self.cost_coefficients.items():
            cost_method_name = f"_cost_{name}"
            if hasattr(self, cost_method_name):
                cost_method = getattr(self, cost_method_name)
                subcost = cost_method(state, control, key)
                breakdown[name] = {
                    'raw_cost': float(subcost),
                    'coefficient': coeff,
                    'weighted_cost': float(coeff * subcost)
                }
        
        breakdown['total_cost'] = float(self.cost(state, control, key))
        return breakdown
    
    def update_cost_coefficient(self, name: str, value: float):
        """Update a cost coefficient for live tuning - DEPRECATED: Use update_mdp_coefficients instead"""
        self.cost_coefficients[name] = value
    
    def with_updated_coefficients(self, **coefficient_updates) -> 'HexapodMDP':
        """
        Create a new MDP instance with updated cost coefficients (JAX-compatible).
        
        Args:
            **coefficient_updates: Keyword arguments for coefficient updates
                                 e.g., forward_velocity=0.1, height=5.0
        
        Returns:
            New HexapodMDP instance with updated coefficients
        """
        new_coefficients = self.cost_coefficients.copy()
        new_coefficients.update(coefficient_updates)
        
        return self.replace(cost_coefficients=new_coefficients)
    
    def list_cost_functions(self) -> list:
        """List all available cost function names"""
        cost_methods = []
        for attr_name in dir(self):
            if attr_name.startswith('_cost_') and callable(getattr(self, attr_name)):
                cost_name = attr_name[6:]  # Remove '_cost_' prefix
                cost_methods.append(cost_name)
        return cost_methods


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