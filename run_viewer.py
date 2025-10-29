#!/usr/bin/env python3

"""
MuJoCo viewer for hexapod robot with async control policy support.

This script launches a MuJoCo viewer and runs a control policy in parallel,
allowing real-time interaction with the robot simulation.
"""

import asyncio
import logging
import mujoco
import mujoco.viewer
import numpy as np
import os
import time
from typing import Optional
import defopt
from rich.logging import RichHandler
from rich.console import Console
import jax
import jax.numpy as jnp
from seher.control.mpc import MPCPolicy
from seher.control.stepper_planner import StepperPlanner  
from seher.stepper.mppi import GaussianMPPIOptimizer
from hexapod_mdp import HexapodMDP, HexapodState
import threading
import concurrent.futures

# Set up rich logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


def simple_walking_pattern(step: int, n_ctrl: int, frequency: float = 2.0, dt: float = 0.002) -> np.ndarray:
    """Generate a simple sinusoidal walking pattern"""
    # Create different phase offsets for different actuators
    phases = []
    for i in range(6):  # 6 legs
        leg_phase = i * np.pi / 3  # 60 degree phase offset between legs
        phases.extend([leg_phase, leg_phase + np.pi/4, leg_phase + np.pi/2])  # yaw, hip, knee
    
    # Generate actions
    actions = []
    for i in range(n_ctrl):
        phase = phases[i % len(phases)]
        amplitude = 0.5
        action = amplitude * np.sin(2 * np.pi * frequency * step * dt + phase)  # Use actual dt
        actions.append(action)
    
    return np.array(actions)


def standing_pose(model: mujoco.MjModel, data: mujoco.MjData):
    """Set the robot to a standing pose"""
    leg_names = ["northeast", "east", "southeast", "southwest", "west", "northwest"]
    
    for leg in leg_names:
        knee_joint_name = f"{leg}_knee_pitch"
        try:
            knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, knee_joint_name)
            if knee_id >= 0:
                data.qpos[model.jnt_qposadr[knee_id]] = -0.525
        except:
            logger.warning(f"Could not find joint {knee_joint_name}")
    
    mujoco.mj_forward(model, data)


class HexapodController:
    """Async controller for the hexapod robot"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, 
                 control_frequency: float = 50.0, gait_frequency: float = 2.0,
                 controller_type: str = "simple", xml_path: str = "hexagonal_robot.xml",
                 slow_motion: float = 1.0):
        self.model = model
        self.data = data
        self.control_dt = 1.0 / control_frequency
        self.gait_frequency = gait_frequency
        self.step_count = 0
        self.running = False
        self.controller_type = controller_type
        self.slow_motion = slow_motion
        
        # Timing monitoring
        self.last_step_time = None
        self.timing_violations = 0
        self.total_steps_timed = 0
        
        # Thread executor for control computation
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # Initialize appropriate controller
        if controller_type == "mppi":
            # Create MDP for MPPI with control frequency dt
            self.mdp = HexapodMDP.create(xml_path, dt=self.control_dt)
            self.jax_key = jax.random.PRNGKey(42)
            
            # Create MPPI planner using seher framework
            planner = StepperPlanner(
                mdp=self.mdp,
                n_iter=3,  # Small for real-time
                n_plan_steps=4,  # Short horizon for real-time
                warm_start=True,
                optimizer=GaussianMPPIOptimizer(
                    objective=None,
                    n_candidates=8,
                    top_k=2,
                    min_scale=0.01,
                    temperature=1.0,
                    # Use 1D controls per timestep like the working LQR test
                    initial_loc=jnp.zeros(self.mdp.n_ctrl),  # (n_ctrl,) 
                    initial_scale=jnp.ones(self.mdp.n_ctrl) * 0.3,  # (n_ctrl,)
                ),
            )
            
            # Create MPC policy
            self.mpc_policy = MPCPolicy(mdp=self.mdp, planner=planner)
            self.policy_carry = self.mpc_policy.initial_carry()
            
            logger.info(f"Seher MPC+MPPI Controller initialized: {control_frequency}Hz control")
        else:
            logger.info(f"Simple Controller initialized: {control_frequency}Hz control, {gait_frequency}Hz gait")
    
    def _compute_mppi_control(self, jax_state, subkey):
        """Compute MPPI control in a separate thread"""
        self.policy_carry, control = self.mpc_policy(
            carry=self.policy_carry,
            obs=jax_state,
            control=self.mdp.empty_control(),
            key=subkey
        )
        return np.array(control)
    
    def _compute_simple_control(self):
        """Compute simple control in a separate thread"""
        return simple_walking_pattern(
            self.step_count,
            self.model.nu,
            self.gait_frequency,
            self.control_dt
        )
    
    async def run(self):
        """Main control loop"""
        self.running = True
        logger.info(f"Starting {self.controller_type} control loop")
        
        while self.running:
            step_start_time = time.time()
            
            # Check timing from previous step
            if self.last_step_time is not None:
                actual_dt = step_start_time - self.last_step_time
                target_dt = self.control_dt
                
                if actual_dt > target_dt * 1.5:  # 50% tolerance
                    self.timing_violations += 1
                    if self.timing_violations <= 5:  # Only warn for first few violations
                        logger.warning(f"Control frequency violation: {actual_dt:.3f}s actual vs {target_dt:.3f}s target")
                
                self.total_steps_timed += 1
                
                # Log timing summary every 100 steps
                if self.total_steps_timed % 100 == 0:
                    violation_rate = self.timing_violations / self.total_steps_timed * 100
                    avg_frequency = 1.0 / (actual_dt if actual_dt > 0 else target_dt)
                    if violation_rate > 10:  # More than 10% violations
                        logger.warning(f"Timing summary: {violation_rate:.1f}% violations, avg freq: {avg_frequency:.1f}Hz")
            
            self.last_step_time = step_start_time
            # Measure control computation time
            control_start_time = time.time()
            
            if self.controller_type == "mppi":
                # MPC+MPPI control using seher framework in a separate thread
                # Convert MuJoCo state to JAX state
                jax_state = HexapodState(
                    qpos=jnp.array(self.data.qpos.copy()),
                    qvel=jnp.array(self.data.qvel.copy())
                )
                
                # Get control from MPC policy in thread
                self.jax_key, subkey = jax.random.split(self.jax_key)
                loop = asyncio.get_event_loop()
                control = await loop.run_in_executor(
                    self.executor, 
                    self._compute_mppi_control, 
                    jax_state, 
                    subkey
                )
            else:
                # Simple walking pattern in thread
                loop = asyncio.get_event_loop()
                control = await loop.run_in_executor(
                    self.executor,
                    self._compute_simple_control
                )
            
            control_time = time.time() - control_start_time
            
            # Warn if control computation takes too long
            if control_time > self.control_dt * 0.8:  # 80% of available time
                logger.warning(f"Control computation slow: {control_time:.3f}s (target: {self.control_dt:.3f}s)")
            
            # Apply control (viewer handles simulation stepping)
            self.data.ctrl[:] = control
            
            self.step_count += 1
            
            # Log occasionally
            log_freq = 50 if self.controller_type == "mppi" else 250  # More frequent for MPPI
            if self.step_count % log_freq == 0:
                height = self.data.qpos[2]
                vel_x = self.data.qvel[0]
                ctrl_norm = np.linalg.norm(control)
                if self.controller_type == "mppi":
                    logger.info(f"Step {self.step_count}: height={height:.3f}m, vel_x={vel_x:.3f}m/s, |ctrl|={ctrl_norm:.3f}, ctrl_time={control_time:.3f}s")
                else:
                    logger.info(f"Step {self.step_count}: height={height:.3f}m, vel_x={vel_x:.3f}m/s, |ctrl|={ctrl_norm:.3f}")
            
            # Sleep until next control step (adjusted for slow motion)
            adjusted_control_dt = self.control_dt * (1.0 / self.slow_motion)
            await asyncio.sleep(adjusted_control_dt)
    
    def stop(self):
        """Stop the control loop"""
        self.running = False
        # Shutdown thread executor
        self.executor.shutdown(wait=False)
        logger.info("Control loop stopped")


class AsyncMuJoCoViewer:
    """Async wrapper for MuJoCo viewer with configurable simulation frequency"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, 
                 sim_frequency: float = 500.0, slow_motion: float = 1.0):
        self.model = model
        self.data = data
        self.viewer = None
        self.running = False
        self.slow_motion = slow_motion
        self.sim_frequency = sim_frequency
        
        # Calculate simulation timestep for target frequency, adjusted for slow motion
        self.sim_dt = (1.0 / sim_frequency) / slow_motion
        
    async def run(self):
        """Run the viewer loop"""
        try:
            logger.info("Launching MuJoCo viewer...")
            
            # Use passive viewer for better control over simulation stepping
            try:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    self.viewer = viewer
                    self.running = True
                    logger.info("Viewer launched successfully (passive mode)")
                    
                    # Viewer loop - step simulation at target frequency
                    last_step_time = time.time()
                    logger.info(f"Viewer running at {self.sim_frequency}Hz simulation frequency (slow motion: {self.slow_motion}x)")
                    
                    while self.running and viewer.is_running():
                        current_time = time.time()
                        
                        # Step simulation at target frequency with slow motion
                        if current_time - last_step_time >= self.sim_dt:
                            mujoco.mj_step(self.model, self.data)
                            last_step_time = current_time
                        
                        # Sync viewer to display current state
                        viewer.sync()
                        
                        # Sleep for a short time to prevent busy-waiting
                        await asyncio.sleep(0.0001)  # Very short sleep
                    
                    logger.info("Viewer loop ended")
                    
            except RuntimeError as e:
                raise RuntimeError(f"Passive viewer failed: {e}. Make sure MuJoCo is properly installed with mjpython support.")
                
        except Exception as e:
            logger.error(f"Could not launch viewer: {e}")
            logger.info("Try running with --headless flag for simulation without viewer")
            raise
    
    def stop(self):
        """Stop the viewer"""
        self.running = False
        if self.viewer:
            logger.info("Viewer stopped")


async def run_simulation(
    xml_path: str,
    control_frequency: float = 50.0,
    gait_frequency: float = 2.0,
    headless: bool = False,
    controller_type: str = "simple",
    slow_motion: float = 1.0,
    sim_frequency: float = 500.0
):
    """Run the hexapod simulation with async viewer and controller"""
    
    # Load model
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(os.path.dirname(__file__), xml_path)
    
    logger.info(f"Loading model from {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    logger.info(f"Model loaded: {model.nu} actuators, {model.nq} positions, {model.nv} velocities")
    
    # Set initial pose
    standing_pose(model, data)
    logger.info("Robot set to standing pose")
    
    # Create controller
    controller = HexapodController(
        model, data, 
        control_frequency=control_frequency,
        gait_frequency=gait_frequency,
        controller_type=controller_type,
        xml_path=xml_path,
        slow_motion=slow_motion
    )
    
    # Create tasks
    tasks = []
    
    # Add controller task
    controller_task = asyncio.create_task(controller.run())
    tasks.append(controller_task)
    
    # Add viewer task if not headless
    if not headless:
        viewer = AsyncMuJoCoViewer(model, data, sim_frequency=sim_frequency, slow_motion=slow_motion)
        viewer_task = asyncio.create_task(viewer.run())
        tasks.append(viewer_task)
        
        # Set up cleanup for viewer
        def cleanup():
            viewer.stop()
            controller.stop()
    else:
        logger.info("Running in headless mode")
        
        # Set up cleanup for headless mode
        def cleanup():
            controller.stop()
    
    try:
        # Wait for any task to complete (usually viewer closing)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        logger.info("Simulation ended")
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    
    finally:
        # Clean up
        cleanup()
        
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Cleanup completed")


def main(
    xml_path: str = "hexagonal_robot.xml",
    *,
    control_frequency: float = 50.0,
    gait_frequency: float = 2.0,
    headless: bool = False,
    controller: str = "simple",
    slow_motion: float = 1.0,
    sim_frequency: float = 500.0
):
    """
    Run MuJoCo viewer with async hexapod controller.
    
    Args:
        xml_path: Path to the MuJoCo XML model file
        control_frequency: Control loop frequency in Hz
        gait_frequency: Walking gait frequency in Hz (for simple controller)  
        headless: Run without viewer (simulation only)
        controller: Controller type ('simple' or 'mppi')
        slow_motion: Slow motion factor (1.0 = real time, 0.5 = half speed, 2.0 = double speed)
        sim_frequency: Simulation frequency in Hz (how fast the physics runs)
    """
    
    logger.info("🤖 Starting Hexapod Robot Simulation")
    logger.info(f"Model: {xml_path}")
    logger.info(f"Controller: {controller}")
    logger.info(f"Control frequency: {control_frequency} Hz")
    if controller == "simple":
        logger.info(f"Gait frequency: {gait_frequency} Hz")
    logger.info(f"Headless mode: {headless}")
    logger.info(f"Simulation frequency: {sim_frequency} Hz")
    if slow_motion != 1.0:
        logger.info(f"Slow motion: {slow_motion}x speed")
    
    # Run the async simulation
    asyncio.run(run_simulation(
        xml_path=xml_path,
        control_frequency=control_frequency,
        gait_frequency=gait_frequency,
        headless=headless,
        controller_type=controller,
        slow_motion=slow_motion,
        sim_frequency=sim_frequency
    ))
    
    logger.info("🏁 Simulation finished")


if __name__ == "__main__":
    defopt.run(main)
