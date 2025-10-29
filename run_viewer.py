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
from mppi_controller import MPPIController
from hexapod_mdp import HexapodMDP, HexapodState

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
                 controller_type: str = "simple", xml_path: str = "hexagonal_robot.xml"):
        self.model = model
        self.data = data
        self.control_dt = 1.0 / control_frequency
        self.gait_frequency = gait_frequency
        self.step_count = 0
        self.running = False
        self.controller_type = controller_type
        
        # Timing monitoring
        self.last_step_time = None
        self.timing_violations = 0
        self.total_steps_timed = 0
        
        # Initialize appropriate controller
        if controller_type == "mppi":
            # Create MDP for MPPI with control frequency dt
            self.mdp = HexapodMDP.create(xml_path, dt=self.control_dt)
            self.jax_key = jax.random.PRNGKey(42)
            
            # Initialize MPPI controller
            self.mppi_controller = MPPIController(
                mdp=self.mdp,
                horizon=4,
                n_candidates=8,
                top_k=2,  # Must be <= n_candidates
                temperature=1.0,
                noise_scale=0.3
            )
            
            logger.info(f"MPPI Controller initialized: {control_frequency}Hz control")
        else:
            logger.info(f"Simple Controller initialized: {control_frequency}Hz control, {gait_frequency}Hz gait")
    
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
                # MPPI control
                # Convert MuJoCo state to JAX state
                jax_state = HexapodState(
                    qpos=jnp.array(self.data.qpos.copy()),
                    qvel=jnp.array(self.data.qvel.copy())
                )
                
                # Get control from MPPI
                self.jax_key, subkey = jax.random.split(self.jax_key)
                control, _ = self.mppi_controller.get_control(jax_state, subkey)
                control = np.array(control)
            else:
                # Simple walking pattern
                control = simple_walking_pattern(
                    self.step_count, 
                    self.model.nu, 
                    self.gait_frequency,
                    self.control_dt
                )
            
            control_time = time.time() - control_start_time
            
            # Warn if control computation takes too long
            if control_time > self.control_dt * 0.8:  # 80% of available time
                logger.warning(f"Control computation slow: {control_time:.3f}s (target: {self.control_dt:.3f}s)")
            
            # Apply control
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
            
            # Sleep until next control step
            await asyncio.sleep(self.control_dt)
    
    def stop(self):
        """Stop the control loop"""
        self.running = False
        logger.info("Control loop stopped")


class AsyncMuJoCoViewer:
    """Async wrapper for MuJoCo viewer"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.viewer = None
        self.running = False
        
    async def run(self):
        """Run the viewer loop"""
        try:
            logger.info("Launching MuJoCo viewer...")
            
            # Try different viewer approaches
            try:
                # First try the passive viewer (works on Linux/Windows)
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    self.viewer = viewer
                    self.running = True
                    logger.info("Viewer launched successfully (passive mode)")
                    
                    # Viewer loop
                    while self.running and viewer.is_running():
                        # Step simulation
                        mujoco.mj_step(self.model, self.data)
                        
                        # Sync viewer (this should be non-blocking)
                        viewer.sync()
                        
                        # Small sleep to prevent busy waiting
                        await asyncio.sleep(0.001)  # ~1000 FPS max
                    
                    logger.info("Viewer loop ended")
                    
            except RuntimeError as e:
                if "mjpython" in str(e):
                    logger.info("Passive viewer not available, trying active viewer...")
                    
                    # Fallback to active viewer (might work on macOS)
                    viewer = mujoco.viewer.launch(self.model, self.data)
                    self.viewer = viewer
                    self.running = True
                    logger.info("Viewer launched successfully (active mode)")
                    
                    # For active viewer, we don't need a loop - it runs in its own thread
                    while self.running:
                        # Just step the simulation, viewer updates automatically
                        mujoco.mj_step(self.model, self.data)
                        await asyncio.sleep(0.001)
                    
                    logger.info("Viewer loop ended")
                else:
                    raise
                
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
    controller_type: str = "simple"
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
        xml_path=xml_path
    )
    
    # Create tasks
    tasks = []
    
    # Add controller task
    controller_task = asyncio.create_task(controller.run())
    tasks.append(controller_task)
    
    # Add viewer task if not headless
    if not headless:
        viewer = AsyncMuJoCoViewer(model, data)
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
    controller: str = "simple"
):
    """
    Run MuJoCo viewer with async hexapod controller.
    
    Args:
        xml_path: Path to the MuJoCo XML model file
        control_frequency: Control loop frequency in Hz
        gait_frequency: Walking gait frequency in Hz (for simple controller)  
        headless: Run without viewer (simulation only)
        controller: Controller type ('simple' or 'mppi')
    """
    
    logger.info("🤖 Starting Hexapod Robot Simulation")
    logger.info(f"Model: {xml_path}")
    logger.info(f"Controller: {controller}")
    logger.info(f"Control frequency: {control_frequency} Hz")
    if controller == "simple":
        logger.info(f"Gait frequency: {gait_frequency} Hz")
    logger.info(f"Headless mode: {headless}")
    
    # Run the async simulation
    asyncio.run(run_simulation(
        xml_path=xml_path,
        control_frequency=control_frequency,
        gait_frequency=gait_frequency,
        headless=headless,
        controller_type=controller
    ))
    
    logger.info("🏁 Simulation finished")


if __name__ == "__main__":
    defopt.run(main)
