#!/usr/bin/env python3

"""
MPPI-based controller for the hexapod robot using seher framework.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from typing import NamedTuple
import logging

from seher.stepper.mppi import GaussianMPPIOptimizer, GaussianMPPIOptimizerCarry
from seher.control.mpc import calc_costs_of_plan, MPCCarry
from hexapod_mdp import HexapodMDP, HexapodState

logger = logging.getLogger(__name__)


class MPPIObjective:
    """Objective function for MPPI that evaluates control sequences on the hexapod MDP"""
    
    def __init__(self, mdp: HexapodMDP, horizon: int = 20):
        self.mdp = mdp
        self.horizon = horizon
        
    def __call__(self, controls: jnp.ndarray, state: HexapodState, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Evaluate a batch of control sequences
        
        Args:
            controls: (n_candidates, horizon, n_ctrl) control sequences
            state: Current robot state
            key: Random key
            
        Returns:
            costs: (n_candidates,) cost for each sequence
        """
        n_candidates = controls.shape[0]
        
        # Debug shapes
        # print(f"DEBUG: controls shape: {controls.shape}, expected: ({n_candidates}, {self.horizon}, {self.mdp.n_ctrl})")
        
        # Vectorize over candidates
        def evaluate_single_sequence(control_seq, key):
            """Evaluate a single control sequence"""
            current_state = state
            total_cost = 0.0
            
            # Roll out the sequence
            for t in range(self.horizon):
                key, subkey = jax.random.split(key)
                cost = self.mdp.cost(current_state, control_seq[t], subkey)
                
                key, subkey = jax.random.split(key)
                current_state = self.mdp.transit(current_state, control_seq[t], subkey)
                
                total_cost += cost
                
                # Add penalty for low height (instead of early termination)
                height_penalty = jnp.where(current_state.qpos[2] < 0.05, 1000.0, 0.0)
                total_cost += height_penalty
            
            return total_cost
        
        # Split keys for each candidate
        keys = jax.random.split(key, n_candidates)
        
        # Vectorize evaluation
        costs = jax.vmap(evaluate_single_sequence)(controls, keys)
        
        # Debug result
        # print(f"DEBUG: costs shape: {costs.shape}, expected: ({n_candidates},)")
        
        # Return costs and empty auxiliary data (required by seher interface)
        return costs, None


class MPPIController:
    """MPPI-based controller for the hexapod robot using seher framework"""
    
    def __init__(
        self, 
        mdp: HexapodMDP,
        horizon: int = 20,
        n_candidates: int = 1000,
        top_k: int = 100,
        temperature: float = 1.0,
        noise_scale: float = 0.5
    ):
        self.mdp = mdp
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.top_k = top_k
        self.temperature = temperature
        self.noise_scale = noise_scale
        
        # Create MPPI optimizer
        self.objective = MPPIObjective(mdp, horizon)
        
        # Initialize MPPI stepper
        initial_controls = jnp.zeros((horizon, mdp.n_ctrl))
        initial_scale = jnp.ones((horizon, mdp.n_ctrl)) * noise_scale
        
        self.mppi = GaussianMPPIOptimizer(
            objective=self._vectorized_objective,
            n_candidates=n_candidates,
            top_k=top_k,
            initial_loc=initial_controls,
            initial_scale=initial_scale,
            temperature=temperature,
            warm_start=True,
            min_scale=0.05
        )
        
        # Initialize carry 
        sample_parameter = jnp.zeros((horizon, mdp.n_ctrl))
        self.carry = self.mppi.initial_carry(sample_parameter)
        
        logger.info(f"Seher MPPI Controller initialized:")
        logger.info(f"  Horizon: {horizon} steps")
        logger.info(f"  Candidates: {n_candidates}")
        logger.info(f"  Top-k: {top_k}")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Noise scale: {noise_scale}")
    
    def _vectorized_objective(self, controls: jnp.ndarray, problem_data, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Wrapper to match seher's objective interface"""
        state, _ = problem_data  # problem_data is (state, extra_info)
        return self.objective(controls, state, key)
    
    def get_control(self, state: HexapodState, key: jax.random.PRNGKey) -> tuple[jnp.ndarray, any]:
        """
        Get the next control action using MPPI
        
        Args:
            state: Current robot state
            key: Random key
            
        Returns:
            control: (n_ctrl,) control action
            carry: Updated MPPI carry
        """
        # Prepare problem data for seher interface
        problem_data = (state, None)
        
        # Run MPPI step
        key, subkey = jax.random.split(key)
        self.carry, best_solution, _ = self.mppi(
            carry=self.carry,
            problem_data=problem_data,
            key=subkey
        )
        
        # Extract first control from the plan
        control = self.carry.current[0]  # Take first timestep of the plan
        
        # Shift the plan for warm-starting next iteration
        # (this is a simple warm-start strategy)
        shifted_plan = jnp.concatenate([
            self.carry.current[1:],  # Remove first timestep
            self.carry.current[-1:],  # Repeat last timestep
        ], axis=0)
        
        self.carry = self.carry.replace(current=shifted_plan)
        
        return control, self.carry
    
    def reset(self, key: jax.random.PRNGKey):
        """Reset the controller with a new random plan"""
        # Generate a new random plan
        controls = jax.random.normal(key, (self.horizon, self.mdp.n_ctrl)) * self.noise_scale
        controls = jnp.clip(controls, self.mdp.control_min, self.mdp.control_max)
        
        self.carry = self.carry.replace(current=controls)
        
        logger.info("MPPI controller reset with new random plan")


async def test_mppi_controller():
    """Test function for the MPPI controller"""
    import asyncio
    
    # Create MDP
    mdp = HexapodMDP.create("hexagonal_robot.xml")
    key = jax.random.PRNGKey(42)
    
    # Initialize state
    key, subkey = jax.random.split(key)
    state = mdp.init(subkey)
    
    # Create MPPI controller
    controller = MPPIController(
        mdp=mdp,
        horizon=10,  # Shorter horizon for faster testing
        n_candidates=100,  # Fewer candidates for faster testing
        top_k=10
    )
    
    logger.info("Testing MPPI controller...")
    
    # Run for a few steps
    for step in range(5):
        logger.info(f"Step {step}")
        
        # Get control
        key, subkey = jax.random.split(key)
        control, carry = controller.get_control(state, subkey)
        
        logger.info(f"  Control norm: {jnp.linalg.norm(control):.3f}")
        logger.info(f"  Height: {state.qpos[2]:.3f}")
        
        # Step simulation
        key, subkey = jax.random.split(key)
        state = mdp.transit(state, control, subkey)
        
        # Add some delay to see the progression
        await asyncio.sleep(0.1)
    
    logger.info("MPPI test completed!")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_mppi_controller())