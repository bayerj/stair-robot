#!/usr/bin/env python3
"""Test MPPI with our hexapod MDP, based on seher's integration tests."""

import jax
import jax.numpy as jnp
import jax.random as jr

from seher.control.mpc import MPCPolicy
from seher.control.stepper_planner import StepperPlanner
from seher.simulate import simulate
from seher.stepper.mppi import GaussianMPPIOptimizer
from hexapod_mdp import HexapodMDP


def test_mppi_on_hexapod():
    """Test MPPI with our hexapod MDP - basic functionality test."""
    
    # Create our hexapod MDP
    mdp = HexapodMDP.create("hexagonal_robot.xml")
    print(f"MDP control dimension: {mdp.n_ctrl}")
    print(f"Control bounds: {mdp.control_min[0]:.3f} to {mdp.control_max[0]:.3f}")
    
    # Create MPPI planner with 2D controls (horizon, n_ctrl)
    # Following the pattern from the LQR test that works
    planner = StepperPlanner(
        mdp=mdp,
        n_iter=3,  # Small for testing
        n_plan_steps=5,  # Small horizon for testing
        warm_start=True,
        optimizer=GaussianMPPIOptimizer(
            objective=None,
            n_candidates=16,  # Small number for testing
            top_k=4,
            min_scale=0.01,
            temperature=1.0,
            # Key: Use 2D arrays like the LQR test
            initial_loc=jnp.zeros(mdp.n_ctrl),  # (n_ctrl,) - 1D per timestep
            initial_scale=jnp.ones(mdp.n_ctrl) * 0.1,  # (n_ctrl,) - 1D per timestep
        ),
    )
    
    # Create MPC policy
    policy = MPCPolicy(mdp=mdp, planner=planner)
    
    # Run simulation for a few steps
    key = jr.PRNGKey(42)
    n_steps = 5  # Very short test
    
    print(f"Running {n_steps} steps with MPPI...")
    
    history = simulate(
        policy=policy,
        mdp=mdp,
        key=key,
        n_steps=n_steps,
        initial_state=mdp.init(key),
    )
    
    print(f"Simulation completed!")
    print(f"Final costs: {history.costs}")
    print(f"Average cost: {history.costs.mean():.4f}")
    print(f"History states type: {type(history.states)}")
    print(f"History states shape: {history.states.qpos.shape}")
    
    # Get final state - states are batched over time
    final_height = history.states.qpos[-1, 2]  # Last timestep, z-coordinate
    print(f"Final height: {final_height:.3f}")
    
    # Basic success check - robot shouldn't fall immediately
    assert final_height > 0.02, "Robot fell too low"
    print("Test passed!")


if __name__ == "__main__":
    test_mppi_on_hexapod()