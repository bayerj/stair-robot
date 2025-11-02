"""Shared utilities for trajectory optimization."""

import functools
import sys
from pathlib import Path

import dill
import jax.random as jr
import optax
from seher.ars import ars_value_and_grad
from seher.control.mpc import MPCPolicy
from seher.control.stepper_planner import StepperPlanner
from seher.simulate import simulate
from seher.stepper.optax import OptaxOptimizer

# Add parent directory to path to import hexapod_mdp
sys.path.insert(0, str(Path(__file__).parent.parent))
from hexapod_mdp import HexapodMDP


def create_mdp(cost_coefficients: dict, dt: float = 0.05) -> HexapodMDP:
    """Create HexapodMDP with specified cost coefficients.

    Parameters
    ----------
    cost_coefficients : dict
        Cost function coefficients to use.
    dt : float
        Timestep for control (default: 0.05s = 20Hz).

    Returns
    -------
    HexapodMDP
        Configured MDP instance.

    """
    mdp = HexapodMDP.create("hexagonal_robot.xml", dt=dt)
    mdp = mdp.replace(cost_coefficients=cost_coefficients)
    return mdp


def create_ars_optimizer(optimizer_params: dict) -> OptaxOptimizer:
    """Create ARS-based optimizer with specified parameters.

    Parameters
    ----------
    optimizer_params : dict
        Must contain: learning_rate, std, n_perturbations, top_k_ratio

    Returns
    -------
    OptaxOptimizer
        Configured optimizer instance.

    """
    top_k = max(
        1,
        int(
            optimizer_params['top_k_ratio']
            * optimizer_params['n_perturbations']
        ),
    )

    optimizer = OptaxOptimizer(
        objective=None,  # Set by StepperPlanner
        optimizer=optax.adam(optimizer_params['learning_rate']),
        value_and_grad=functools.partial(
            ars_value_and_grad,
            std=optimizer_params['std'],
            n_perturbations=optimizer_params['n_perturbations'],
            top_k=top_k,
        ),
    )

    return optimizer


def create_planner(
    mdp: HexapodMDP, optimizer_params: dict, episode_length: int
) -> StepperPlanner:
    """Create StepperPlanner with ARS optimizer.

    Parameters
    ----------
    mdp : HexapodMDP
        MDP instance to plan for.
    optimizer_params : dict
        Must contain: n_iter and ARS parameters.
    episode_length : int
        Number of steps to plan for.

    Returns
    -------
    StepperPlanner
        Configured planner instance.

    """
    optimizer = create_ars_optimizer(optimizer_params)

    planner = StepperPlanner(
        mdp=mdp,
        n_iter=optimizer_params['n_iter'],
        n_plan_steps=episode_length,
        warm_start=False,  # No warm start as requested
        optimizer=optimizer,
    )

    return planner


def run_optimization_iterative(
    mdp: HexapodMDP,
    optimizer_params: dict,
    episode_length: int,
    n_rollouts: int,
    key: jr.PRNGKey,
    progress_callback=None,
    initial_plan=None,
):
    """Run trajectory optimization with manual iteration loop.

    This allows progress updates after each optimizer iteration by
    compiling only a single optimization step and looping in Python.

    Parameters
    ----------
    mdp : HexapodMDP
        MDP instance.
    optimizer_params : dict
        Optimizer parameters including n_iter.
    episode_length : int
        Number of timesteps to simulate.
    n_rollouts : int
        Number of rollouts to average over.
    key : jr.PRNGKey
        Random key for simulation.
    progress_callback : callable, optional
        Called with (iteration, cost) after each optimizer iteration.
    initial_plan : jnp.ndarray, optional
        Initial control plan to warm-start from. If provided, must have
        shape [episode_length, n_ctrl]. If None, starts from empty
        controls.

    Returns
    -------
    history : History
        Simulation history.
    avg_cost : float
        Average cost.
    all_costs : jnp.ndarray
        Cost array.
    iteration_costs : list[float]
        Cost after each optimizer iteration.

    """
    import jax
    import jax.numpy as jnp
    from seher.control.stepper_planner import calc_costs_of_plan

    n_total_iters = optimizer_params['n_iter']

    # Create planner with n_iter=1 for single-step optimization
    single_iter_params = optimizer_params.copy()
    single_iter_params['n_iter'] = 1
    planner = create_planner(mdp, single_iter_params, episode_length)

    # Initialize optimizer carry
    # Create initial plan as 2D array [episode_length, n_ctrl]
    from seher.jax_util import tree_stack

    if initial_plan is None:
        initial_plan = tree_stack(
            [mdp.empty_control() for _ in range(episode_length)]
        )
    else:
        # Ensure initial_plan has correct shape
        if initial_plan.shape[0] != episode_length:
            raise ValueError(
                f"Initial plan has wrong length: {initial_plan.shape[0]} "
                f"!= {episode_length}"
            )
        # Convert to JAX array if needed
        initial_plan = jnp.asarray(initial_plan)

    opt_carry = planner.optimizer.initial_carry(sample_parameter=initial_plan)

    # JIT compile single optimizer step
    def single_opt_step(carry, state, step_key):
        """Run one optimizer iteration."""
        # Create objective function (must match ARS signature: parameter, problem_data, key)
        def objective(parameter, problem_data, key):
            plan = planner.decode_plan(parameter)
            cost = calc_costs_of_plan(mdp, plan, problem_data, key)
            return cost, None

        # Update optimizer with objective
        optimizer = planner.optimizer.replace(objective=objective)

        # Single optimization step
        new_carry, new_params, _ = optimizer(
            carry=carry,
            problem_data=state,
            key=step_key,
        )

        # Calculate cost for reporting
        plan = planner.decode_plan(new_params)
        cost = calc_costs_of_plan(mdp, plan, state, step_key)

        return new_carry, cost

    jit_opt_step = jax.jit(single_opt_step)

    # Initialize state
    key, subkey = jr.split(key)
    state = mdp.init(subkey)

    # Manual optimization loop
    iteration_costs = []
    for iteration in range(n_total_iters):
        key, subkey = jr.split(key)

        # Single optimization step (JIT'd)
        opt_carry, iter_cost = jit_opt_step(opt_carry, state, subkey)
        iteration_costs.append(float(iter_cost))

        # Progress callback
        if progress_callback:
            progress_callback(iteration, float(iter_cost))

    # Extract the optimized plan and run final evaluation
    from seher.control.basic_policies import OpenLoopPolicy

    optimized_plan = planner.decode_plan(opt_carry.current)

    # Evaluate the optimized plan to get trajectory history
    # This creates the history needed for playback
    if n_rollouts == 1:
        key, subkey = jr.split(key)
        policy = OpenLoopPolicy(plan=optimized_plan)
        history = simulate(
            mdp=mdp,
            policy=policy,
            n_steps=episode_length,
            key=subkey,
            initial_state=state,
        )
        avg_cost = float(history.costs.sum())
        all_costs = history.costs.reshape(1, -1)
    else:
        # For multiple rollouts, evaluate the plan multiple times
        def single_rollout(key):
            policy = OpenLoopPolicy(plan=optimized_plan)
            return simulate(
                mdp=mdp,
                policy=policy,
                n_steps=episode_length,
                key=key,
                initial_state=state,
            )

        histories = jax.tree.map(
            lambda *xs: jnp.stack(xs),
            *[single_rollout(k) for k in jr.split(key, n_rollouts)]
        )
        history = jax.tree.map(lambda leaf: leaf[0], histories)
        avg_cost = float(histories.costs.sum(axis=1).mean())
        all_costs = histories.costs

    return history, avg_cost, all_costs, iteration_costs


def run_optimization(
    mdp: HexapodMDP,
    optimizer_params: dict,
    episode_length: int,
    n_rollouts: int,
    key: jr.PRNGKey,
):
    """Run trajectory optimization and return history.

    Parameters
    ----------
    mdp : HexapodMDP
        MDP instance.
    optimizer_params : dict
        Optimizer parameters.
    episode_length : int
        Number of timesteps to simulate.
    n_rollouts : int
        Number of rollouts to average over.
    key : jr.PRNGKey
        Random key for simulation.

    Returns
    -------
    history : History
        Simulation history (single rollout if n_rollouts=1, otherwise
        first rollout).
    avg_cost : float
        Average cost across all rollouts.
    all_costs : jnp.ndarray
        Cost array for all rollouts (shape: [n_rollouts, episode_length]).

    """
    import jax

    planner = create_planner(mdp, optimizer_params, episode_length)
    policy = MPCPolicy(mdp=mdp, planner=planner)

    if n_rollouts == 1:
        # Single rollout with JIT
        jit_simulate = jax.jit(
            simulate,
            static_argnums=(0, 1, 2),
        )
        history = jit_simulate(
            mdp,
            policy,
            episode_length,
            key,
        )
        avg_cost = float(history.costs.sum())
        all_costs = history.costs.reshape(1, -1)
    else:
        # Multiple rollouts (batched)
        batch_simulate = jax.jit(
            jax.vmap(simulate, in_axes=(None, None, None, 0)),
            static_argnums=(0, 1, 2),
        )
        histories = batch_simulate(
            mdp,
            policy,
            episode_length,
            jr.split(key, n_rollouts),
        )

        # Return first rollout as representative history
        history = jax.tree.map(lambda leaf: leaf[0], histories)
        avg_cost = float(histories.costs.sum(axis=1).mean())
        all_costs = histories.costs

    return history, avg_cost, all_costs


def save_trajectory(
    path: str,
    history,
    optimizer_params: dict,
    cost_coefficients: dict,
    final_cost: float,
    cost_history: list[float],
    dt: float,
):
    """Save optimized trajectory to dill file.

    Parameters
    ----------
    path : str
        Output file path.
    history : History
        Simulation history from seher.
    optimizer_params : dict
        Optimizer parameters used.
    cost_coefficients : dict
        Cost coefficients used.
    final_cost : float
        Final total cost.
    cost_history : list[float]
        Cost per optimization iteration.
    dt : float
        Timestep used during optimization.

    """
    data = {
        'history': history,
        'optimizer_params': optimizer_params,
        'cost_coefficients': cost_coefficients,
        'final_cost': final_cost,
        'cost_history': cost_history,
        'dt': dt,
    }

    with open(path, 'wb') as f:
        dill.dump(data, f)

    print(f"Trajectory saved to {path}")


def load_trajectory(path: str) -> dict:
    """Load trajectory from dill file.

    Parameters
    ----------
    path : str
        Input file path.

    Returns
    -------
    dict
        Dictionary containing history, optimizer_params, cost_coefficients,
        final_cost, and cost_history.

    """
    with open(path, 'rb') as f:
        data = dill.load(f)

    return data


def extract_plan_from_trajectory(path: str):
    """Extract optimized control plan from trajectory file.

    Parameters
    ----------
    path : str
        Path to trajectory dill file.

    Returns
    -------
    jnp.ndarray
        Control plan array of shape [episode_length, n_ctrl].

    """
    import jax.numpy as jnp

    data = load_trajectory(path)
    history = data['history']

    # Extract controls from history
    # History.controls has shape [episode_length, n_ctrl]
    return jnp.array(history.controls)
