"""Trajectory optimization command."""

import jax.random as jr
import plotext
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from .config import (
    DEFAULT_COST_COEFFICIENTS,
    DEFAULT_DT,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_N_ROLLOUTS,
    DEFAULT_OPTIMIZER_PARAMS,
    MINIMAL_EPISODE_LENGTH,
    MINIMAL_N_ROLLOUTS,
    MINIMAL_OPTIMIZER_PARAMS,
)
from .utils import (
    create_mdp,
    extract_plan_from_trajectory,
    run_optimization_iterative,
    save_trajectory,
)


def optimize(
    *,
    output: str = "trajectory.dill",
    episode_length: int = DEFAULT_EPISODE_LENGTH,
    n_rollouts: int = DEFAULT_N_ROLLOUTS,
    seed: int = 42,
    minimal: bool = False,
    from_trajectory: str = "",
):
    """Optimize a trajectory using ARS and save to file.

    Parameters
    ----------
    output : str
        Output dill file path (default: trajectory.dill).
    episode_length : int
        Number of timesteps to simulate (default from config).
    n_rollouts : int
        Number of rollouts to average over (default from config).
    seed : int
        Random seed (default: 42).
    minimal : bool
        Use minimal settings for quick testing (default: False).
    from_trajectory : str
        Path to previous trajectory file to warm-start from. If provided,
        optimization will continue from the control plan in that file
        (default: empty string = start from scratch).

    """
    # Use minimal settings if requested
    if minimal:
        optimizer_params = MINIMAL_OPTIMIZER_PARAMS
        if episode_length == DEFAULT_EPISODE_LENGTH:
            episode_length = MINIMAL_EPISODE_LENGTH
        if n_rollouts == DEFAULT_N_ROLLOUTS:
            n_rollouts = MINIMAL_N_ROLLOUTS
    else:
        optimizer_params = DEFAULT_OPTIMIZER_PARAMS
    print("=" * 79)
    print("Trajectory Optimization with ARS")
    print("=" * 79)
    print()

    # Print configuration
    if minimal:
        print("*** MINIMAL TEST MODE ***")
        print()

    print("Optimizer Parameters:")
    for key, value in optimizer_params.items():
        print(f"  {key}: {value}")
    print()

    print("Cost Coefficients:")
    for key, value in DEFAULT_COST_COEFFICIENTS.items():
        print(f"  {key}: {value}")
    print()

    print(f"Episode Length: {episode_length}")
    print(f"Number of Rollouts: {n_rollouts}")
    print(f"Random Seed: {seed}")
    print()

    # Load initial plan if specified
    initial_plan = None
    if from_trajectory:
        print(f"Loading initial plan from: {from_trajectory}")
        try:
            initial_plan = extract_plan_from_trajectory(from_trajectory)
            print(f"Loaded plan with shape: {initial_plan.shape}")
            print()
        except Exception as e:
            print(f"Error loading trajectory: {e}")
            print("Starting from scratch instead.")
            print()

    # Create MDP
    print("Creating MDP...")
    mdp = create_mdp(DEFAULT_COST_COEFFICIENTS, dt=DEFAULT_DT)
    print("Done!")
    print()

    # Initialize random key
    key = jr.PRNGKey(seed)

    # Run optimization with progress bar
    print("Running optimization...")
    print()

    # Setup plotext for live plotting
    plotext.clear_figure()
    plotext.plot_size(width=79, height=15)

    # Track iteration costs for plotting
    iteration_costs = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn("Cost: {task.fields[cost]:.2f}"),
    ) as progress:
        task = progress.add_task(
            "Optimizing",
            total=optimizer_params['n_iter'],
            cost=0.0,
        )

        # Progress callback for each iteration
        def progress_callback(iteration, cost):
            progress.update(task, advance=1, cost=cost)
            iteration_costs.append(cost)

            # Update plot every few iterations
            if (iteration + 1) % max(1, optimizer_params['n_iter'] // 10) == 0:
                plotext.clear_figure()
                plotext.plot_size(width=79, height=15)
                plotext.plot(iteration_costs)
                plotext.title("Cost per Optimizer Iteration")
                plotext.xlabel("Iteration")
                plotext.ylabel("Cost")
                print()
                plotext.show()

        try:
            key, subkey = jr.split(key)
            history, avg_cost, all_costs, _ = (
                run_optimization_iterative(
                    mdp=mdp,
                    optimizer_params=optimizer_params,
                    episode_length=episode_length,
                    n_rollouts=n_rollouts,
                    key=subkey,
                    progress_callback=progress_callback,
                    initial_plan=initial_plan,
                )
            )

            # Final plot
            plotext.clear_figure()
            plotext.plot_size(width=79, height=15)
            plotext.plot(iteration_costs)
            plotext.title("Cost per Optimizer Iteration")
            plotext.xlabel("Iteration")
            plotext.ylabel("Cost")
            print()
            plotext.show()

        except KeyboardInterrupt:
            print()
            print("Optimization interrupted by user!")
            print()
            return

    # Save trajectory
    print()
    print(f"Final Cost: {avg_cost:.4f}")
    print()

    save_trajectory(
        path=output,
        history=history,
        optimizer_params=optimizer_params,
        cost_coefficients=DEFAULT_COST_COEFFICIENTS,
        final_cost=avg_cost,
        cost_history=iteration_costs,
        dt=DEFAULT_DT,
    )

    print()
    print("Optimization complete!")
    print(f"To visualize: python -m trajectory_opt playback {output}")
