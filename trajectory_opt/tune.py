"""Hyperparameter tuning with Optuna."""

import functools

import jax.random as jr
import optuna
import plotext

from .config import (
    DEFAULT_DT,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_N_ROLLOUTS,
    OPTUNA_COST_COEFFICIENT_SEARCH_SPACE,
    OPTUNA_OPTIMIZER_SEARCH_SPACE,
    OPTUNA_STORAGE,
    OPTUNA_STUDY_NAME,
)
from .utils import (
    create_mdp,
    create_planner,
    run_optimization,
    save_trajectory,
)


def _suggest_parameters(trial: optuna.Trial) -> dict:
    """Suggest hyperparameters for a trial.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.

    Returns
    -------
    dict
        Dictionary with 'optimizer' and 'cost' parameter dicts.

    """
    optimizer_params = {}
    for name, spec in OPTUNA_OPTIMIZER_SEARCH_SPACE.items():
        if spec['type'] == 'int':
            optimizer_params[name] = trial.suggest_int(
                name,
                low=spec['low'],
                high=spec['high'],
                step=spec.get('step', 1),
            )
        elif spec['type'] == 'float':
            optimizer_params[name] = trial.suggest_float(
                name,
                low=spec['low'],
                high=spec['high'],
                log=spec.get('log', False),
                step=spec.get('step', None),
            )

    cost_coefficients = {}
    for name, spec in OPTUNA_COST_COEFFICIENT_SEARCH_SPACE.items():
        if spec['type'] == 'float':
            cost_coefficients[name] = trial.suggest_float(
                f"cost_{name}",
                low=spec['low'],
                high=spec['high'],
                log=spec.get('log', False),
                step=spec.get('step', None),
            )

    return {
        'optimizer': optimizer_params,
        'cost': cost_coefficients,
    }


def _objective(
    trial: optuna.Trial | None,
    episode_length: int,
    n_rollouts: int,
    key: jr.PRNGKey,
    parameters: dict | None = None,
):
    """Objective function for Optuna.

    Parameters
    ----------
    trial : optuna.Trial | None
        Optuna trial (None when running with fixed parameters).
    episode_length : int
        Number of timesteps to simulate.
    n_rollouts : int
        Number of rollouts to average over.
    key : jr.PRNGKey
        Random key.
    parameters : dict | None
        Fixed parameters (if not using trial suggestions).

    Returns
    -------
    float
        Average cost (to minimize).

    """
    # Get parameters
    if parameters is None:
        parameters = _suggest_parameters(trial)

    optimizer_params = parameters['optimizer']
    cost_coefficients = parameters['cost']

    # Print trial info
    if trial is not None:
        print(f"\nTrial {trial.number}:")
        print("Optimizer params:", optimizer_params)
        print("Cost coefficients:", cost_coefficients)

    # Create MDP and planner
    mdp = create_mdp(cost_coefficients, dt=DEFAULT_DT)
    planner = create_planner(mdp, optimizer_params)

    # Run optimization
    history, avg_cost, all_costs = run_optimization(
        mdp=mdp,
        planner=planner,
        episode_length=episode_length,
        n_rollouts=n_rollouts,
        key=key,
    )

    # Plot costs
    plotext.clear_figure()
    plotext.plot_size(width=79, height=15)
    for i in range(all_costs.shape[0]):
        plotext.plot(all_costs[i].flatten().tolist())
    trial_num = trial.number if trial else 'Test'
    plotext.title(f"Trial {trial_num}: Cost={avg_cost:.2f}")
    plotext.xlabel("Timestep")
    plotext.ylabel("Cost")
    print()
    plotext.show()

    # Save trajectory
    if trial is not None:
        output_path = f"trajectory_trial_{trial.number}.dill"
        save_trajectory(
            path=output_path,
            history=history,
            optimizer_params=optimizer_params,
            cost_coefficients=cost_coefficients,
            final_cost=avg_cost,
            cost_history=[avg_cost],  # Single iteration in tune mode
            dt=DEFAULT_DT,
        )

    return float(avg_cost)


def tune_search(
    *,
    n_trials: int = 100,
    episode_length: int = DEFAULT_EPISODE_LENGTH,
    n_rollouts: int = DEFAULT_N_ROLLOUTS,
    seed: int = 42,
):
    """Run hyperparameter search with Optuna.

    Parameters
    ----------
    n_trials : int
        Number of trials to run (default: 100).
    episode_length : int
        Number of timesteps per episode (default from config).
    n_rollouts : int
        Number of rollouts to average over (default from config).
    seed : int
        Random seed (default: 42).

    """
    print("=" * 79)
    print("Hyperparameter Tuning with Optuna")
    print("=" * 79)
    print()

    print(f"Study Name: {OPTUNA_STUDY_NAME}")
    print(f"Storage: {OPTUNA_STORAGE}")
    print(f"Number of Trials: {n_trials}")
    print(f"Episode Length: {episode_length}")
    print(f"Number of Rollouts: {n_rollouts}")
    print()

    # Create study
    study = optuna.create_study(
        storage=OPTUNA_STORAGE,
        study_name=OPTUNA_STUDY_NAME,
        load_if_exists=True,
        direction="minimize",
    )

    # Create objective with fixed parameters
    key = jr.PRNGKey(seed)
    objective = functools.partial(
        _objective,
        episode_length=episode_length,
        n_rollouts=n_rollouts,
        key=key,
    )

    # Run optimization
    print("Starting optimization...")
    print()

    try:
        study.optimize(objective, n_trials=n_trials)
    except KeyboardInterrupt:
        print()
        print("Optimization interrupted by user!")

    # Print results
    print()
    print("=" * 79)
    print("Optimization Complete!")
    print("=" * 79)
    print()
    print(f"Number of finished trials: {len(study.trials)}")
    print()
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (cost): {trial.value:.4f}")
    print()
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print()


def tune_run_best(
    *,
    episode_length: int = DEFAULT_EPISODE_LENGTH,
    n_rollouts: int = DEFAULT_N_ROLLOUTS,
    seed: int = 42,
    output: str = "trajectory_best.dill",
):
    """Run optimization with best parameters from Optuna study.

    Parameters
    ----------
    episode_length : int
        Number of timesteps per episode (default from config).
    n_rollouts : int
        Number of rollouts to average over (default from config).
    seed : int
        Random seed (default: 42).
    output : str
        Output dill file path (default: trajectory_best.dill).

    """
    print("=" * 79)
    print("Running Best Parameters from Optuna Study")
    print("=" * 79)
    print()

    # Load study
    print(f"Loading study: {OPTUNA_STUDY_NAME}")
    study = optuna.load_study(
        storage=OPTUNA_STORAGE,
        study_name=OPTUNA_STUDY_NAME,
    )

    print(f"Number of trials: {len(study.trials)}")
    print(f"Best value: {study.best_value:.4f}")
    print()

    # Extract best parameters
    best_params = study.best_params
    optimizer_params = {}
    cost_coefficients = {}

    for key, value in best_params.items():
        if key.startswith("cost_"):
            cost_coefficients[key[5:]] = value  # Remove "cost_" prefix
        else:
            optimizer_params[key] = value

    print("Best Optimizer Parameters:")
    for key, value in optimizer_params.items():
        print(f"  {key}: {value}")
    print()

    print("Best Cost Coefficients:")
    for key, value in cost_coefficients.items():
        print(f"  {key}: {value}")
    print()

    # Create MDP and planner
    print("Creating MDP and planner...")
    mdp = create_mdp(cost_coefficients, dt=DEFAULT_DT)
    planner = create_planner(mdp, optimizer_params)
    print("Done!")
    print()

    # Run optimization
    print("Running optimization with best parameters...")
    key = jr.PRNGKey(seed)
    history, avg_cost, _all_costs = run_optimization(
        mdp=mdp,
        planner=planner,
        episode_length=episode_length,
        n_rollouts=n_rollouts,
        key=key,
    )

    print(f"Final Cost: {avg_cost:.4f}")
    print()

    # Save trajectory
    save_trajectory(
        path=output,
        history=history,
        optimizer_params=optimizer_params,
        cost_coefficients=cost_coefficients,
        final_cost=avg_cost,
        cost_history=[avg_cost],
        dt=DEFAULT_DT,
    )

    print()
    print("Done!")
    print(f"To visualize: python -m trajectory_opt playback {output}")


def tune_show_best():
    """Show best parameters from Optuna study."""
    print("=" * 79)
    print("Best Parameters from Optuna Study")
    print("=" * 79)
    print()

    # Load study
    print(f"Loading study: {OPTUNA_STUDY_NAME}")
    study = optuna.load_study(
        storage=OPTUNA_STORAGE,
        study_name=OPTUNA_STUDY_NAME,
    )

    print(f"Number of trials: {len(study.trials)}")
    print()

    # Show best trial
    trial = study.best_trial
    print(f"Best trial: {trial.number}")
    print(f"Best value (cost): {trial.value:.4f}")
    print()

    # Separate parameters
    optimizer_params = {}
    cost_coefficients = {}

    for key, value in trial.params.items():
        if key.startswith("cost_"):
            cost_coefficients[key[5:]] = value
        else:
            optimizer_params[key] = value

    print("Optimizer Parameters:")
    for key, value in optimizer_params.items():
        print(f"  {key}: {value}")
    print()

    print("Cost Coefficients:")
    for key, value in cost_coefficients.items():
        print(f"  {key}: {value}")
    print()
