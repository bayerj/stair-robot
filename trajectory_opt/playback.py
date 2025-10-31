"""Trajectory playback command."""

import time

import mujoco
import mujoco.viewer
import numpy as np

from .utils import create_mdp, load_trajectory


def playback(
    trajectory_path: str,
    *,
    speed: float = 1.0,
    loop: bool = True,
):
    """Playback trajectory in MuJoCo viewer.

    Parameters
    ----------
    trajectory_path : str
        Path to trajectory dill file.
    speed : float
        Playback speed multiplier (default: 1.0).
    loop : bool
        Loop playback continuously (default: True).

    """
    print("=" * 79)
    print("Trajectory Playback")
    print("=" * 79)
    print()

    # Load trajectory
    print(f"Loading trajectory from {trajectory_path}...")
    data = load_trajectory(trajectory_path)
    history = data['history']
    optimizer_params = data['optimizer_params']
    cost_coefficients = data['cost_coefficients']
    final_cost = data['final_cost']
    _cost_history = data['cost_history']  # Not used in playback
    dt = data.get('dt', 0.05)  # Default to 0.05 if not in file

    print("Done!")
    print()

    # Print summary
    print("Optimizer Parameters:")
    for key, value in optimizer_params.items():
        print(f"  {key}: {value}")
    print()

    print("Cost Coefficients:")
    for key, value in cost_coefficients.items():
        print(f"  {key}: {value}")
    print()

    # Get number of timesteps from the qpos shape
    n_timesteps = history.states.qpos.shape[0]

    print(f"Final Cost: {final_cost:.4f}")
    print(f"Number of Timesteps: {n_timesteps}")
    print()

    # Get cost breakdown for each timestep
    print("Computing cost breakdown...")
    mdp = create_mdp(cost_coefficients, dt=dt)

    # Use a dummy key since cost functions shouldn't use randomness
    import jax.random as jr
    key = jr.PRNGKey(0)

    # Calculate per-timestep costs and breakdown
    timestep_costs = []
    cost_breakdowns = []

    for i in range(n_timesteps - 1):
        # Extract state at timestep i
        from hexapod_mdp import HexapodState
        state = HexapodState(
            qpos=history.states.qpos[i],
            qvel=history.states.qvel[i],
        )
        control = history.controls[i]

        cost = mdp.cost(state, control, key)
        breakdown = mdp.get_cost_breakdown(state, control, key)

        timestep_costs.append(float(cost))
        cost_breakdowns.append(breakdown)

    # Print average cost breakdown
    print("\nAverage Cost Breakdown:")
    if cost_breakdowns:
        # Extract weighted costs for each component
        avg_breakdown = {}
        for key in cost_breakdowns[0].keys():
            if key == 'total_cost':
                continue
            # Get weighted_cost values across all timesteps
            values = [
                float(bd[key]['weighted_cost']) for bd in cost_breakdowns
            ]
            avg_breakdown[key] = np.mean(values)

        total = sum(avg_breakdown.values())
        for key, value in sorted(
            avg_breakdown.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        ):
            percentage = (value / total * 100) if total != 0 else 0
            print(f"  {key:20s}: {value:8.4f} ({percentage:5.1f}%)")

        # Print total
        total_costs = [float(bd['total_cost']) for bd in cost_breakdowns]
        avg_total = np.mean(total_costs)
        print(f"  {'---':20s}   {'-' * 8}   {'-' * 7}")
        print(f"  {'TOTAL':20s}: {avg_total:8.4f}")
    print()

    # Load MuJoCo model for visualization
    print("Loading MuJoCo model for visualization...")
    model = mujoco.MjModel.from_xml_path(mdp.xml_path)
    mj_data = mujoco.MjData(model)
    print("Done!")
    print()

    print("Starting visualization...")
    print(f"Playback speed: {speed}x")
    print(f"Loop: {'Yes' if loop else 'No'}")
    print()
    print("Close the viewer window to exit.")
    print()

    # Playback loop
    try:
        try:
            viewer_ctx = mujoco.viewer.launch_passive(model, mj_data)
        except RuntimeError as e:
            if "mjpython" in str(e):
                print()
                print("ERROR: MuJoCo viewer requires mjpython on macOS")
                print()
                print("Please run with:")
                print(
                    f"  mjpython -m trajectory_opt playback "
                    f"{trajectory_path}"
                )
                print()
                return
            else:
                raise

        with viewer_ctx as viewer:
            while viewer.is_running():
                # Iterate through trajectory states
                for i in range(n_timesteps):
                    if not viewer.is_running():
                        break

                    # Set MuJoCo state from MJX state
                    mj_data.qpos[:] = np.array(history.states.qpos[i])
                    mj_data.qvel[:] = np.array(history.states.qvel[i])
                    mj_data.time = float(i * dt)

                    # Forward kinematics (no dynamics step)
                    mujoco.mj_forward(model, mj_data)

                    # Sync viewer
                    viewer.sync()

                    # Sleep according to playback speed
                    time.sleep(dt / speed)

                # Exit if not looping
                if not loop:
                    break

    except KeyboardInterrupt:
        print()
        print("Playback interrupted by user!")

    print()
    print("Playback complete!")
