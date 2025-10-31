"""Configuration for trajectory optimization."""

# Default cost function coefficients
DEFAULT_COST_COEFFICIENTS = {
    'forward_velocity': 50.0,  # Increased: prioritize forward motion
    'height': 5.0,
    'action_penalty': 0.1,
    'stability': 5.0,
    'fall_penalty': 100.0,
    'joint_deviation': 0.1,  # Decreased: allow joint movement for walking
    'orientation': 2.0,
}

# Default optimizer parameters for ARS
DEFAULT_OPTIMIZER_PARAMS = {
    'n_iter': 10,
    'learning_rate': 0.1,
    'std': 0.5,
    'n_perturbations': 8,
    'top_k_ratio': 0.5,
}

# Minimal test configuration for quick testing
MINIMAL_OPTIMIZER_PARAMS = {
    'n_iter': 5,  # Increased from 2
    'learning_rate': 0.1,
    'std': 0.5,
    'n_perturbations': 4,  # Increased from 2
    'top_k_ratio': 0.5,
}

# Default optimization settings
DEFAULT_EPISODE_LENGTH = 500
DEFAULT_N_ROLLOUTS = 1
DEFAULT_DT = 0.05  # 20 Hz control frequency

# Minimal test settings
MINIMAL_EPISODE_LENGTH = 50
MINIMAL_N_ROLLOUTS = 1

# Optuna hyperparameter search spaces
OPTUNA_OPTIMIZER_SEARCH_SPACE = {
    'n_iter': {'type': 'int', 'low': 5, 'high': 20, 'step': 1},
    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.5, 'log': True},
    'std': {'type': 'float', 'low': 0.1, 'high': 2.0, 'step': 0.1},
    'n_perturbations': {'type': 'int', 'low': 4, 'high': 16, 'step': 2},
    'top_k_ratio': {'type': 'float', 'low': 0.1, 'high': 0.8},
}

OPTUNA_COST_COEFFICIENT_SEARCH_SPACE = {
    'forward_velocity': {'type': 'float', 'low': 0.0, 'high': 20.0},
    'height': {'type': 'float', 'low': 0.0, 'high': 10.0},
    'action_penalty': {'type': 'float', 'low': 0.0, 'high': 1.0},
    'stability': {'type': 'float', 'low': 0.0, 'high': 10.0},
    'fall_penalty': {'type': 'float', 'low': 0.0, 'high': 100.0},
    'joint_deviation': {'type': 'float', 'low': 0.0, 'high': 5.0},
    'orientation': {'type': 'float', 'low': 0.0, 'high': 5.0},
}

# Optuna database settings
OPTUNA_STORAGE = "sqlite:///hexapod_trajectory_optimization.db"
OPTUNA_STUDY_NAME = "hexapod-trajectory-ars"
