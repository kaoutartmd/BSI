"""Configuration file for experiments."""

# Environment configurations
ENV_CONFIGS = {
    'harvest': {
        'grid_size': (25, 25),
        'num_agents': 5,
        'view_size': 7,
        'apple_density': 0.3,
        'regrow_prob': 0.01
    },
    'cleanup': {
        'grid_size': (25, 25),
        'num_agents': 5,
        'view_size': 7,
        'apple_density': 0.3,
        'regrow_prob': 0.01,
        'river_width': 3,
        'waste_spawn_prob': 0.5
    }
}

# Training configurations
TRAINING_CONFIG = {
    'num_episodes': 10000,
    'max_episode_length': 1000,
    'checkpoint_interval': 1000,
    'log_interval': 100
}

# Model configurations
MODEL_CONFIG = {
    'hidden_dim': 128,
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'entropy_coef': 0.01,
    'value_loss_coef': 0.5,
    'max_grad_norm': 0.5
}

# Hyperparameter search space
HYPERPARAM_SEARCH = {
    'influence_weight': [0.01, 0.05, 0.1, 0.25],
    'curriculum_steps': [0, 2e7, 3.5e8]
}

# Experiment configurations
EXPERIMENT_CONFIG = {
    'num_seeds': 5,
    'environments': ['harvest', 'cleanup'],
    'models': ['standard_a3c', 'visible_actions', 'influence']
}