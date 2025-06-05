import wandb
import yaml

# Define sweep configuration
sweep_config = {
    'method': 'bayes',  # or 'grid', 'random'
    'name': 'influence-weight-sweep',
    'metric': {
        'name': 'eval/final_collective_reward',
        'goal': 'maximize'
    },
    'parameters': {
        'influence_weight': {
            'values': [0.01, 0.05, 0.1, 0.25, 0.5]
        },
        'curriculum_steps': {
            'values': [0, 1e7, 2e7, 3.5e8]
        },
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': 1e-5,
            'max': 1e-3
        },
        'hidden_dim': {
            'values': [64, 128, 256]
        }
    }
}

def run_sweep():
    """Run WandB sweep."""
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="social-influence-marl")
    
    # Run sweep agent
    wandb.agent(sweep_id, function=train_sweep_run, count=20)

def train_sweep_run():
    """Single training run for sweep."""
    # Initialize wandb
    run = wandb.init()
    
    # Get hyperparameters from sweep
    config = wandb.config
    
    # Run experiment with sweep parameters
    # ... your training code here ...
    
    wandb.finish()

if __name__ == "__main__":
    run_sweep()
