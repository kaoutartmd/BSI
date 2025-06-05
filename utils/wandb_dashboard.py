import wandb
import pandas as pd
from typing import Optional

def create_comparison_report(project_name: str, entity: Optional[str] = None):
    """Create a comparison report for different models."""
    api = wandb.Api()
    
    # Get all runs from project
    runs = api.runs(f"{entity}/{project_name}" if entity else project_name)
    
    # Collect data
    summary_data = []
    for run in runs:
        summary_data.append({
            'name': run.name,
            'model_type': run.config.get('name', 'unknown'),
            'environment': run.config.get('environment', 'unknown'),
            'final_reward': run.summary.get('episode/collective_reward', 0),
            'convergence_episode': run.summary.get('convergence_episode', 0),
            'influence_weight': run.config.get('influence_weight', 0),
            'seed': run.config.get('seed', 0)
        })
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    # Create visualizations
    # ... add your custom plots ...
    
    return df