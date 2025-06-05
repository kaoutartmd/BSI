import wandb
import numpy as np
from typing import Dict, List, Optional, Any
import os


class WandBLogger:
    """Centralized WandB logging for experiments."""
    
    def __init__(self, 
                 project_name: str = "social-influence-marl",
                 entity: Optional[str] = None,
                 config: Optional[Dict] = None,
                 name: Optional[str] = None,
                 group: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 mode: str = "online"):
        """
        Initialize WandB logger.
        
        Args:
            project_name: WandB project name
            entity: WandB entity (username or team)
            config: Experiment configuration
            name: Run name
            group: Group name for organizing runs
            tags: List of tags for the run
            mode: "online", "offline", or "disabled"
        """
        self.project_name = project_name
        self.entity = entity
        
        # Initialize WandB
        self.run = wandb.init(
            project=project_name,
            entity=entity,
            config=config,
            name=name,
            group=group,
            tags=tags,
            mode=mode
        )
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to WandB."""
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    
    def log_episode(self, episode: int, episode_metrics: Dict[str, Any]):
        """Log episode-level metrics."""
        # Add episode prefix to metrics
        episode_data = {f"episode/{k}": v for k, v in episode_metrics.items()}
        episode_data["episode/number"] = episode
        
        self.log_metrics(episode_data, step=episode)
    
    def log_training_step(self, step: int, step_metrics: Dict[str, Any]):
        """Log training step metrics."""
        # Add training prefix
        train_data = {f"train/{k}": v for k, v in step_metrics.items()}
        train_data["train/step"] = step
        
        self.log_metrics(train_data, step=step)
    
    def log_evaluation(self, eval_metrics: Dict[str, Any], step: int):
        """Log evaluation metrics."""
        eval_data = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.log_metrics(eval_data, step=step)
    
    def log_model_gradients(self, model, step: int):
        """Log model gradients for debugging."""
        wandb.watch(model, log_freq=100)
    
    def save_model(self, model_path: str, metadata: Optional[Dict] = None):
        """Save model artifact to WandB."""
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            metadata=metadata
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
    
    def create_histogram(self, data: np.ndarray, name: str, step: int):
        """Create histogram for data distribution."""
        wandb.log({name: wandb.Histogram(data)}, step=step)
    
    def create_plot(self, figure, name: str, step: int):
        """Log matplotlib figure."""
        wandb.log({name: wandb.Image(figure)}, step=step)
    
    def finish(self):
        """Finish the WandB run."""
        wandb.finish()


