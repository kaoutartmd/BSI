import os
import json
import numpy as np
import torch
import random
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import wandb

from environments import HarvestEnvironment, CleanupEnvironment
from models import EnhancedInfluenceA3C, VisibleActionsBaseline
from utils import MetricsTracker
from models.attention_influence_a3c import AttentionInfluenceA3C

class ExperimentRunner:
    """Main experiment runner with logging and checkpointing."""
    
    def __init__(self, 
                 env_name: str = "Harvest",
                 num_agents: int = 5,
                 num_episodes: int = 10000,
                 log_dir: str = "experiments",
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 use_wandb: bool = False,
                 wandb_project: str = "social-influence-marl",
                 wandb_entity: Optional[str] = None):
        
        self.env_name = env_name
        self.num_agents = num_agents
        self.num_episodes = num_episodes
        self.log_dir = log_dir
        self.device = device
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(log_dir, f"{env_name}_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Hyperparameter configurations
        self.configs = self._load_configs()
    
    def _load_configs(self) -> Dict:
        """Load experiment configurations."""
        return {
            'standard_a3c': {
                'influence_weight': 0.0,
                'curriculum_steps': 0,
                'visible_actions': False
            },
            'visible_actions': {
                'influence_weight': 0.0,
                'curriculum_steps': 0,
                'visible_actions': True
            },
            'attention_influence': {
                'influence_weight': [0.01, 0.05, 0.1, 0.25],
                'curriculum_steps': [0, 2e7, 3.5e8],
                'visible_actions': False,
                'use_attention': True
            }
        }
    
    def create_environment(self):
        """Create environment instance."""
        if self.env_name.lower() == "harvest":
            return HarvestEnvironment(num_agents=self.num_agents)
        elif self.env_name.lower() == "cleanup":
            return CleanupEnvironment(num_agents=self.num_agents)
        else:
            raise ValueError(f"Unknown environment: {self.env_name}")
    
    def run_single_experiment(self, 
                            config: Dict, 
                            seed: int,
                            save_checkpoints: bool = True) -> Dict:
        """Run single experiment with given configuration."""
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        # Initialize WandB if enabled
        if self.use_wandb:
            wandb_config = {
                "environment": self.env_name,
                "num_agents": self.num_agents,
                "num_episodes": self.num_episodes,
                "seed": seed,
                **config
            }
            
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                config=wandb_config,
                name=f"{self.env_name}_{config['name']}_seed{seed}",
                group=f"{self.env_name}_{config['name']}",
                tags=[self.env_name, config['name'], f"seed{seed}"],
                reinit=True
            )
        
        # Create environment
        env = self.create_environment()
        
        # Determine state dimension
        obs_shape = env.observation_space.shape
        state_dim = np.prod(obs_shape)
        
        # Create model based on configuration
        if config.get('visible_actions', False) and config.get('influence_weight', 0) == 0:
            model = VisibleActionsBaseline(
                env=env,
                num_agents=self.num_agents,
                state_dim=state_dim,
                action_dim=env.action_space.n,
                device=self.device
            )
        else:
            model = AttentionInfluenceA3C(
                env=env,
                num_agents=self.num_agents,
                state_dim=state_dim,
                action_dim=env.action_space.n,
                influence_weight=config.get('influence_weight', 0.0),
                curriculum_steps=config.get('curriculum_steps', 0),
                device=self.device
            )
        
        # Metrics tracker
        metrics = MetricsTracker()
        
        # Training loop
        results = {
            'config': config,
            'seed': seed,
            'collective_rewards': [],
            'episode_lengths': [],
            'influence_weights': []
        }
        
        pbar = tqdm(range(self.num_episodes), desc=f"Training {config['name']}")
        
        for episode in pbar:
            reward = model.train_episode()
            
            # Update metrics
            metrics.add_scalar('collective_reward', reward)
            episode_length = len(model.metrics['episode_lengths']) if model.metrics['episode_lengths'] else 0
            metrics.add_scalar('episode_length', episode_length)
            
            # Log to WandB if enabled
            if self.use_wandb:
                log_data = {
                    'episode/collective_reward': reward,
                    'episode/influence_weight': model.get_current_influence_weight(),
                }
                
                # Add training metrics if available
                if model.metrics['losses']:
                    recent_losses = list(model.metrics['losses'])[-10:]
                    if recent_losses:
                        log_data['train/loss'] = np.mean(recent_losses)
                
                # Add individual agent rewards if available
                if hasattr(model, 'last_episode_rewards'):
                    for i, r in enumerate(model.last_episode_rewards):
                        log_data[f'agent/reward_{i}'] = r
                
                wandb.log(log_data, step=model.step_count)
            
            # Log progress every 100 episodes
            if episode % 100 == 0:
                avg_reward = metrics.get_average('collective_reward', last_n=100)
                current_influence = model.get_current_influence_weight()
                
                results['collective_rewards'].append(avg_reward)
                results['influence_weights'].append(current_influence)
                
                pbar.set_postfix({
                    'avg_reward': f"{avg_reward:.2f}",
                    'influence_w': f"{current_influence:.3f}"
                })
            
            # Save checkpoint
            if save_checkpoints and episode % 1000 == 0 and episode > 0:
                checkpoint_path = os.path.join(
                    self.exp_dir,
                    f"{config['name']}_seed{seed}_ep{episode}.pt"
                )
                model.save_checkpoint(checkpoint_path)
                
                # Log model to WandB if enabled
                if self.use_wandb:
                    artifact = wandb.Artifact(
                        f"model-{config['name']}-seed{seed}",
                        type="model",
                        metadata={"episode": episode, "config": config}
                    )
                    artifact.add_file(checkpoint_path)
                    wandb.log_artifact(artifact)
        
        # Save final results
        results_path = os.path.join(
            self.exp_dir,
            f"{config['name']}_seed{seed}_results.json"
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save final checkpoint
        if save_checkpoints:
            final_checkpoint = os.path.join(
                self.exp_dir,
                f"{config['name']}_seed{seed}_final.pt"
            )
            model.save_checkpoint(final_checkpoint)
            
            if self.use_wandb:
                # Log final model
                final_artifact = wandb.Artifact(
                    f"model-{config['name']}-seed{seed}-final",
                    type="model",
                    metadata={"episode": self.num_episodes, "config": config}
                )
                final_artifact.add_file(final_checkpoint)
                wandb.log_artifact(final_artifact)
                
                # Log final metrics
                final_metrics = {
                    'final/collective_reward': np.mean(results['collective_rewards'][-10:]) if results['collective_rewards'] else 0,
                    'final/convergence_episode': self._find_convergence(results['collective_rewards'])
                }
                wandb.log(final_metrics)
        
        # Finish WandB run
        if self.use_wandb:
            wandb.finish()
        
        return results
    
    def _find_convergence(self, rewards: List[float], threshold: float = 0.95) -> int:
        """Find episode where performance converges."""
        if len(rewards) < 10:
            return -1
        
        final_performance = np.mean(rewards[-10:])
        target = threshold * final_performance
        
        for i in range(len(rewards) - 10):
            window_mean = np.mean(rewards[i:i+10])
            if window_mean >= target:
                return i * 100
        
        return len(rewards) * 100
    
    def run_all_experiments(self, models_to_run=None) -> Dict:
        """Run all experiment configurations."""
        if models_to_run is None:
            models_to_run = ['all'] 
        all_results = {}
        
        # Standard A3C
        if 'all' in models_to_run or 'standard_a3c' in models_to_run:
            print(f"\n{'='*60}")
            print(f"Running Standard A3C baseline for {self.env_name}")
            print(f"{'='*60}")
            
            standard_results = []
            for seed in range(5):
                config = self.configs['standard_a3c'].copy()
                config['name'] = 'standard_a3c'
                results = self.run_single_experiment(config, seed)
                standard_results.append(results)
            all_results['standard_a3c'] = standard_results
        
        # Visible Actions Baseline
        if 'all' in models_to_run or 'visible_actions' in models_to_run:
            print(f"\n{'='*60}")
            print(f"Running Visible Actions baseline for {self.env_name}")
            print(f"{'='*60}")
            
            visible_results = []
            for seed in range(5):
                config = self.configs['visible_actions'].copy()
                config['name'] = 'visible_actions'
                results = self.run_single_experiment(config, seed)
                visible_results.append(results)
            all_results['visible_actions'] = visible_results
        
        # Attention-Influence models
        if 'all' in models_to_run or 'attention_influence' in models_to_run:
            print(f"\n{'='*60}")
            print(f"Running Attention-Influence experiments for {self.env_name}")
            print(f"{'='*60}")
            
            influence_results = []
            best_config = None
            best_performance = -float('inf')
            
            for influence_weight in self.configs['attention_influence']['influence_weight']:
                for curriculum_steps in self.configs['attention_influence']['curriculum_steps']:
                    config_results = []
                    
                    print(f"\nTesting influence_weight={influence_weight}, "
                          f"curriculum_steps={curriculum_steps:.1e}")
                    
                    for seed in range(5):
                        config = {
                            'influence_weight': influence_weight,
                            'curriculum_steps': curriculum_steps,
                            'visible_actions': False,
                            'use_attention': True,
                            'name': f'attention_w{influence_weight}_c{curriculum_steps}'
                        }
                        
                        results = self.run_single_experiment(config, seed)
                        config_results.append(results)
                        
                        # Track best configuration
                        if results['collective_rewards']:
                            avg_performance = np.mean(results['collective_rewards'][-10:])
                            if avg_performance > best_performance:
                                best_performance = avg_performance
                                best_config = config
                    
                    influence_results.append({
                        'config': config,
                        'results': config_results
                    })
            
            all_results['attention_influence'] = influence_results
            if best_config:
                all_results['best_influence_config'] = best_config
        
        # Save all results
        results_path = os.path.join(self.exp_dir, "all_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Create WandB summary if enabled
        if self.use_wandb:
            self._create_wandb_summary(all_results)
        
        print(f"\nExperiments completed! Results saved to: {self.exp_dir}")
        
        return all_results
    
    def _create_wandb_summary(self, all_results: Dict):
        """Create a summary run in WandB comparing all models."""
        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=f"{self.env_name}_summary",
            job_type="summary",
            tags=[self.env_name, "summary"]
        )
        
        # Create comparison table
        summary_data = []
        
        # Standard A3C
        for result in all_results.get('standard_a3c', []):
            if result['collective_rewards']:
                summary_data.append({
                    'model': 'standard_a3c',
                    'seed': result['seed'],
                    'final_reward': np.mean(result['collective_rewards'][-10:]),
                    'convergence_episode': self._find_convergence(result['collective_rewards'])
                })
        
        # Visible Actions
        for result in all_results.get('visible_actions', []):
            if result['collective_rewards']:
                summary_data.append({
                    'model': 'visible_actions',
                    'seed': result['seed'],
                    'final_reward': np.mean(result['collective_rewards'][-10:]),
                    'convergence_episode': self._find_convergence(result['collective_rewards'])
                })
        
        # Best Influence
        if 'best_influence_config' in all_results and all_results['best_influence_config']:
            best_config = all_results['best_influence_config']
            for influence_data in all_results.get('attention_influence', []):
                if influence_data['config'] == best_config:
                    for result in influence_data['results']:
                        if result['collective_rewards']:
                            summary_data.append({
                                'model': 'influence',
                                'seed': result['seed'],
                                'final_reward': np.mean(result['collective_rewards'][-10:]),
                                'convergence_episode': self._find_convergence(result['collective_rewards']),
                                'influence_weight': best_config['influence_weight'],
                                'curriculum_steps': best_config['curriculum_steps']
                            })
        
        # Log table
        if summary_data:
            table = wandb.Table(dataframe=pd.DataFrame(summary_data))
            wandb.log({"summary_table": table})
        
        wandb.finish()