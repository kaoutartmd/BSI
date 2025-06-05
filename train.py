#!/usr/bin/env python3
"""
Social Influence in Multi-Agent Reinforcement Learning
Main experiment script
"""

import argparse
import os
import torch
from datetime import datetime

from experiments import ExperimentRunner, ExperimentAnalyzer
from configs.experiment_config import EXPERIMENT_CONFIG, TRAINING_CONFIG


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run social influence experiments in multi-agent RL"
    )
    
    parser.add_argument(
        '--env', 
        type=str, 
        choices=['harvest', 'cleanup', 'both'],
        default='both',
        help='Environment to run experiments on'
    )
    
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=TRAINING_CONFIG['num_episodes'],
        help='Number of episodes to train'
    )
    
    parser.add_argument(
        '--num-agents',
        type=int,
        default=5,
        help='Number of agents in environment'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='experiments',
        help='Directory to save experiment logs'
    )
    
    parser.add_argument(
        '--analyze-only',
        type=str,
        default=None,
        help='Path to experiment directory to analyze (skip training)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Use Weights & Biases for experiment tracking'
    )
    
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='basic_influence',
        help='WandB project name'
    )
    
    parser.add_argument(
        '--wandb-entity',
        type=str,
        default=None,
        help='WandB entity (username or team)'
    )
    
    parser.add_argument(
        '--wandb-mode',
        type=str,
        choices=['online', 'offline', 'disabled'],
        default='online',
        help='WandB mode'
    )
    
    return parser.parse_args()
    


def run_experiments(env_name: str, args):
    """Run experiments for a specific environment."""
    print(f"\n{'='*70}")
    print(f"Running {env_name.upper()} experiments")
    print(f"{'='*70}\n")
    
    # Set WandB mode if specified
    if hasattr(args, 'wandb_mode') and args.use_wandb:
        os.environ['WANDB_MODE'] = args.wandb_mode
    
    # Create experiment runner with WandB support
    runner = ExperimentRunner(
        env_name=env_name,
        num_agents=args.num_agents,
        num_episodes=args.num_episodes,
        log_dir=args.log_dir,
        device=args.device,
        use_wandb=args.use_wandb if hasattr(args, 'use_wandb') else False,
        wandb_project=args.wandb_project if hasattr(args, 'wandb_project') else 'social-influence-marl',
        wandb_entity=args.wandb_entity if hasattr(args, 'wandb_entity') else None
    )
    
    # Run all experiments
    results = runner.run_all_experiments()
    
    # Analyze results
    analyzer = ExperimentAnalyzer(args.log_dir)
    
    # Generate report
    analyzer.generate_report(runner.exp_dir)
    
    # Generate plots
    if not args.no_plots:
        analyzer.generate_plots(runner.exp_dir)
    
    return runner.exp_dir

def analyze_existing(exp_dir: str, args):
    """Analyze existing experiment results."""
    print(f"\nAnalyzing experiment: {exp_dir}")
    
    analyzer = ExperimentAnalyzer(os.path.dirname(exp_dir))
    
    # Generate report
    analyzer.generate_report(exp_dir)
    
    # Generate plots
    if not args.no_plots:
        analyzer.generate_plots(exp_dir)


def main():
    """Main entry point."""
    args = parse_args()
    
    print("Social Influence in Multi-Agent RL")
    print(f"Device: {args.device}")
    print(f"Log directory: {args.log_dir}")
    
    # If analyzing existing results
    if args.analyze_only:
        analyze_existing(args.analyze_only, args)
        return
    
    # Run experiments
    exp_dirs = []
    
    if args.env in ['harvest', 'both']:
        exp_dir = run_experiments('harvest', args)
        exp_dirs.append(exp_dir)
    
    if args.env in ['cleanup', 'both']:
        exp_dir = run_experiments('cleanup', args)
        exp_dirs.append(exp_dir)
    
    # Summary
    print(f"\n{'='*70}")
    print("EXPERIMENTS COMPLETED!")
    print(f"{'='*70}")
    print("\nResults saved to:")
    for exp_dir in exp_dirs:
        print(f"  - {exp_dir}")
    
    print("\nTo re-analyze results, run:")
    for exp_dir in exp_dirs:
        print(f"  python main.py --analyze-only {exp_dir}")


if __name__ == "__main__":
    main()