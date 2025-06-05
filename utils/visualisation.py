import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict


class Visualizer:
    """Visualization utilities for experiment results."""
    
    def __init__(self, style: str = 'seaborn'):
        plt.style.use(style)
        sns.set_palette("husl")
    
    @staticmethod
    def plot_learning_curves(results: Dict[str, List], 
                           window_size: int = 200,
                           confidence: float = 0.995) -> plt.Figure:
        """Plot learning curves with confidence intervals."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {'standard_a3c': 'blue', 
                 'visible_actions': 'orange',
                 'influence': 'green'}
        
        for model_type, model_results in results.items():
            if model_type not in colors:
                continue
                
            # Aggregate results across seeds
            all_rewards = []
            min_length = min(len(r['collective_rewards']) for r in model_results)
            
            for result in model_results:
                all_rewards.append(result['collective_rewards'][:min_length])
            
            all_rewards = np.array(all_rewards)
            mean_rewards = np.mean(all_rewards, axis=0)
            
            # Compute confidence interval
            n_seeds = len(model_results)
            sem = stats.sem(all_rewards, axis=0)
            h = sem * stats.t.ppf((1 + confidence) / 2, n_seeds - 1)
            
            # Smooth with sliding window
            if window_size > 0:
                mean_smoothed = Visualizer._smooth(mean_rewards, window_size)
                lower_smoothed = Visualizer._smooth(mean_rewards - h, window_size)
                upper_smoothed = Visualizer._smooth(mean_rewards + h, window_size)
            else:
                mean_smoothed = mean_rewards
                lower_smoothed = mean_rewards - h
                upper_smoothed = mean_rewards + h
            
            x = np.arange(len(mean_smoothed))
            
            ax.plot(x, mean_smoothed, label=model_type.replace('_', ' ').title(), 
                   color=colors[model_type], linewidth=2)
            ax.fill_between(x, lower_smoothed, upper_smoothed, 
                          alpha=0.2, color=colors[model_type])
        
        ax.set_xlabel('Episodes (×100)', fontsize=12)
        ax.set_ylabel('Collective Reward', fontsize=12)
        ax.set_title('Learning Curves Comparison', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_influence_heatmap(influence_matrix: np.ndarray,
                             action_correlations: np.ndarray) -> plt.Figure:
        """Plot influence patterns as heatmaps."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Action correlations
        sns.heatmap(action_correlations, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, ax=ax1,
                   cbar_kws={'label': 'Correlation'})
        ax1.set_title('Action Correlations Between Agents')
        ax1.set_xlabel('Agent')
        ax1.set_ylabel('Agent')
        
        # Influence matrix
        if np.any(influence_matrix):
            sns.heatmap(influence_matrix, annot=True, fmt='.2f',
                       cmap='viridis', ax=ax2,
                       cbar_kws={'label': 'Influence'})
            ax2.set_title('Causal Influence Matrix')
            ax2.set_xlabel('Influenced Agent')
            ax2.set_ylabel('Influencing Agent')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_hyperparameter_analysis(hp_results: Dict) -> plt.Figure:
        """Plot hyperparameter sensitivity analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Influence weight sensitivity
        weights = []
        performances = []
        
        for config, results in hp_results.items():
            if 'influence_weight' in config:
                weight = config['influence_weight']
                perf = np.mean([np.mean(r['collective_rewards'][-10:]) 
                              for r in results])
                weights.append(weight)
                performances.append(perf)
        
        ax1.scatter(weights, performances, s=100, alpha=0.6)
        ax1.set_xlabel('Influence Weight')
        ax1.set_ylabel('Final Performance')
        ax1.set_xscale('log')
        ax1.set_title('Influence Weight Sensitivity')
        ax1.grid(True, alpha=0.3)
        
        # Curriculum steps analysis
        curriculum_data = defaultdict(list)
        
        for config, results in hp_results.items():
            if 'curriculum_steps' in config:
                steps = config['curriculum_steps']
                perf = np.mean([np.mean(r['collective_rewards'][-10:]) 
                              for r in results])
                curriculum_data[steps].append(perf)
        
        steps = sorted(curriculum_data.keys())
        means = [np.mean(curriculum_data[s]) for s in steps]
        stds = [np.std(curriculum_data[s]) for s in steps]
        
        ax2.errorbar(steps, means, yerr=stds, marker='o', capsize=5)
        ax2.set_xlabel('Curriculum Steps')
        ax2.set_ylabel('Final Performance')
        ax2.set_title('Curriculum Learning Impact')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def _smooth(data: np.ndarray, window_size: int) -> np.ndarray:
        """Apply sliding window smoothing."""
        if window_size == 0 or window_size >= len(data):
            return data
            
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(data, kernel, mode='valid')
        
        # Pad to maintain size
        pad_size = len(data) - len(smoothed)
        left_pad = pad_size // 2
        right_pad = pad_size - left_pad
        
        smoothed = np.pad(smoothed, (left_pad, right_pad), mode='edge')
        
        return smoothed