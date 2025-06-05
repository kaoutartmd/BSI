import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from typing import Optional


from utils import Visualizer


class ExperimentAnalyzer:
    """Tools for analyzing and comparing experiment results."""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.visualizer = Visualizer()
    
    def load_results(self, exp_dir: str) -> Dict:
        """Load experiment results from directory."""
        results_path = os.path.join(exp_dir, "all_results.json")
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def compute_statistics(self, results: Dict) -> pd.DataFrame:
        """Compute summary statistics for all models."""
        stats_data = []
        
        for model_type in ['standard_a3c', 'visible_actions']:
            model_results = results.get(model_type, [])
            
            final_performances = []
            convergence_episodes = []
            
            for seed_result in model_results:
                rewards = seed_result['collective_rewards']
                if len(rewards) > 10:
                    final_perf = np.mean(rewards[-10:])
                    final_performances.append(final_perf)
                    
                    # Find convergence point (95% of final performance)
                    conv_ep = self._find_convergence(rewards, threshold=0.95)
                    convergence_episodes.append(conv_ep)
            
            if final_performances:
                stats_data.append({
                    'model': model_type,
                    'mean_performance': np.mean(final_performances),
                    'std_performance': np.std(final_performances),
                    'mean_convergence': np.mean(convergence_episodes),
                    'n_seeds': len(final_performances)
                })
        
        # Handle influence results
        if 'influence' in results:
            best_idx = self._find_best_influence_config(results['influence'])
            best_influence = results['influence'][best_idx]
            
            final_performances = []
            convergence_episodes = []
            
            for seed_result in best_influence['results']:
                rewards = seed_result['collective_rewards']
                if len(rewards) > 10:
                    final_perf = np.mean(rewards[-10:])
                    final_performances.append(final_perf)
                    
                    conv_ep = self._find_convergence(rewards, threshold=0.95)
                    convergence_episodes.append(conv_ep)
            
            if final_performances:
                stats_data.append({
                    'model': 'influence',
                    'mean_performance': np.mean(final_performances),
                    'std_performance': np.std(final_performances),
                    'mean_convergence': np.mean(convergence_episodes),
                    'n_seeds': len(final_performances)
                })
        
        return pd.DataFrame(stats_data)
    
    def statistical_comparison(self, results: Dict) -> Dict:
        """Perform statistical comparisons between models."""
        comparisons = {}
        
        # Extract final performances
        model_performances = {}
        
        for model_type in ['standard_a3c', 'visible_actions']:
            perfs = []
            for seed_result in results.get(model_type, []):
                rewards = seed_result['collective_rewards']
                if len(rewards) > 10:
                    perfs.append(np.mean(rewards[-10:]))
            model_performances[model_type] = perfs
        
        # Get best influence configuration
        if 'influence' in results:
            best_idx = self._find_best_influence_config(results['influence'])
            best_influence = results['influence'][best_idx]
            
            perfs = []
            for seed_result in best_influence['results']:
                rewards = seed_result['collective_rewards']
                if len(rewards) > 10:
                    perfs.append(np.mean(rewards[-10:]))
            model_performances['influence'] = perfs
        
        # Pairwise comparisons
        models = list(model_performances.keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model1, model2 = models[i], models[j]
                
                if model_performances[model1] and model_performances[model2]:
                    t_stat, p_value = stats.ttest_ind(
                        model_performances[model1],
                        model_performances[model2]
                    )
                    
                    comparisons[f"{model1}_vs_{model2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant_at_995': p_value < 0.005,
                        'mean_difference': np.mean(model_performances[model1]) - 
                                         np.mean(model_performances[model2])
                    }
        
        return comparisons
    
    def generate_report(self, exp_dir: str, save_path: Optional[str] = None):
        """Generate comprehensive analysis report."""
        results = self.load_results(exp_dir)
        
        # Create report
        report = f"# Experiment Analysis Report\n\n"
        report += f"## Environment: {os.path.basename(exp_dir).split('_')[0]}\n\n"
        
        # Summary statistics
        stats_df = self.compute_statistics(results)
        report += "## Summary Statistics\n\n"
        report += stats_df.to_markdown(index=False)
        report += "\n\n"
        
        # Statistical comparisons
        comparisons = self.statistical_comparison(results)
        report += "## Statistical Comparisons\n\n"
        
        for comparison, stats in comparisons.items():
            report += f"### {comparison}\n"
            report += f"- t-statistic: {stats['t_statistic']:.3f}\n"
            report += f"- p-value: {stats['p_value']:.4f}\n"
            report += f"- Significant at 99.5% CI: {stats['significant_at_995']}\n"
            report += f"- Mean difference: {stats['mean_difference']:.3f}\n\n"
        
        # Best configuration
        if 'best_influence_config' in results:
            report += "## Best Influence Configuration\n\n"
            best_config = results['best_influence_config']
            report += f"- Influence Weight: {best_config['influence_weight']}\n"
            report += f"- Curriculum Steps: {best_config['curriculum_steps']:.1e}\n\n"
        
        # Save report
        if save_path is None:
            save_path = os.path.join(exp_dir, "analysis_report.md")
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {save_path}")
        
        return report
    
    def generate_plots(self, exp_dir: str):
        """Generate all analysis plots."""
        results = self.load_results(exp_dir)
        
        # Prepare data for plotting
        plot_data = {}
        for model_type in ['standard_a3c', 'visible_actions']:
            if model_type in results:
                plot_data[model_type] = results[model_type]
        
        # Add best influence configuration
        if 'influence' in results:
            best_idx = self._find_best_influence_config(results['influence'])
            plot_data['influence'] = results['influence'][best_idx]['results']
        
        # Generate plots
        # 1. Learning curves
        fig1 = self.visualizer.plot_learning_curves(plot_data)
        fig1.savefig(os.path.join(exp_dir, "learning_curves.png"), dpi=300, bbox_inches='tight')
        
        # 2. Hyperparameter analysis (if influence results exist)
        if 'influence' in results:
            hp_data = {}
            for influence_data in results['influence']:
                config_key = f"w{influence_data['config']['influence_weight']}_c{influence_data['config']['curriculum_steps']}"
                hp_data[influence_data['config']] = influence_data['results']
            
            fig2 = self.visualizer.plot_hyperparameter_analysis(hp_data)
            fig2.savefig(os.path.join(exp_dir, "hyperparameter_analysis.png"), dpi=300, bbox_inches='tight')
        
        plt.close('all')
        print(f"Plots saved to: {exp_dir}")
    
    def _find_convergence(self, rewards: List[float], threshold: float = 0.95) -> int:
        """Find episode where performance converges."""
        if len(rewards) < 10:
            return -1
        
        final_performance = np.mean(rewards[-10:])
        target = threshold * final_performance
        
        # Use sliding window
        window_size = 10
        for i in range(len(rewards) - window_size):
            window_mean = np.mean(rewards[i:i+window_size])
            if window_mean >= target:
                return i * 100  # Convert to actual episode number
        
        return len(rewards) * 100
    
    def _find_best_influence_config(self, influence_results: List) -> int:
        """Find index of best performing configuration."""
        best_idx = 0
        best_perf = -float('inf')
        
        for i, data in enumerate(influence_results):
            perfs = []
            for r in data['results']:
                if len(r['collective_rewards']) >= 10:
                    perfs.append(np.mean(r['collective_rewards'][-10:]))
            
            if perfs:
                avg_perf = np.mean(perfs)
                if avg_perf > best_perf:
                    best_perf = avg_perf
                    best_idx = i
        
        return best_idx