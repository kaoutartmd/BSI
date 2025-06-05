import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional
import json


class MetricsTracker:
    """Track and manage experiment metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.episode_metrics = defaultdict(list)
        
    def add_scalar(self, name: str, value: float, step: Optional[int] = None):
        """Add a scalar value to tracking."""
        self.metrics[name].append(value)
        
    def add_episode_metrics(self, episode: int, metrics: Dict[str, float]):
        """Add metrics for a complete episode."""
        for key, value in metrics.items():
            self.episode_metrics[key].append((episode, value))
    
    def get_average(self, name: str, last_n: Optional[int] = None) -> float:
        """Get average of a metric over last n values."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        
        values = list(self.metrics[name])
        if last_n is not None:
            values = values[-last_n:]
            
        return np.mean(values)
    
    def get_std(self, name: str, last_n: Optional[int] = None) -> float:
        """Get standard deviation of a metric."""
        if name not in self.metrics or len(self.metrics[name]) < 2:
            return 0.0
        
        values = list(self.metrics[name])
        if last_n is not None:
            values = values[-last_n:]
            
        return np.std(values)
    
    def save(self, filepath: str):
        """Save metrics to file."""
        data = {
            'window_metrics': {k: list(v) for k, v in self.metrics.items()},
            'episode_metrics': dict(self.episode_metrics)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load metrics from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metrics.clear()
        self.episode_metrics.clear()
        
        for key, values in data['window_metrics'].items():
            self.metrics[key] = deque(values, maxlen=self.window_size)
        
        self.episode_metrics.update(data['episode_metrics'])