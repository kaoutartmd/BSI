import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network with proper initialization."""
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for output layers
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
    
    def forward(self, x):
        """Forward pass returning action probabilities and value."""
        # Ensure input is properly shaped
        if len(x.shape) == 4:  # [batch, channels, height, width]
            x = x.view(x.size(0), -1)
        elif len(x.shape) == 3:  # [channels, height, width] - single observation
            x = x.view(1, -1)
        elif len(x.shape) == 2 and x.shape[1] != self.input_dim:  # Wrong 2D shape
            x = x.view(x.size(0), -1)
            
        features = self.features(x)
        
        # Actor output
        action_logits = self.actor(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic output
        value = self.critic(features)
        
        return action_probs, value