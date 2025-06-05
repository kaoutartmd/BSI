import torch
import numpy as np
from .influence_a3c import EnhancedInfluenceA3C
from .actor_critic import ActorCriticNetwork
from typing import List


class VisibleActionsBaseline(EnhancedInfluenceA3C):
    """
    Ablated version where agents can see others' actions but don't receive
    influence reward.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override to not use influence reward
        self.influence_weight = 0.0
        
        # Rebuild networks with extended state space
        self.agents = []
        self.optimizers = []
        
        # Extend state dimension to include other agents' actions
        extended_state_dim = self.state_dim + (self.num_agents - 1) * self.action_dim
        
        for i in range(self.num_agents):
            agent = ActorCriticNetwork(
                extended_state_dim, 
                self.action_dim, 
                self.hidden_dim
            ).to(self.device)
            self.agents.append(agent)
            self.optimizers.append(torch.optim.Adam(agent.parameters(), lr=1e-4))
    
    def _extend_state_with_actions(self, state: torch.Tensor, 
                                  agent_idx: int, 
                                  actions: List[int]) -> torch.Tensor:
        """Extend state to include other agents' actions."""
        # Create one-hot encoding of other agents' actions
        other_actions = []
        for i, action in enumerate(actions):
            if i != agent_idx:
                one_hot = torch.zeros(self.action_dim)
                one_hot[action] = 1.0
                other_actions.append(one_hot)
        
        other_actions_tensor = torch.cat(other_actions).to(self.device)
        
        # Concatenate with original state
        extended_state = torch.cat([state.flatten(), other_actions_tensor])
        
        return extended_state.unsqueeze(0)