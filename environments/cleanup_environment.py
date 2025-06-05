import numpy as np
from .harvest_environment import HarvestEnvironment
from .base_environment import Actions, AgentState


class CleanupEnvironment(HarvestEnvironment):
    """
    Cleanup environment: agents must clean river to spawn apples.
    """
    
    def __init__(self, *args, river_width: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.river_width = river_width
        self.pollution_level = np.ones((self.grid_size[0], river_width)) * 0.5
        self.waste_spawn_prob = 0.5
        
    def reset(self):
        """Reset with polluted river."""
        obs = super().reset()
        
        # Initialize pollution
        self.pollution_level.fill(0.5)
        
        # No apples spawn initially in cleanup
        self.apple_grid.fill(False)
        
        return obs
    
    def _execute_action(self, agent: AgentState, action: int) -> float:
        """Execute action with cleanup beam."""
        if action == Actions.FIRE:
            # Clean in front of agent
            reward = self._cleanup_beam(agent)
            return reward
        else:
            return super()._execute_action(agent, action)
    
    def _cleanup_beam(self, agent: AgentState) -> float:
        """Fire cleanup beam and return reward."""
        x, y = agent.pos
        
        # Determine beam direction based on orientation
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        dx, dy = directions[agent.orientation]
        
        cleaned = 0
        beam_length = 5
        
        for i in range(1, beam_length + 1):
            beam_x = x + i * dx
            beam_y = y + i * dy
            
            # Check if beam hits river
            if 0 <= beam_x < self.grid_size[0] and 0 <= beam_y < self.river_width:
                reduction = 0.1
                self.pollution_level[beam_x, beam_y] = max(
                    0, self.pollution_level[beam_x, beam_y] - reduction
                )
                cleaned += reduction
                
        return cleaned * 0.1  # Small reward for cleaning
    
    def _regrow_apples(self):
        """Apples only grow if river is clean."""
        # Check average pollution level
        avg_pollution = np.mean(self.pollution_level)
        
        if avg_pollution < 0.3:  # Clean enough
            super()._regrow_apples()
        
        # Pollution slowly increases
        self.pollution_level = np.minimum(
            1.0, self.pollution_level + self.waste_spawn_prob * 0.01
        )