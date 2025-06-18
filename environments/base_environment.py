import numpy as np
import gym
from gym import spaces
from enum import IntEnum
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any


class Actions(IntEnum):
    """Action space for agents."""
    NOOP = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4
    ROTATE_CLOCKWISE = 5
    ROTATE_COUNTERCLOCKWISE = 6
    FIRE = 7  # For cleanup beam


@dataclass
class AgentState:
    """State of an individual agent."""
    pos: Tuple[int, int]
    orientation: int  # 0: North, 1: East, 2: South, 3: West
    id: int


class BaseEnvironment(gym.Env):
    """Base class for social dilemma environments."""
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = (25, 25),
                 num_agents: int = 5,
                 view_size: int = 7):
        
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.view_size = view_size
        
        # Initialize grid
        self.grid = np.zeros(grid_size, dtype=np.int32)
        
        # Agent states
        self.agents = []
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(len(Actions))
        
        # Observation: local view + agent info
        obs_height = view_size
        obs_width = view_size
        obs_channels = 4  # terrain, resources, agents, self
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(obs_channels, obs_height, obs_width), #(4,7,7)
            dtype=np.float32
        )
        
        # Metrics
        self.step_count = 0
        self.episode_rewards = {}
    
    def _get_agent_observation(self, agent: AgentState) -> np.ndarray:
        """Get partial observation for single agent."""
        x, y = agent.pos
        view_radius = self.view_size // 2
        
        # Create observation channels
        obs = np.zeros((4, self.view_size, self.view_size), dtype=np.float32)
        
        # Get view bounds
        x_start = max(0, x - view_radius)
        x_end = min(self.grid_size[0], x + view_radius + 1)
        y_start = max(0, y - view_radius)
        y_end = min(self.grid_size[1], y + view_radius + 1)
        
        # Local coordinates in observation
        local_x_start = view_radius - (x - x_start)
        local_y_start = view_radius - (y - y_start)
        
        # Fill observation channels (to be implemented by subclasses)
        self._fill_observation_channels(obs, agent, 
                                      (x_start, x_end, y_start, y_end),
                                      (local_x_start, local_y_start))
        
        # Rotate observation based on agent orientation
        if agent.orientation != 0:
            obs = np.rot90(obs, k=agent.orientation, axes=(1, 2))
            
        return obs
    
    def _fill_observation_channels(self, obs, agent, global_bounds, local_start):
        """To be implemented by subclasses."""
        raise NotImplementedError