import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
from .base_environment import BaseEnvironment, Actions, AgentState


class HarvestEnvironment(BaseEnvironment):
    """
    Harvest environment: agents collect apples that regrow based on local density.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (25, 25),
        num_agents: int = 5,
        view_size: int = 7,
        apple_density: float = 0.3,
        regrow_prob: float = 0.01,
    ):

        super().__init__(grid_size, num_agents, view_size)

        self.apple_density = apple_density
        self.regrow_prob = regrow_prob

        # Apple grid
        self.apple_grid = np.zeros(grid_size, dtype=bool)

    def reset(self):
        """Reset environment to initial state."""
        self.grid.fill(0)
        self.apple_grid.fill(False)
        self.agents = []
        self.step_count = 0
        self.episode_rewards = defaultdict(float)

        # Place agents randomly
        for i in range(self.num_agents):
            while True:
                x = np.random.randint(0, self.grid_size[0])
                y = np.random.randint(0, self.grid_size[1])
                if self.grid[x, y] == 0:
                    agent = AgentState(pos=(x, y), orientation=0, id=i)
                    self.agents.append(agent)
                    self.grid[x, y] = i + 1  # Agent IDs start from 1
                    break

        # Initialize apples
        num_apples = int(self.grid_size[0] * self.grid_size[1] * self.apple_density)
        apple_positions = np.random.choice(
            self.grid_size[0] * self.grid_size[1], num_apples, replace=False
        )

        for pos in apple_positions:
            x = pos // self.grid_size[1]
            y = pos % self.grid_size[1]
            if self.grid[x, y] == 0:  # No agent at this position
                self.apple_grid[x, y] = True

        return self._get_observations()

    def step(self, actions: List[int]):
        """Execute actions for all agents."""
        rewards = np.zeros(self.num_agents)

        # Execute actions
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            reward = self._execute_action(agent, action)
            rewards[i] = reward
            self.episode_rewards[i] += reward

        # Regrow apples based on local density
        self._regrow_apples()

        # Get observations
        observations = self._get_observations()

        # Check if episode is done (could add time limit)
        done = self.step_count >= 1000

        self.step_count += 1

        info = {
            "collective_reward": np.sum(rewards),
            "episode_rewards": dict(self.episode_rewards),
        }

        return observations, rewards, done, info

    def _execute_action(self, agent: AgentState, action: int) -> float:
        """Execute single agent action and return reward."""
        reward = 0.0
        old_pos = agent.pos

        if action == Actions.NOOP:
            pass
        elif action == Actions.MOVE_LEFT:
            new_pos = (agent.pos[0], max(0, agent.pos[1] - 1))
        elif action == Actions.MOVE_RIGHT:
            new_pos = (agent.pos[0], min(self.grid_size[1] - 1, agent.pos[1] + 1))
        elif action == Actions.MOVE_UP:
            new_pos = (max(0, agent.pos[0] - 1), agent.pos[1])
        elif action == Actions.MOVE_DOWN:
            new_pos = (min(self.grid_size[0] - 1, agent.pos[0] + 1), agent.pos[1])
        elif action == Actions.ROTATE_CLOCKWISE:
            agent.orientation = (agent.orientation + 1) % 4
            return reward
        elif action == Actions.ROTATE_COUNTERCLOCKWISE:
            agent.orientation = (agent.orientation - 1) % 4
            return reward
        else:
            return reward

        # Try to move
        if action in [
            Actions.MOVE_LEFT,
            Actions.MOVE_RIGHT,
            Actions.MOVE_UP,
            Actions.MOVE_DOWN,
        ]:
            if self.grid[new_pos] == 0:  # Empty cell
                # Update grid
                self.grid[old_pos] = 0
                self.grid[new_pos] = agent.id + 1
                agent.pos = new_pos

                # Check for apple
                if self.apple_grid[new_pos]:
                    reward = 1.0
                    self.apple_grid[new_pos] = False

        return reward

    def _regrow_apples(self):
        """Regrow apples based on local density."""
        new_apples = self.apple_grid.copy()

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if not self.apple_grid[i, j] and self.grid[i, j] == 0:
                    # Count nearby apples
                    nearby_apples = 0
                    for di in range(-2, 3):
                        for dj in range(-2, 3):
                            ni, nj = i + di, j + dj
                            if (
                                0 <= ni < self.grid_size[0]
                                and 0 <= nj < self.grid_size[1]
                            ):
                                if self.apple_grid[ni, nj]:
                                    nearby_apples += 1

                    # Probability of regrowth increases with nearby apple density
                    regrow_prob = self.regrow_prob * (nearby_apples / 25.0)
                    if np.random.random() < regrow_prob:
                        new_apples[i, j] = True

        self.apple_grid = new_apples

    def _get_observations(self) -> List[np.ndarray]:
        """Get partial observations for all agents."""
        observations = []

        for agent in self.agents:
            obs = self._get_agent_observation(agent)
            observations.append(obs)

        return observations

    def _fill_observation_channels(self, obs, agent, global_bounds, local_start):
        """Fill observation channels for harvest environment."""
        x_start, x_end, y_start, y_end = global_bounds
        local_x_start, local_y_start = local_start

        # Channel 0: Terrain (walls, etc - not used in basic harvest)
        # Channel 1: Apples
        obs[
            1,
            local_x_start : local_x_start + x_end - x_start,
            local_y_start : local_y_start + y_end - y_start,
        ] = self.apple_grid[x_start:x_end, y_start:y_end]
        

        # Channel 2: Other agents
        for other_agent in self.agents:
            if other_agent.id != agent.id:
                ox, oy = other_agent.pos
                if x_start <= ox < x_end and y_start <= oy < y_end:
                    local_ox = local_x_start + (ox - x_start)
                    local_oy = local_y_start + (oy - y_start)
                    obs[2, local_ox, local_oy] = 1.0

        # Channel 3: Self
        obs[3, self.view_size // 2, self.view_size // 2] = 1.0
