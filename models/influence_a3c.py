import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
from torch.distributions import Categorical

from .actor_critic import ActorCriticNetwork


class EnhancedInfluenceA3C:
    """
    Complete implementation with counterfactual computation and metrics.
    """
    
    def __init__(self,
                 env: gym.Env,
                 num_agents: int,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 influence_weight: float = 0.1,
                 curriculum_steps: int = 2e7,
                 device: str = 'cpu'):
        
        self.env = env
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.influence_weight = influence_weight
        self.curriculum_steps = curriculum_steps
        self.device = torch.device(device)
        
        # Networks
        self.agents = []
        self.optimizers = []
        
        for _ in range(num_agents):
            agent = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.agents.append(agent)
            self.optimizers.append(optim.Adam(agent.parameters(), lr=lr))
        
        # Metrics tracking
        def create_deque_1000():
            return deque(maxlen=1000)

        self.metrics = {
            'collective_rewards': deque(maxlen=1000),
            'individual_rewards': defaultdict(create_deque_1000),
            'influence_rewards': deque(maxlen=1000),
            'losses': deque(maxlen=1000),
            'episode_lengths': deque(maxlen=100)
        }
        
        self.step_count = 0
        self.episode_count = 0

    def compute_counterfactual_probs(self, states: List[torch.Tensor], agent_idx: int) -> List[torch.Tensor]:
        """
        Compute counterfactual action probabilities for other agents.
        """
        counterfactual_probs = []
        for j in range(self.num_agents):
            if j != agent_idx:
                uniform_prob = torch.ones(self.action_dim) / self.action_dim
                counterfactual_probs.append(uniform_prob.to(self.device))
        return counterfactual_probs

    def compute_influence_reward(self,
                               agent_idx: int,
                               states: List[torch.Tensor],
                               actions: List[int],
                               next_states: List[torch.Tensor]) -> float:
        """
        Compute causal influence reward using counterfactual reasoning.
        """
        influence = 0.0
        counterfactual_probs = self.compute_counterfactual_probs(states, agent_idx)

        for j, agent in enumerate(self.agents):
            if j != agent_idx:
                with torch.no_grad():
                    actual_probs, _ = agent(next_states[j])
                cf_prob = counterfactual_probs[j if j < agent_idx else j - 1]
                kl_div = F.kl_div(torch.log(actual_probs + 1e-8), cf_prob, reduction='sum')
                influence += kl_div.item()

        return influence / (self.num_agents - 1)

    def get_current_influence_weight(self) -> float:
        """Curriculum learning schedule."""
        if self.curriculum_steps == 0:
            return self.influence_weight
        progress = min(1.0, self.step_count / self.curriculum_steps)
        return self.influence_weight * progress

    def train_episode(self):
        """Run one episode and train agents."""
        states = self.env.reset()
        episode_reward = 0
        episode_length = 0
        individual_rewards = np.zeros(self.num_agents)  # Track per-agent rewards
        
        experiences = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }

        done = False

        while not done and episode_length < 1000:
            state_tensors = []
            for s in states:
                state_tensor = torch.FloatTensor(s.copy()).to(self.device)
                if len(state_tensor.shape) == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                elif len(state_tensor.shape) == 2:
                    state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
                state_tensors.append(state_tensor)

            actions = []
            log_probs = []
            values = []

            for agent, state_t in zip(self.agents, state_tensors):
                probs, value = agent(state_t)
                dist = Categorical(probs)
                action = dist.sample()
                actions.append(action.item())
                log_probs.append(dist.log_prob(action))
                values.append(value)

            next_states, rewards, done, info = self.env.step(actions)
            individual_rewards += rewards

            next_state_tensors = []
            for s in next_states:
                next_state_tensor = torch.FloatTensor(s.copy()).to(self.device)
                if len(next_state_tensor.shape) == 1:
                    next_state_tensor = next_state_tensor.unsqueeze(0)
                elif len(next_state_tensor.shape) == 2:
                    next_state_tensor = next_state_tensor.unsqueeze(0).unsqueeze(0)
                next_state_tensors.append(next_state_tensor)

            experiences['states'].append(state_tensors)
            experiences['actions'].append(actions)
            experiences['rewards'].append(rewards)
            experiences['next_states'].append(next_state_tensors)
            experiences['log_probs'].append(log_probs)
            experiences['values'].append(values)
            experiences['dones'].append(done)

            episode_reward += info.get('collective_reward', sum(rewards))
            episode_length += 1
            states = next_states

        self.metrics['collective_rewards'].append(episode_reward)
        self.metrics['episode_lengths'].append(episode_length)

        self.train_on_episode(experiences)
        self.episode_count += 1
        return episode_reward

    def train_on_episode(self, experiences: Dict):
        """Train agents on collected episode."""
        episode_length = len(experiences['rewards'])
        current_influence_weight = self.get_current_influence_weight()

        for agent_idx in range(self.num_agents):
            returns = []
            advantages = []

            next_values = [0] * episode_length
            for t in range(episode_length):
                if t < episode_length - 1:
                    with torch.no_grad():
                        if agent_idx < len(experiences['next_states'][t]):
                            _, next_value = self.agents[agent_idx](experiences['next_states'][t][agent_idx])
                            next_values[t] = next_value.item()
                        else:
                            next_values[t] = 0
                else:
                    next_values[t] = 0

            for t in reversed(range(episode_length)):
                env_reward = experiences['rewards'][t][agent_idx]
                influence_reward = 0
                if current_influence_weight > 0:
                    influence_reward = self.compute_influence_reward(
                        agent_idx,
                        experiences['states'][t],
                        experiences['actions'][t],
                        experiences['next_states'][t]
                    )

                total_reward = env_reward + current_influence_weight * influence_reward
                value = experiences['values'][t][agent_idx].item()
                td_error = total_reward + self.gamma * next_values[t] - value

                returns.insert(0, total_reward + self.gamma * next_values[t])
                advantages.insert(0, td_error)

            returns = torch.FloatTensor(returns).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)

            total_loss = 0

            for t in range(episode_length):
                log_prob = experiences['log_probs'][t][agent_idx]
                value = experiences['values'][t][agent_idx]
                actor_loss = -log_prob * advantages[t].detach()
                critic_loss = F.mse_loss(value.squeeze(), returns[t].detach())

                state = experiences['states'][t][agent_idx]
                probs, _ = self.agents[agent_idx](state)
                entropy = Categorical(probs).entropy()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                total_loss += loss

            self.optimizers[agent_idx].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[agent_idx].parameters(), 0.5)
            self.optimizers[agent_idx].step()

            self.metrics['losses'].append(total_loss.item())

        self.step_count += episode_length

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        metrics_to_save = {
            'collective_rewards': list(self.metrics['collective_rewards']),
            'individual_rewards': {k: list(v) for k, v in self.metrics['individual_rewards'].items()},
            'influence_rewards': list(self.metrics['influence_rewards']),
            'losses': list(self.metrics['losses']),
            'episode_lengths': list(self.metrics['episode_lengths'])
        }
        
        checkpoint = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'metrics': metrics_to_save,
            'agent_states': [agent.state_dict() for agent in self.agents],
            'optimizer_states': [opt.state_dict() for opt in self.optimizers]
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)
        
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        
        # Reconstruct metrics with proper types
        self.metrics['collective_rewards'] = deque(checkpoint['metrics']['collective_rewards'], maxlen=1000)
        self.metrics['influence_rewards'] = deque(checkpoint['metrics']['influence_rewards'], maxlen=1000)
        self.metrics['losses'] = deque(checkpoint['metrics']['losses'], maxlen=1000)
        self.metrics['episode_lengths'] = deque(checkpoint['metrics']['episode_lengths'], maxlen=100)
        
        # Reconstruct individual_rewards as defaultdict
        def create_deque_1000():
            return deque(maxlen=1000)
            
        self.metrics['individual_rewards'] = defaultdict(create_deque_1000)
        for k, v in checkpoint['metrics']['individual_rewards'].items():
            self.metrics['individual_rewards'][k] = deque(v, maxlen=1000)
        
        for i, (agent, opt) in enumerate(zip(self.agents, self.optimizers)):
            agent.load_state_dict(checkpoint['agent_states'][i])
            opt.load_state_dict(checkpoint['optimizer_states'][i])