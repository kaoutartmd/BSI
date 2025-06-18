import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional


class AttentionModule(nn.Module):
    """Simple attention module for processing neighbor observations."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Networks for attention mechanism
        self.embed_net = nn.Linear(state_dim, hidden_dim)
        self.key_net = nn.Linear(hidden_dim, hidden_dim)
        self.query_net = nn.Linear(state_dim + hidden_dim, hidden_dim)
        self.attention_net = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, my_state: torch.Tensor, neighbor_states: List[torch.Tensor]):
        """
        Compute attention weights for neighbors.
        
        Returns:
            weighted_embedding: Weighted sum of neighbor embeddings
            attention_weights: Attention score for each neighbor
        """
        if not neighbor_states:
            return torch.zeros(1, self.embed_net.out_features).to(my_state.device), None
        
        # Embed all neighbors
        embeddings = []
        keys = []
        for neighbor_state in neighbor_states:
            embed = self.embed_net(neighbor_state)
            embeddings.append(embed)
            keys.append(self.key_net(embed))
        
        embeddings = torch.stack(embeddings)  # [num_neighbors, hidden_dim]
        keys = torch.stack(keys)
        
        # Create query from my state + mean of embeddings
        mean_embedding = embeddings.mean(dim=0, keepdim=True)
        query_input = torch.cat([my_state, mean_embedding.squeeze(0)], dim=-1)
        query = self.query_net(query_input).unsqueeze(0)  # [1, hidden_dim]
        
        # Compute attention scores
        scores = []
        for key in keys:
            score_input = torch.cat([query.squeeze(0), key], dim=-1)
            score = self.attention_net(score_input)
            scores.append(score)
        
        scores = torch.cat(scores)  # [num_neighbors]
        attention_weights = F.softmax(scores, dim=0)
        
        # Apply attention to embeddings
        weighted_embedding = (embeddings * attention_weights.unsqueeze(1)).sum(dim=0)
        
        return weighted_embedding, attention_weights


class AttentionActorCritic(nn.Module):
    """Actor-Critic with attention mechanism."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Attention module
        self.attention = AttentionModule(state_dim, hidden_dim=64)
        
        # Main network
        self.fc1 = nn.Linear(state_dim + 64, hidden_dim)  # state + attention embedding
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output heads
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, neighbor_states: List[torch.Tensor]):
        # Get attention-weighted neighbor representation
        neighbor_embedding, attention_weights = self.attention(state, neighbor_states)
        
        # Combine with own state
        x = torch.cat([state, neighbor_embedding], dim=-1)
        
        # Process through network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Get outputs
        action_probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        
        return action_probs, value, attention_weights


class AttentionInfluenceA3C:
    """Simplified A3C with attention-based influence rewards."""
    
    def __init__(self,
                 env,
                 num_agents: int,
                 state_dim: int,
                 action_dim: int,
                 influence_weight: float = 0.1,
                 curriculum_steps: int = 0,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 beta: float = 0.01,
                 device: str = 'cpu'):
        
        self.env = env
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Influence parameters
        self.base_influence_weight = influence_weight
        self.curriculum_steps = curriculum_steps
        self.current_influence_weight = 0.0 if curriculum_steps > 0 else influence_weight
        
        # Create networks
        self.agents = []
        self.optimizers = []
        for _ in range(num_agents):
            agent = AttentionActorCritic(state_dim, action_dim).to(device)
            self.agents.append(agent)
            self.optimizers.append(optim.Adam(agent.parameters(), lr=lr))
        
        # Training parameters
        self.gamma = gamma
        self.beta = beta
        
        # Metrics
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
    
    def get_current_influence_weight(self):
        """Update influence weight based on curriculum."""
        if self.curriculum_steps > 0:
            progress = min(1.0, self.step_count / self.curriculum_steps)
            self.current_influence_weight = self.base_influence_weight * progress
        return self.current_influence_weight
    
    def compute_influence_with_attention(self, 
                                       agent_idx: int,
                                       states: List[torch.Tensor],
                                       neighbor_indices: List[int],
                                       attention_weights: torch.Tensor) -> float:
        """
        Compute influence reward weighted by attention scores.
        
        *** THIS IS WHERE WE USE ATTENTION SCORES IN INFLUENCE ***
        
        Formula: Influence = Σ_j attention[j] * KL(P(a_j|with_i) || P(a_j|without_i))
        
        The attention weight determines how much we care about influencing each neighbor.
        High attention = this neighbor is important to influence
        Low attention = this neighbor is less important
        """
        if not neighbor_indices or attention_weights is None:
            return 0.0
        
        total_influence = 0.0
        
        # For each neighbor j
        for idx, j in enumerate(neighbor_indices):
            # Get j's current state
            j_state = states[j]
            
            # Get j's neighbors (including agent i)
            j_neighbors_with_i = [states[k] for k in neighbor_indices if k != j]
            
            # Get j's action probabilities WITH agent i present
            with torch.no_grad():
                probs_with_i, _, _ = self.agents[j](j_state, j_neighbors_with_i)
            
            # Get j's action probabilities WITHOUT agent i
            j_neighbors_without_i = [states[k] for k in neighbor_indices if k != j and k != agent_idx]
            with torch.no_grad():
                probs_without_i, _, _ = self.agents[j](j_state, j_neighbors_without_i)
            
            # Compute KL divergence between the two distributions
            kl_divergence = F.kl_div(
                probs_without_i.log(), 
                probs_with_i, 
                reduction='sum'
            ).item()
            
            # *** ATTENTION WEIGHTING HAPPENS HERE ***
            # Weight the KL divergence by the attention score for this neighbor
            attention_weight = attention_weights[idx].item()
            weighted_influence = attention_weight * kl_divergence
            
            total_influence += weighted_influence
        
        return total_influence
    
    def train_episode(self):
        """Run one episode and train."""
        states = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Storage
        experiences = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'attention_weights': []
        }
        
        done = False
        
        while not done and episode_length < 1000:
            # Convert states to tensors
            state_tensors = [torch.FloatTensor(s.flatten()).unsqueeze(0).to(self.device) 
                           for s in states]
            
            # Get actions for all agents
            actions = []
            log_probs = []
            values = []
            all_attention_weights = []
            
            for i in range(self.num_agents):
                # Simple neighbor selection: all other agents
                neighbor_indices = [j for j in range(self.num_agents) if j != i]
                neighbor_states = [state_tensors[j] for j in neighbor_indices]
                
                # Forward pass with attention
                probs, value, attention_weights = self.agents[i](
                    state_tensors[i], 
                    neighbor_states
                )
                
                # Sample action
                dist = Categorical(probs)
                action = dist.sample()
                
                actions.append(action.item())
                log_probs.append(dist.log_prob(action))
                values.append(value)
                all_attention_weights.append((neighbor_indices, attention_weights))
            
            # Environment step
            next_states, rewards, done, info = self.env.step(actions)
            
            # Compute influence rewards using attention
            influence_rewards = []
            for i in range(self.num_agents):
                neighbor_indices, attention_weights = all_attention_weights[i]
                
                # *** ATTENTION-BASED INFLUENCE COMPUTATION ***
                influence = self.compute_influence_with_attention(
                    i, state_tensors, neighbor_indices, attention_weights
                )
                influence_rewards.append(influence)
            
            # Combine environment rewards with influence rewards
            modified_rewards = []
            for i in range(self.num_agents):
                total_reward = rewards[i] + self.get_current_influence_weight() * influence_rewards[i]
                modified_rewards.append(total_reward)
            
            # Store experience
            experiences['states'].append(state_tensors)
            experiences['actions'].append(actions)
            experiences['rewards'].append(modified_rewards)
            experiences['log_probs'].append(log_probs)
            experiences['values'].append(values)
            experiences['attention_weights'].append(all_attention_weights)
            
            # Update for next step
            episode_reward += info.get('collective_reward', sum(rewards))
            episode_length += 1
            states = next_states
        
        # Train on collected experience
        self._train_on_batch(experiences, episode_length)
        
        # Update metrics
        self.metrics['collective_rewards'].append(episode_reward)
        self.metrics['episode_lengths'].append(episode_length)
        self.step_count += episode_length
        self.episode_count += 1
        
        return episode_reward
    
    def _train_on_batch(self, experiences: Dict, episode_length: int):
        """Train agents on collected batch."""
        # For each agent
        for agent_idx in range(self.num_agents):
            # Calculate returns
            returns = []
            G = 0
            for t in reversed(range(episode_length)):
                G = experiences['rewards'][t][agent_idx] + self.gamma * G
                returns.insert(0, G)
            
            returns = torch.FloatTensor(returns).to(self.device)
            
            # Calculate loss for whole episode
            total_loss = 0
            
            for t in range(episode_length):
                log_prob = experiences['log_probs'][t][agent_idx]
                value = experiences['values'][t][agent_idx]
                
                # Advantage
                advantage = returns[t] - value.item()
                
                # Losses
                actor_loss = -log_prob * advantage
                critic_loss = F.mse_loss(value.squeeze(), returns[t])
                
                # Entropy (from current policy)
                state = experiences['states'][t][agent_idx]
                neighbor_indices, _ = experiences['attention_weights'][t][agent_idx]
                neighbor_states = [experiences['states'][t][j] for j in neighbor_indices]
                
                probs, _, _ = self.agents[agent_idx](state, neighbor_states)
                entropy = -(probs * probs.log()).sum()
                
                # Combined loss
                loss = actor_loss + 0.5 * critic_loss - self.beta * entropy
                total_loss += loss
            
            # Update
            self.optimizers[agent_idx].zero_grad()
            total_loss.backward()
            self.optimizers[agent_idx].step()
            
            self.metrics['losses'].append(total_loss.item())
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'agent_states': [agent.state_dict() for agent in self.agents],
            'optimizer_states': [opt.state_dict() for opt in self.optimizers]
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)
        
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        
        for i, (agent, opt) in enumerate(zip(self.agents, self.optimizers)):
            agent.load_state_dict(checkpoint['agent_states'][i])
            opt.load_state_dict(checkpoint['optimizer_states'][i])