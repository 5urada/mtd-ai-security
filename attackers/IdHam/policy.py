import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from config import Config

class ActorCritic(nn.Module):
    """Actor-Critic network for ID-HAM."""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (outputs logits over K candidates)
        self.actor = nn.Linear(hidden_dim, 1)  # Will be applied per candidate
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns: (logits over K actions, value)
        """
        features = self.encoder(obs)
        
        # For K candidates, we use same features to score each
        logits = self.actor(features).expand(-1, K)  # Shape: (batch, K)
        value = self.critic(features)  # Shape: (batch, 1)
        
        return logits, value
    
    def select_action(self, obs: torch.Tensor, K: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action using current policy."""
        logits, value = self.forward(obs.unsqueeze(0), K)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value
