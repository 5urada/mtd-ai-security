import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from typing import Tuple, List
import numpy as np

class ActorCritic(nn.Module):
    """Actor-Critic network for ID-HAM with candidate-aware scoring."""
    
    def __init__(self, obs_dim: int, N: int, B: int, hidden_dim: int = 256):
        super().__init__()
        self.N = N
        self.B = B
        
        # Shared encoder for observation
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Candidate encoder: encodes each assignment vector
        self.candidate_encoder = nn.Sequential(
            nn.Linear(N, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Actor: combines observation features with candidate features
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Critic: value function
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs: torch.Tensor, candidates: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that scores each candidate assignment.
        
        Args:
            obs: Observation tensor (batch, obs_dim)
            candidates: List of K assignment arrays, each shape (N,)
        
        Returns:
            logits: (batch, K) - score for each candidate
            value: (batch, 1) - state value
        """
        batch_size = obs.shape[0]
        K = len(candidates)
        
        # Encode observation
        obs_features = self.encoder(obs)  # (batch, hidden_dim)
        
        # Encode each candidate assignment
        candidate_tensors = torch.FloatTensor(np.array(candidates)).to(obs.device)  # (K, N)
        candidate_features = self.candidate_encoder(candidate_tensors.float())  # (K, hidden_dim//2)
        
        # Combine observation with each candidate
        obs_expanded = obs_features.unsqueeze(1).expand(batch_size, K, -1)  # (batch, K, hidden_dim)
        cand_expanded = candidate_features.unsqueeze(0).expand(batch_size, K, -1)  # (batch, K, hidden_dim//2)
        
        combined = torch.cat([obs_expanded, cand_expanded], dim=-1)  # (batch, K, hidden_dim + hidden_dim//2)
        
        # Score each candidate
        logits = self.actor(combined).squeeze(-1)  # (batch, K)
        
        # Compute value
        value = self.critic(obs_features)  # (batch, 1)
        
        return logits, value
    
    def select_action(self, obs: torch.Tensor, candidates: List[np.ndarray]) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action using current policy."""
        logits, value = self.forward(obs.unsqueeze(0), candidates)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value