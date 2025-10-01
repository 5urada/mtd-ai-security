import pytest
import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from env import IDHAMEnv, AttackerType
from constraints import generate_feasible_assignments, _verify_constraints
from policy import ActorCritic
from metrics import compute_flip_sensitivity

def test_config_defaults():
    """Test default configuration values."""
    cfg = Config()
    assert cfg.N == 24
    assert cfg.B == 6
    assert cfg.T == 64
    assert len(cfg.capacities) == cfg.B

def test_env_reset():
    """Test environment reset."""
    cfg = Config(seed=42)
    env = IDHAMEnv(cfg)
    obs = env.reset()
    
    assert obs.shape[0] == env.observation_dim
    assert env.S is not None
    assert env.A_prev is not None
    assert env.episode_step == 0

def test_env_step():
    """Test environment step without NaNs."""
    cfg = Config(seed=42)
    env = IDHAMEnv(cfg, AttackerType.STATIC)
    obs = env.reset()
    
    # Generate feasible actions
    S = obs[:cfg.N].astype(np.int32)
    A_prev = np.argmax(obs[cfg.N:cfg.N + cfg.N * cfg.B].reshape(cfg.N, cfg.B), axis=1)
    feasible = generate_feasible_assignments(cfg, A_prev, S, cfg.K, use_ortools=False, seed=42)
    
    assert len(feasible) > 0, "Should generate at least one feasible action"
    
    next_obs, reward, done, info = env.step(0, feasible)
    
    assert not np.any(np.isnan(next_obs)), "Observation should not contain NaNs"
    assert not np.isnan(reward), "Reward should not be NaN"
    assert isinstance(done, bool)

def test_constraint_generation():
    """Test feasible action generation satisfies constraints."""
    cfg = Config(N=12, B=4, seed=42)
    A_prev = np.array([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3])
    S = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    
    feasible = generate_feasible_assignments(cfg, A_prev, S, K=5, use_ortools=False, seed=42)
    
    assert len(feasible) > 0, "Should generate at least one feasible assignment"
    
    for A_new in feasible:
        assert _verify_constraints(cfg, A_prev, A_new), "Generated assignment should satisfy constraints"

def test_policy_forward():
    """Test policy forward pass."""
    cfg = Config()
    policy = ActorCritic(obs_dim=cfg.N + cfg.N * cfg.B + cfg.N)
    
    obs = torch.randn(1, cfg.N + cfg.N * cfg.B + cfg.N)
    K = 8
    
    logits, value = policy(obs, K)
    
    assert logits.shape == (1, K)
    assert value.shape == (1, 1)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(value).any()

def test_training_smoke():
    """Smoke test training for a few steps."""
    from train import train
    
    cfg = Config(N=10, B=3, T=8, total_steps=100, seed=42)
    cfg.runs_dir = "test_runs"
    
    try:
        train(cfg, AttackerType.STATIC)
        assert os.path.exists(os.path.join(cfg.runs_dir, "ckpt_latest.pt"))
    finally:
        # Cleanup
        import shutil
        if os.path.exists(cfg.runs_dir):
            shutil.rmtree(cfg.runs_dir)

def test_sensitivity_metrics_range():
    """Test sensitivity metrics produce values in valid ranges."""
    cfg = Config(N=8, B=3, seed=42)
    env = IDHAMEnv(cfg)
    policy = ActorCritic(env.observation_dim)
    
    # Collect a few states
    states = []
    obs = env.reset()
    for _ in range(5):
        states.append(obs.copy())
        S = obs[:cfg.N].astype(np.int32)
        A_prev = np.argmax(obs[cfg.N:cfg.N + cfg.N * cfg.B].reshape(cfg.N, cfg.B), axis=1)
        feasible = generate_feasible_assignments(cfg, A_prev, S, 4, use_ortools=False, seed=42)
        if feasible:
            obs, _, done, _ = env.step(0, feasible)
            if done:
                obs = env.reset()
    
    device = torch.device("cpu")
    results = compute_flip_sensitivity(policy, states, cfg, device)
    
    # Check ranges
    assert 0 <= results["flip_rate_overall"] <= 1
    assert results["kl_mean"] >= 0
    assert 0 <= results["tv_mean"] <= 1
    assert 0 <= results["assignment_hamming_mean"] <= 1

