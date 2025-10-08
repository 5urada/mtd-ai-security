from config import Config
from env import IDHAMEnv, AttackerType
from policy import ActorCritic
from constraints import generate_feasible_assignments
import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from pathlib import Path

def train(config: Config, attacker_type: AttackerType = AttackerType.SEQUENTIAL):
    """Train ID-HAM policy using A2C."""
    from env import IDHAMEnv
    from policy import ActorCritic
    from constraints import generate_feasible_assignments
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    env = IDHAMEnv(config, attacker_type)
    obs_dim = env.observation_dim
    
    policy = ActorCritic(obs_dim, config.N, config.B).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)
    
    # Training loop
    obs = env.reset()
    episode_rewards = []
    episode_reward = 0
    step = 0
    
    Path(config.runs_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Training ID-HAM with {attacker_type.value} attacker...")
    print(f"Observation dim: {obs_dim}, Device: {device}")
    
    while step < config.total_steps:
        # Generate feasible actions
        S = obs[:config.N].astype(np.int32)
        A_prev_flat = obs[config.N:config.N + config.N * config.B]
        A_prev = np.argmax(A_prev_flat.reshape(config.N, config.B), axis=1)
        
        feasible = generate_feasible_assignments(
            config, A_prev, S, config.K, use_ortools=False, seed=config.seed + step
        )
        
        if len(feasible) == 0:
            obs = env.reset()
            continue
        
        # Select action
        obs_tensor = torch.FloatTensor(obs).to(device)
        action, log_prob, value = policy.select_action(obs_tensor, feasible)
        
        # Environment step
        next_obs, reward, done, info = env.step(action, feasible)
        episode_reward += reward
        step += 1
        
        # Compute advantage (1-step TD)
        with torch.no_grad():
            next_obs_tensor = torch.FloatTensor(next_obs).to(device)
            _, next_value = policy(next_obs_tensor.unsqueeze(0), feasible)
            if done:
                next_value = torch.zeros_like(next_value)
            advantage = reward + config.gamma * next_value - value
        
        # Policy loss
        policy_loss = -(log_prob * advantage.detach())
        
        # Value loss
        value_loss = config.value_coef * F.mse_loss(value, torch.FloatTensor([[reward + config.gamma * next_value.item()]]).to(device))
        
        # Entropy bonus
        with torch.no_grad():
            logits, _ = policy(obs_tensor.unsqueeze(0), feasible)
            probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        entropy_loss = -config.entropy_coef * entropy
        
        # Total loss
        loss = policy_loss + value_loss + entropy_loss
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        if done:
            episode_rewards.append(episode_reward)
            if len(episode_rewards) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Step {step}/{config.total_steps} | Avg Reward: {avg_reward:.2f}")
            episode_reward = 0
            obs = env.reset()
        else:
            obs = next_obs
    
    # Save checkpoint
    ckpt_path = os.path.join(config.runs_dir, "ckpt_latest.pt")
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "config": config,
        "episode_rewards": episode_rewards
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    
    # Save summary
    summary = {
        "total_steps": config.total_steps,
        "final_avg_reward": float(np.mean(episode_rewards[-100:])) if len(episode_rewards) >= 100 else 0.0,
        "attacker_type": attacker_type.value
    }
    with open(os.path.join(config.runs_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
