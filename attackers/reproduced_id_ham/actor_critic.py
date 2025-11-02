"""
Advantage Actor-Critic Algorithm for Host Address Mutation
Based on Algorithm 2 and Section VI of the paper
"""

import tensorflow as tf
import numpy as np
from typing import List, Tuple
import random

# For TensorFlow 1.x compatibility
tf.compat.v1.disable_eager_execution()

class ActorCriticNetwork:
    """
    Advantage Actor-Critic implementation for HAM
    """
    
    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 learning_rate_actor: float = 0.001,
                 learning_rate_critic: float = 0.001,
                 gamma: float = 0.99,
                 beta: float = 0.01):
        """
        Initialize Actor-Critic networks
        
        Args:
            state_dim: Dimension of state space (number of hosts)
            num_actions: Number of feasible actions
            learning_rate_actor: Learning rate for actor (α_a)
            learning_rate_critic: Learning rate for critic (α_c)
            gamma: Discount factor
            beta: Entropy regularization coefficient
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.beta = beta
        
        # Reset default graph to avoid variable reuse issues
        tf.compat.v1.reset_default_graph()
        
        # Build networks
        self.sess = tf.compat.v1.Session()
        
        # Placeholders
        self.state_ph = tf.compat.v1.placeholder(tf.float32, [None, state_dim], name='state')
        self.action_ph = tf.compat.v1.placeholder(tf.int32, [None], name='action')
        self.advantage_ph = tf.compat.v1.placeholder(tf.float32, [None], name='advantage')
        self.target_value_ph = tf.compat.v1.placeholder(tf.float32, [None], name='target_value')
        
        # Actor network
        with tf.compat.v1.variable_scope('actor', reuse=tf.compat.v1.AUTO_REUSE):
            self.actor_logits, self.actor_params = self._build_actor_network()
            self.action_probs = tf.nn.softmax(self.actor_logits)
        
        # Critic network
        with tf.compat.v1.variable_scope('critic', reuse=tf.compat.v1.AUTO_REUSE):
            self.state_value, self.critic_params = self._build_critic_network()
        
        # Actor loss (Equation 14)
        action_masks = tf.one_hot(self.action_ph, self.num_actions)
        log_probs = tf.nn.log_softmax(self.actor_logits)
        selected_log_probs = tf.reduce_sum(log_probs * action_masks, axis=1)
        
        # Entropy for exploration (Equation 15)
        entropy = -tf.reduce_sum(self.action_probs * log_probs, axis=1)
        
        # Actor loss
        self.actor_loss = -tf.reduce_mean(
            self.advantage_ph * selected_log_probs + self.beta * entropy
        )
        
        # Critic loss (Equation 18)
        self.critic_loss = tf.reduce_mean(
            tf.square(self.target_value_ph - self.state_value)
        )
        
        # Optimizers
        self.actor_optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate_actor)
        self.critic_optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate_critic)
        
        # Training operations
        self.actor_train_op = self.actor_optimizer.minimize(
            self.actor_loss, var_list=self.actor_params
        )
        self.critic_train_op = self.critic_optimizer.minimize(
            self.critic_loss, var_list=self.critic_params
        )
        
        # Initialize variables
        self.sess.run(tf.compat.v1.global_variables_initializer())
    
    def _build_actor_network(self) -> Tuple:
        """
        Build actor network
        
        Returns:
            logits: Action logits
            params: Network parameters
        """
        # Hidden layers as per Table I
        hidden1 = tf.compat.v1.layers.dense(
            self.state_ph,
            256,
            activation=tf.nn.relu,
            name='actor_hidden1'
        )
        
        hidden2 = tf.compat.v1.layers.dense(
            hidden1,
            256,
            activation=tf.nn.relu,
            name='actor_hidden2'
        )
        
        # Output layer (action probabilities)
        logits = tf.compat.v1.layers.dense(
            hidden2,
            self.num_actions,
            activation=None,
            name='actor_output'
        )
        
        params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
            scope='actor'
        )
        
        return logits, params
    
    def _build_critic_network(self) -> Tuple:
        """
        Build critic network
        
        Returns:
            value: State value estimate
            params: Network parameters
        """
        # Hidden layers as per Table I
        hidden1 = tf.compat.v1.layers.dense(
            self.state_ph,
            256,
            activation=tf.nn.relu,
            name='critic_hidden1'
        )
        
        hidden2 = tf.compat.v1.layers.dense(
            hidden1,
            256,
            activation=tf.nn.relu,
            name='critic_hidden2'
        )
        
        # Output layer (state value)
        value = tf.compat.v1.layers.dense(
            hidden2,
            1,
            activation=None,
            name='critic_output'
        )
        
        value = tf.squeeze(value, axis=1)
        
        params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
            scope='critic'
        )
        
        return value, params
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action according to policy π(A_t|S_t; θ_a)
        
        Args:
            state: Current network state
        
        Returns:
            action: Selected action index
        """
        state = state.reshape(1, -1)
        action_probs = self.sess.run(
            self.action_probs,
            feed_dict={self.state_ph: state}
        )[0]
        
        # Sample action from probability distribution
        action = np.random.choice(self.num_actions, p=action_probs)
        return action
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Get state value V(S_t; θ_c)
        
        Args:
            state: Current network state
        
        Returns:
            value: Estimated state value
        """
        state = state.reshape(1, -1)
        value = self.sess.run(
            self.state_value,
            feed_dict={self.state_ph: state}
        )[0]
        return value
    
    def update(self, 
               state: np.ndarray,
               action: int,
               reward: float,
               next_state: np.ndarray) -> Tuple[float, float]:
        """
        Update actor and critic networks
        
        Args:
            state: Current state S_t
            action: Action taken A_t
            reward: Reward R_t
            next_state: Next state S_{t+1}
        
        Returns:
            actor_loss: Actor network loss
            critic_loss: Critic network loss
        """
        state = state.reshape(1, -1)
        next_state = next_state.reshape(1, -1)
        
        # Get state values
        current_value = self.get_value(state)
        next_value = self.get_value(next_state)
        
        # Calculate advantage (Equation 16)
        # A(A_t, S_t) = R_t + γV(S_{t+1}) - V(S_t)
        advantage = reward + self.gamma * next_value - current_value
        
        # Target value for critic
        target_value = reward + self.gamma * next_value
        
        # Update critic network
        _, critic_loss = self.sess.run(
            [self.critic_train_op, self.critic_loss],
            feed_dict={
                self.state_ph: state,
                self.target_value_ph: [target_value]
            }
        )
        
        # Update actor network
        _, actor_loss = self.sess.run(
            [self.actor_train_op, self.actor_loss],
            feed_dict={
                self.state_ph: state,
                self.action_ph: [action],
                self.advantage_ph: [advantage]
            }
        )
        
        return actor_loss, critic_loss
    
    def save_model(self, path: str):
        """Save model weights"""
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, path)
    
    def load_model(self, path: str):
        """Load model weights"""
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, path)


class IDHAMAgent:
    """
    Intelligence-Driven Host Address Mutation Agent
    Implements Algorithm 2 from the paper
    """
    
    def __init__(self,
                 mdp,
                 feasible_actions: List[np.ndarray],
                 T_AS: int = 64,
                 learning_rate_actor: float = 0.001,
                 learning_rate_critic: float = 0.001,
                 gamma: float = 0.99,
                 beta: float = 0.01):
        """
        Initialize ID-HAM agent
        
        Args:
            mdp: MDP environment
            feasible_actions: List of feasible address block allocations
            T_AS: Address shuffling interval
            learning_rate_actor: Actor learning rate
            learning_rate_critic: Critic learning rate
            gamma: Discount factor
            beta: Entropy coefficient
        """
        self.mdp = mdp
        self.feasible_actions = feasible_actions
        self.T_AS = T_AS
        
        # Initialize actor-critic network
        self.ac_network = ActorCriticNetwork(
            state_dim=mdp.get_state_dim(),
            num_actions=len(feasible_actions),
            learning_rate_actor=learning_rate_actor,
            learning_rate_critic=learning_rate_critic,
            gamma=gamma,
            beta=beta
        )
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
    
    def train(self, num_epochs: int, steps_per_epoch: int) -> dict:
        """
        Train ID-HAM agent (Algorithm 2)
        
        Args:
            num_epochs: Number of training epochs (U in Algorithm 2)
            steps_per_epoch: Number of steps per epoch (T in Algorithm 2)
        
        Returns:
            Training statistics
        """
        print(f"Training ID-HAM Agent")
        print(f"Epochs: {num_epochs}, Steps per epoch: {steps_per_epoch}")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            # Reset environment (Line 5)
            state = self.mdp.reset()
            
            epoch_reward = 0
            actor_losses = []
            critic_losses = []
            
            for step in range(steps_per_epoch):
                # Select action (Line 7)
                action_idx = self.ac_network.select_action(state)
                action = self.feasible_actions[action_idx]
                
                # Execute action (Lines 8-13)
                # Simulate mutation and scanning
                scan_results = self._simulate_scanning(action)
                
                # Observe reward and next state (Lines 14-15)
                next_state, reward, done = self.mdp.step(action, scan_results)
                
                # Update networks (Lines 16-20)
                actor_loss, critic_loss = self.ac_network.update(
                    state, action_idx, reward, next_state
                )
                
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                epoch_reward += reward
                
                # Update state (Line 21)
                state = next_state
            
            # Store statistics
            self.episode_rewards.append(epoch_reward)
            self.episode_lengths.append(steps_per_epoch)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_actor_loss = np.mean(actor_losses)
                avg_critic_loss = np.mean(critic_losses)
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Actor Loss: {avg_actor_loss:.4f} | "
                      f"Critic Loss: {avg_critic_loss:.4f}")
        
        return {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths
        }
    
    def _simulate_scanning(self, action: np.ndarray) -> dict:
        """
        Simulate adversarial scanning
        Returns dict of host_id: num_hits
        """
        # Simple simulation: random hosts get scanned
        scan_results = {}
        num_scanned = np.random.randint(0, 5)
        
        for _ in range(num_scanned):
            host_id = np.random.randint(0, self.mdp.num_hosts)
            scan_results[host_id] = scan_results.get(host_id, 0) + 1
        
        return scan_results


if __name__ == '__main__':
    from mdp_model import HAM_MDP
    
    print("Testing Actor-Critic for HAM")
    print("=" * 50)
    
    # Create MDP
    mdp = HAM_MDP(num_hosts=30, num_blocks=50)
    
    # Create dummy feasible actions (replace with SMT-generated actions)
    num_actions = 100
    feasible_actions = []
    for _ in range(num_actions):
        action = np.zeros((30, 50))
        # Randomly assign blocks
        for host in range(30):
            num_blocks = np.random.randint(1, 5)
            blocks = np.random.choice(50, num_blocks, replace=False)
            action[host, blocks] = 1
        feasible_actions.append(action)
    
    # Create agent
    agent = IDHAMAgent(mdp, feasible_actions)
    
    # Train for a few epochs
    print("\nTraining for 20 epochs...")
    stats = agent.train(num_epochs=20, steps_per_epoch=10)
    
    print(f"\nFinal average reward: {np.mean(stats['rewards'][-10:]):.2f}")