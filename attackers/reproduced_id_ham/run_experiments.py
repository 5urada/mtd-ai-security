"""
Main Experiment Runner for ID-HAM - IMPROVED VERSION
Uses block-aware MDP and tracks scanned addresses
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import json
import os
from datetime import datetime

from mdp_model_improved import ImprovedHAM_MDP
from smt_constraints import generate_feasible_actions
from actor_critic import IDHAMAgent
from scanning_strategies import (
    LocalPreferenceScanning,
    SequentialScanning,
    DivideConquerScanning,
    DynamicScanning,
    NetworkScanner
)


class ExperimentRunner:
    """Run experiments to reproduce paper results"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def run_defense_performance_experiment(self,
                                          num_hosts: int,
                                          num_blocks: int,
                                          num_switches: int,
                                          block_size: int = 128,
                                          num_epochs: int = 3000,
                                          steps_per_epoch: int = 10):
        """
        Reproduce Fig. 5/6/7/8: Defense performance comparison
        
        IMPROVED: Uses block-aware MDP and tracks scanned addresses
        """
        print("\n" + "="*70)
        print("DEFENSE PERFORMANCE EXPERIMENT - IMPROVED VERSION")
        print(f"Network: {num_hosts} hosts, {num_blocks} blocks, {num_switches} switches")
        print("="*70)
        
        address_space = num_blocks * block_size
        scanning_rate = 16
        
        print(f"\nAddress space: {address_space} addresses")
        print(f"Scanning rate: {scanning_rate} addresses/period")
        print(f"Expected hit rate: ~{(num_hosts * 25) / address_space * 100:.1f}%")
        
        # Generate feasible actions using SMT
        print("\n[1/5] Generating feasible actions with SMT solver...")
        feasible_actions = generate_feasible_actions(
            num_hosts=num_hosts,
            num_blocks=num_blocks,
            num_switches=num_switches,
            max_solutions=500
        )
        
        if len(feasible_actions) == 0:
            print("ERROR: No feasible actions found! Relaxing constraints...")
            feasible_actions = self._generate_dummy_actions(num_hosts, num_blocks, 100)
        
        print(f"Generated {len(feasible_actions)} feasible actions")
        
        # Scanning strategies
        strategies = {
            'local_preference': LocalPreferenceScanning(address_space, scanning_rate),
            'sequential': SequentialScanning(address_space, scanning_rate),
            'divide_conquer': DivideConquerScanning(address_space, scanning_rate),
            'dynamic': DynamicScanning(address_space, scanning_rate)
        }
        
        results = {}
        
        # Test each scanning strategy
        for strategy_name, scanner in strategies.items():
            print(f"\n[2/5] Testing against {strategy_name} scanning...")
            
            # ID-HAM
            print("  Training ID-HAM...")
            id_ham_tsh = self._train_and_evaluate(
                'ID-HAM',
                num_hosts,
                num_blocks,
                block_size,
                feasible_actions,
                scanner,
                num_epochs,
                steps_per_epoch
            )
            
            # RHM (Improved with hypothesis test)
            print("  Evaluating RHM...")
            rhm_tsh = self._evaluate_rhm_improved(
                num_hosts,
                num_blocks,
                block_size,
                feasible_actions,
                scanner,
                num_epochs,
                steps_per_epoch
            )
            
            # FRVM (Fixed random - no adaptivity)
            print("  Evaluating FRVM...")
            frvm_tsh = self._evaluate_fixed_random(
                num_hosts,
                num_blocks,
                block_size,
                feasible_actions,
                scanner,
                num_epochs,
                steps_per_epoch
            )
            
            results[strategy_name] = {
                'ID-HAM': id_ham_tsh,
                'RHM': rhm_tsh,
                'FRVM': frvm_tsh
            }
        
        # Save results
        print("\n[5/5] Saving results...")
        self._save_results(results, f'defense_performance_{num_hosts}h_{num_blocks}b.json')
        self._plot_defense_performance(results, num_hosts, num_blocks)
        
        return results
    
    def _train_and_evaluate(self,
                           method_name: str,
                           num_hosts: int,
                           num_blocks: int,
                           block_size: int,
                           feasible_actions: List[np.ndarray],
                           scanner,
                           num_epochs: int,
                           steps_per_epoch: int) -> List[float]:
        """
        Train ID-HAM and evaluate average TSH over epochs
        
        IMPROVED: Uses ImprovedHAM_MDP with block-aware rewards
        """
        
        # Create IMPROVED MDP
        mdp = ImprovedHAM_MDP(num_hosts=num_hosts, num_blocks=num_blocks, block_size=block_size)
        
        # Create network scanner
        network_scanner = NetworkScanner(num_hosts, num_blocks, block_size, scanner)
        
        # Create agent (note: state dim is now larger!)
        agent = IDHAMAgent(
            mdp=mdp,
            feasible_actions=feasible_actions,
            learning_rate_actor=0.001,
            learning_rate_critic=0.001
        )
        
        # Track TSH and rewards
        tsh_history = []
        rewards_per_epoch = []
        
        # Training loop
        for epoch in range(num_epochs):
            state = mdp.reset()
            epoch_tsh = 0
            epoch_rewards = []
            
            for step in range(steps_per_epoch):
                # Select action
                action_idx = agent.ac_network.select_action(state)
                action = feasible_actions[action_idx]
                
                # Update network mapping
                moving_hosts = mdp.get_moving_hosts()
                network_scanner.update_address_mapping(action, moving_hosts)
                
                # Perform scanning - NOW RETURNS TWO VALUES
                scan_results, scanned_addresses = network_scanner.perform_scan()
                epoch_tsh += sum(scan_results.values())
                
                # Update MDP - NOW TAKES THREE ARGUMENTS
                next_state, reward, done = mdp.step(action, scan_results, scanned_addresses)
                
                # Track reward
                epoch_rewards.append(reward)
                
                # Update network
                agent.ac_network.update(state, action_idx, reward, next_state)
                
                state = next_state
            
            # Average TSH for this epoch
            avg_tsh = epoch_tsh / steps_per_epoch
            tsh_history.append(avg_tsh)
            rewards_per_epoch.append(epoch_rewards)
            
            # Print progress with diagnostics
            if (epoch + 1) % 500 == 0:
                recent_avg = np.mean(tsh_history[-100:])
                recent_rewards = [r for epoch_r in rewards_per_epoch[-100:] for r in epoch_r]
                reward_std = np.std(recent_rewards)
                
                # Block scan heat (top 5)
                top_heat = np.sort(mdp.block_scan_history)[-5:]
                
                print(f"    Epoch {epoch+1}/{num_epochs}")
                print(f"      Avg TSH: {recent_avg:.2f}")
                print(f"      Reward std: {reward_std:.3f}")
                print(f"      Top block heat: {top_heat}")
        
        # Verify reward variance
        all_rewards = [r for epoch_r in rewards_per_epoch for r in epoch_r]
        reward_std = np.std(all_rewards)
        
        print(f"\n  Training complete!")
        print(f"    Final TSH: {np.mean(tsh_history[-50:]):.2f}")
        print(f"    Reward std: {reward_std:.3f}")
        
        if reward_std < 1.0:
            print(f"    ⚠ WARNING: Low reward variance ({reward_std:.3f})")
            print(f"       Agent may struggle to differentiate actions")
        else:
            print(f"    ✓ Reward variance is good ({reward_std:.3f})")
        
        return tsh_history
    
    def _evaluate_rhm_improved(self,
                               num_hosts: int,
                               num_blocks: int,
                               block_size: int,
                               feasible_actions: List[np.ndarray],
                               scanner,
                               num_epochs: int,
                               steps_per_epoch: int) -> List[float]:
        """
        Evaluate RHM with simplified hypothesis test
        
        IMPROVED: Now tracks scanning patterns and avoids hot blocks
        """
        
        mdp = ImprovedHAM_MDP(num_hosts=num_hosts, num_blocks=num_blocks, block_size=block_size)
        network_scanner = NetworkScanner(num_hosts, num_blocks, block_size, scanner)
        
        tsh_history = []
        
        # Track block scan counts for hypothesis test
        cumulative_block_scans = np.zeros(num_blocks)
        test_interval = 100  # Test every 100 epochs
        
        for epoch in range(num_epochs):
            state = mdp.reset()
            epoch_tsh = 0
            
            for step in range(steps_per_epoch):
                # Every test_interval epochs, do hypothesis test
                if epoch > 0 and epoch % test_interval == 0:
                    # Simple test: are some blocks scanned much more than others?
                    if np.max(cumulative_block_scans) > 3 * np.mean(cumulative_block_scans):
                        # Non-uniform scanning detected
                        # Bias action selection away from hot blocks
                        block_heat = cumulative_block_scans / np.max(cumulative_block_scans)
                        
                        # Score actions by how much they avoid hot blocks
                        action_scores = []
                        for action in feasible_actions:
                            # Average heat of blocks used by this action
                            used_blocks = np.where(action.sum(axis=0) > 0)[0]
                            if len(used_blocks) > 0:
                                avg_heat = np.mean(block_heat[used_blocks])
                                action_scores.append(1.0 - avg_heat)  # Lower heat = higher score
                            else:
                                action_scores.append(0.5)
                        
                        # Sample action with bias towards low-heat actions
                        probs = np.array(action_scores)
                        probs = probs / np.sum(probs)
                        action_idx = np.random.choice(len(feasible_actions), p=probs)
                    else:
                        # Uniform scanning - random action
                        action_idx = np.random.randint(0, len(feasible_actions))
                else:
                    # Random action
                    action_idx = np.random.randint(0, len(feasible_actions))
                
                action = feasible_actions[action_idx]
                
                moving_hosts = mdp.get_moving_hosts()
                network_scanner.update_address_mapping(action, moving_hosts)
                
                scan_results, scanned_addresses = network_scanner.perform_scan()
                epoch_tsh += sum(scan_results.values())
                
                # Update cumulative block scans
                for addr in scanned_addresses:
                    block_id = addr // block_size
                    if 0 <= block_id < num_blocks:
                        cumulative_block_scans[block_id] += 1
            
            avg_tsh = epoch_tsh / steps_per_epoch
            tsh_history.append(avg_tsh)
        
        return tsh_history
    
    def _evaluate_fixed_random(self,
                               num_hosts: int,
                               num_blocks: int,
                               block_size: int,
                               feasible_actions: List[np.ndarray],
                               scanner,
                               num_epochs: int,
                               steps_per_epoch: int) -> List[float]:
        """Evaluate FRVM (fixed random - no learning)"""
        
        mdp = ImprovedHAM_MDP(num_hosts=num_hosts, num_blocks=num_blocks, block_size=block_size)
        network_scanner = NetworkScanner(num_hosts, num_blocks, block_size, scanner)
        
        tsh_history = []
        
        for epoch in range(num_epochs):
            state = mdp.reset()
            epoch_tsh = 0
            
            for step in range(steps_per_epoch):
                # Completely random action
                action_idx = np.random.randint(0, len(feasible_actions))
                action = feasible_actions[action_idx]
                
                moving_hosts = mdp.get_moving_hosts()
                network_scanner.update_address_mapping(action, moving_hosts)
                
                scan_results, scanned_addresses = network_scanner.perform_scan()
                epoch_tsh += sum(scan_results.values())
            
            avg_tsh = epoch_tsh / steps_per_epoch
            tsh_history.append(avg_tsh)
        
        return tsh_history
    
    def _generate_dummy_actions(self, num_hosts: int, num_blocks: int, num_actions: int) -> List[np.ndarray]:
        """Generate dummy feasible actions if SMT fails"""
        print(f"Generating {num_actions} random valid actions...")
        actions = []
        
        for _ in range(num_actions):
            action = np.zeros((num_hosts, num_blocks))
            
            used_blocks = set()
            
            for host in range(num_hosts):
                num_blocks_assign = np.random.randint(2, min(5, num_blocks // num_hosts + 1))
                
                available = [b for b in range(num_blocks) if b not in used_blocks]
                if len(available) < num_blocks_assign:
                    available = list(range(num_blocks))
                
                blocks = np.random.choice(available, num_blocks_assign, replace=False)
                action[host, blocks] = 1
                used_blocks.update(blocks)
            
            actions.append(action)
        
        print(f"✓ Generated {len(actions)} valid actions")
        return actions
    
    def _plot_defense_performance(self, results: Dict, num_hosts: int, num_blocks: int):
        """Plot defense performance comparison (reproduce Fig. 5-8)"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        strategies = ['local_preference', 'sequential', 'divide_conquer', 'dynamic']
        titles = ['Local Preference', 'Sequential', 'Divide-Conquer', 'Dynamic']
        
        for idx, (strategy, title) in enumerate(zip(strategies, titles)):
            ax = axes[idx]
            
            data = results[strategy]
            
            # Plot each method
            colors = {'ID-HAM': '#1f77b4', 'RHM': '#ff7f0e', 'FRVM': '#2ca02c'}
            for method in ['ID-HAM', 'RHM', 'FRVM']:
                tsh = data[method]
                # Smooth with moving average
                window = 50
                if len(tsh) >= window:
                    smoothed = np.convolve(tsh, np.ones(window)/window, mode='valid')
                    ax.plot(smoothed, label=method, linewidth=2, color=colors[method])
                else:
                    ax.plot(tsh, label=method, linewidth=2, color=colors[method])
            
            ax.set_xlabel('Episode', fontsize=11)
            ax.set_ylabel('Average Times of Scanning Hits', fontsize=11)
            ax.set_title(f'({chr(97+idx)}) Defense performance under\n{title} scanning', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = f'{self.results_dir}/defense_performance_{num_hosts}h_{num_blocks}b.png'
        plt.savefig(filepath, dpi=300)
        print(f"  Saved plot: {filepath}")
        plt.close()
        
        # Bar chart comparison
        self._plot_comparison_bars(results, num_hosts, num_blocks)
    
    def _plot_comparison_bars(self, results: Dict, num_hosts: int, num_blocks: int):
        """Create bar chart comparison"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        strategies = ['local_preference', 'sequential', 'divide_conquer', 'dynamic']
        labels = ['Local\nPreference', 'Sequential', 'Divide-\nConquer', 'Dynamic']
        methods = ['ID-HAM', 'RHM', 'FRVM']
        
        # Get final average TSH (last 100 episodes)
        final_tsh = {}
        for strategy in strategies:
            final_tsh[strategy] = {}
            for method in methods:
                tsh = results[strategy][method]
                final_tsh[strategy][method] = np.mean(tsh[-100:])
        
        x = np.arange(len(strategies))
        width = 0.25
        
        colors = {'ID-HAM': '#1f77b4', 'RHM': '#ff7f0e', 'FRVM': '#2ca02c'}
        for i, method in enumerate(methods):
            values = [final_tsh[s][method] for s in strategies]
            ax.bar(x + i*width, values, width, label=method, color=colors[method])
        
        ax.set_ylabel('Average Times of Scanning Hits', fontsize=12)
        ax.set_xlabel('Scanning Strategy', fontsize=12)
        ax.set_title(f'Defense Performance Comparison ({num_hosts} hosts, {num_blocks} blocks)', fontsize=13)
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = f'{self.results_dir}/defense_comparison_bars_{num_hosts}h_{num_blocks}b.png'
        plt.savefig(filepath, dpi=300)
        print(f"  Saved plot: {filepath}")
        plt.close()
    
    def _save_results(self, results: Dict, filename: str):
        """Save results to JSON"""
        serializable_results = {}
        for strategy, data in results.items():
            serializable_results[strategy] = {}
            for method, tsh in data.items():
                serializable_results[strategy][method] = [float(x) for x in tsh]
        
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"  Saved results: {filename}")


def main():
    """Run main experiments"""
    
    print("\n" + "="*70)
    print("ID-HAM ARTIFACT EVALUATION - IMPROVED VERSION")
    print("Reproducing: How to Disturb Network Reconnaissance")
    print("Zhang et al., IEEE TIFS 2023")
    print("WITH BLOCK-AWARE REWARDS AND SCANNING HISTORY")
    print("="*70)
    
    runner = ExperimentRunner(results_dir="results_improved")
    
    # Experiment 1: Small network (matches Fig. 5-6)
    print("\n\nEXPERIMENT 1: Small Network Scenario")
    print("-" * 70)
    results_small = runner.run_defense_performance_experiment(
        num_hosts=30,
        num_blocks=50,
        num_switches=5,
        block_size=128,
        num_epochs=3000,
        steps_per_epoch=10
    )
    
    # Print final results
    print("\n" + "="*70)
    print("FINAL RESULTS - SMALL NETWORK")
    print("="*70)
    for strategy, data in results_small.items():
        print(f"\n{strategy.upper().replace('_', ' ')}:")
        for method, tsh in data.items():
            final = np.mean(tsh[-100:])
            initial = np.mean(tsh[:100])
            improvement = ((initial - final) / initial) * 100
            print(f"  {method:10s}: Final={final:.2f}, Initial={initial:.2f}, Improvement={improvement:.1f}%")
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETED!")
    print("Results saved in 'results_improved/' directory")
    print("="*70)


if __name__ == '__main__':
    main()