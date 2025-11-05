"""
Diagnostic script to understand Divide-Conquer scanning behavior
"""

import numpy as np
import matplotlib.pyplot as plt
from scanning_strategies import DivideConquerScanning, NetworkScanner

def test_divide_conquer_growth():
    """Test how the number of attackers grows over time"""
    
    print("="*70)
    print("DIVIDE-CONQUER DIAGNOSTIC TEST")
    print("="*70)
    
    # Setup
    address_space = 6400
    scanning_rate = 16
    num_hosts = 30
    num_blocks = 50
    block_size = 128
    
    # Test different infection probabilities
    infection_probs = [0.1, 0.01, 0.001, 0.0001]
    
    for infection_prob in infection_probs:
        print(f"\n{'='*70}")
        print(f"Testing infection_prob = {infection_prob}")
        print(f"{'='*70}")
        
        # Create scanner
        scanner = DivideConquerScanning(
            address_space, 
            scanning_rate,
            num_initial_attackers=3,
            infection_prob=infection_prob
        )
        
        # Create network scanner
        net_scanner = NetworkScanner(num_hosts, num_blocks, block_size, scanner)
        
        # Create dummy allocation
        allocation = np.zeros((num_hosts, num_blocks))
        for i in range(num_hosts):
            blocks = np.random.choice(num_blocks, 3, replace=False)
            allocation[i, blocks] = 1
        
        # Update mapping
        moving_hosts = list(range(num_hosts))
        net_scanner.update_address_mapping(allocation, moving_hosts)
        
        # Simulate 100 epochs (1000 scans total)
        num_epochs = 100
        steps_per_epoch = 10
        
        attacker_history = []
        tsh_history = []
        
        for epoch in range(num_epochs):
            epoch_tsh = 0
            
            for step in range(steps_per_epoch):
                # Perform scan
                scan_results = net_scanner.perform_scan()
                epoch_tsh += sum(scan_results.values())
            
            # Record statistics
            avg_tsh = epoch_tsh / steps_per_epoch
            tsh_history.append(avg_tsh)
            attacker_history.append(scanner.num_attackers)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}: "
                      f"Attackers={scanner.num_attackers:4d}, "
                      f"Controlled={len(scanner.controlled_hosts):4d}, "
                      f"TSH={avg_tsh:.2f}")
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(attacker_history)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Number of Attackers')
        ax1.set_title(f'Attacker Growth (infection_prob={infection_prob})')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(tsh_history)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('TSH per Epoch')
        ax2.set_title(f'TSH Over Time (infection_prob={infection_prob})')
        ax2.axhline(y=1.5, color='r', linestyle='--', label='Expected (~1.5)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'divide_conquer_diagnostic_prob_{infection_prob}.png', dpi=150)
        plt.close()
        
        print(f"\n  Final state:")
        print(f"    Total attackers: {scanner.num_attackers}")
        print(f"    Controlled hosts: {len(scanner.controlled_hosts)}")
        print(f"    Average TSH (last 20): {np.mean(tsh_history[-20:]):.2f}")
        print(f"    Expected TSH: ~1.5")
        print(f"    Saved plot: divide_conquer_diagnostic_prob_{infection_prob}.png")


def test_single_epoch_detail():
    """Detailed analysis of a single epoch"""
    
    print(f"\n{'='*70}")
    print("SINGLE EPOCH DETAILED ANALYSIS")
    print(f"{'='*70}")
    
    address_space = 6400
    scanning_rate = 16
    num_hosts = 30
    
    # Test with infection_prob = 0.01
    scanner = DivideConquerScanning(
        address_space, 
        scanning_rate,
        num_initial_attackers=3,
        infection_prob=0.01
    )
    
    # Create simple host mapping
    active_hosts = {}
    for i in range(num_hosts):
        # Each host gets 25 addresses
        for j in range(25):
            addr = i * 100 + j  # Spread them out
            active_hosts[addr] = i
    
    print(f"\nInitial state:")
    print(f"  Attackers: {scanner.num_attackers}")
    print(f"  Active hosts: {len(active_hosts)}")
    print(f"  Total host addresses: {len(active_hosts)}")
    
    # Perform 10 scans (1 epoch)
    total_hits = 0
    for step in range(10):
        results = scanner.scan(active_hosts)
        hits = sum(results.values())
        total_hits += hits
        print(f"\n  Step {step+1}: {hits} hits, "
              f"Attackers={scanner.num_attackers}, "
              f"Controlled={len(scanner.controlled_hosts)}")
        
        if results:
            print(f"    Hit hosts: {list(results.keys())[:5]}...")
    
    print(f"\nAfter 1 epoch (10 steps):")
    print(f"  Total hits: {total_hits}")
    print(f"  Average TSH: {total_hits / 10:.2f}")
    print(f"  Final attackers: {scanner.num_attackers}")
    print(f"  New infections: {len(scanner.controlled_hosts)}")


def analyze_scanner_distribution():
    """Analyze how scanners are distributed"""
    
    print(f"\n{'='*70}")
    print("SCANNER DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    
    address_space = 6400
    scanning_rate = 16
    
    scanner = DivideConquerScanning(
        address_space,
        scanning_rate,
        num_initial_attackers=3,
        infection_prob=0.01
    )
    
    print(f"\nInitial scanners:")
    for i, s in enumerate(scanner.scanners):
        print(f"  Scanner {i}: rate={s.scanning_rate}, start={s.current_address}")
    
    # Simulate infections
    active_hosts = {100: 0, 200: 1, 300: 2}
    
    for round in range(5):
        print(f"\nRound {round + 1}:")
        results = scanner.scan(active_hosts)
        print(f"  Hits: {results}")
        print(f"  Total attackers: {scanner.num_attackers}")
        print(f"  Total scanners: {len(scanner.scanners)}")
        
        if len(scanner.scanners) > 3:
            print(f"  New scanner rates: {[s.scanning_rate for s in scanner.scanners[-3:]]}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("DIVIDE-CONQUER DEBUGGING SUITE")
    print("="*70)
    
    # Test 1: Growth over time with different infection probabilities
    test_divide_conquer_growth()
    
    # Test 2: Single epoch detail
    test_single_epoch_detail()
    
    # Test 3: Scanner distribution
    analyze_scanner_distribution()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("Check the generated PNG files for visualizations")
    print("="*70)