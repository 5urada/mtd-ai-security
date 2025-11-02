"""
Adversarial Scanning Strategies
Implements scanning strategies from Section III-A
"""

import numpy as np
from typing import List, Dict, Set
import random

class ScanningStrategy:
    """Base class for scanning strategies"""
    
    def __init__(self, address_space_size: int, scanning_rate: int):
        """
        Initialize scanner
        
        Args:
            address_space_size: Size of address space to scan
            scanning_rate: Number of addresses to scan per period (η)
        """
        self.address_space_size = address_space_size
        self.scanning_rate = scanning_rate
        self.scanned_addresses = set()
        
    def scan(self, active_hosts: Dict[int, str]) -> Dict[int, int]:
        """
        Perform scanning
        
        Args:
            active_hosts: Dict mapping address to host_id
        
        Returns:
            Dict mapping host_id to number of successful scans
        """
        raise NotImplementedError


class LocalPreferenceScanning(ScanningStrategy):
    """
    Local Preference Scanning Strategy
    
    Scans randomly, but prefers addresses near successfully scanned hosts
    """
    
    def __init__(self, address_space_size: int, scanning_rate: int, locality_prob: float = 0.7):
        super().__init__(address_space_size, scanning_rate)
        self.locality_prob = locality_prob
        self.successful_scans = []  # Track successful scan locations
        self.locality_range = 256  # Scan within this range of successful hits
        
    def scan(self, active_hosts: Dict[int, str]) -> Dict[int, int]:
        """Perform local preference scanning"""
        scan_results = {}
        
        for _ in range(self.scanning_rate):
            # With probability locality_prob, scan near previous hits
            if self.successful_scans and random.random() < self.locality_prob:
                # Choose a previous hit and scan nearby
                base_addr = random.choice(self.successful_scans)
                offset = random.randint(-self.locality_range, self.locality_range)
                target_addr = max(0, min(self.address_space_size - 1, base_addr + offset))
            else:
                # Random scanning
                target_addr = random.randint(0, self.address_space_size - 1)
            
            self.scanned_addresses.add(target_addr)
            
            # Check if hit
            if target_addr in active_hosts:
                host_id = active_hosts[target_addr]
                scan_results[host_id] = scan_results.get(host_id, 0) + 1
                self.successful_scans.append(target_addr)
        
        return scan_results


class SequentialScanning(ScanningStrategy):
    """
    Sequential Scanning Strategy
    
    Scans addresses sequentially, periodically reselecting start address
    """
    
    def __init__(self, address_space_size: int, scanning_rate: int, reselect_period: int = 1000):
        super().__init__(address_space_size, scanning_rate)
        self.current_address = random.randint(0, address_space_size - 1)
        self.reselect_period = reselect_period
        self.scan_count = 0
        
    def scan(self, active_hosts: Dict[int, str]) -> Dict[int, int]:
        """Perform sequential scanning"""
        scan_results = {}
        
        for _ in range(self.scanning_rate):
            target_addr = self.current_address
            self.scanned_addresses.add(target_addr)
            
            # Check if hit
            if target_addr in active_hosts:
                host_id = active_hosts[target_addr]
                scan_results[host_id] = scan_results.get(host_id, 0) + 1
            
            # Move to next address
            self.current_address = (self.current_address + 1) % self.address_space_size
            self.scan_count += 1
            
            # Periodically reselect starting address
            if self.scan_count % self.reselect_period == 0:
                self.current_address = random.randint(0, self.address_space_size - 1)
        
        return scan_results


class DivideConquerScanning(ScanningStrategy):
    """
    Divide-Conquer Scanning Strategy
    
    Uses multiple controlled hosts to scan in parallel
    Scanned hosts may become controlled with probability ρ
    """
    
    def __init__(self, 
                 address_space_size: int,
                 scanning_rate: int,
                 num_initial_attackers: int = 3,
                 infection_prob: float = 0.1):
        super().__init__(address_space_size, scanning_rate)
        self.num_attackers = num_initial_attackers
        self.infection_prob = infection_prob  # ρ in paper
        self.controlled_hosts = set()
        
        # Each attacker has its own sequential scanner
        self.scanners = [
            SequentialScanning(address_space_size, scanning_rate // num_initial_attackers)
            for _ in range(num_initial_attackers)
        ]
    
    def scan(self, active_hosts: Dict[int, str]) -> Dict[int, int]:
        """Perform divide-conquer scanning"""
        scan_results = {}
        
        # Each controlled host scans independently
        for scanner in self.scanners:
            results = scanner.scan(active_hosts)
            
            # Merge results
            for host_id, hits in results.items():
                scan_results[host_id] = scan_results.get(host_id, 0) + hits
                
                # May infect scanned host
                if random.random() < self.infection_prob:
                    if host_id not in self.controlled_hosts:
                        self.controlled_hosts.add(host_id)
                        # Add new scanner for this host
                        new_scanner = SequentialScanning(
                            self.address_space_size,
                            self.scanning_rate // self.num_attackers
                        )
                        self.scanners.append(new_scanner)
                        self.num_attackers += 1
        
        return scan_results


class DynamicScanning(ScanningStrategy):
    """
    Dynamic Scanning Strategy
    
    Randomly switches between different scanning strategies
    """
    
    def __init__(self, 
                 address_space_size: int,
                 scanning_rate: int,
                 strategy_switch_period: int = 500):
        super().__init__(address_space_size, scanning_rate)
        self.strategy_switch_period = strategy_switch_period
        self.scan_count = 0
        
        # Available strategies
        self.strategies = [
            LocalPreferenceScanning(address_space_size, scanning_rate),
            SequentialScanning(address_space_size, scanning_rate),
            DivideConquerScanning(address_space_size, scanning_rate)
        ]
        
        self.current_strategy = random.choice(self.strategies)
        
    def scan(self, active_hosts: Dict[int, str]) -> Dict[int, int]:
        """Perform dynamic scanning"""
        # Periodically switch strategy
        if self.scan_count % self.strategy_switch_period == 0:
            self.current_strategy = random.choice(self.strategies)
            print(f"Switched to {self.current_strategy.__class__.__name__}")
        
        self.scan_count += 1
        
        # Use current strategy
        return self.current_strategy.scan(active_hosts)


class NetworkScanner:
    """
    Simulate network reconnaissance with HAM defense
    """
    
    def __init__(self,
                 num_hosts: int,
                 address_space_size: int,
                 scanning_strategy: ScanningStrategy):
        """
        Initialize network scanner
        
        Args:
            num_hosts: Number of hosts in network
            address_space_size: Size of address space
            scanning_strategy: Scanning strategy to use
        """
        self.num_hosts = num_hosts
        self.address_space_size = address_space_size
        self.scanner = scanning_strategy
        
        # Current vIP to host mapping
        self.vip_to_host = {}
        
    def update_address_mapping(self, allocations: np.ndarray, moving_hosts: List[int]):
        """
        Update virtual IP to host mapping after mutation
        
        Args:
            allocations: Address block allocations (num_hosts x num_blocks)
            moving_hosts: List of moving host indices
        """
        self.vip_to_host = {}
        
        # For each moving host, randomly select vIP from assigned blocks
        for host_id in moving_hosts:
            assigned_blocks = np.where(allocations[host_id] == 1)[0]
            if len(assigned_blocks) > 0:
                # Randomly choose a block
                block_id = random.choice(assigned_blocks)
                # Randomly choose address in block (simplified)
                block_start = block_id * 128  # Assuming block_size = 128
                vip = block_start + random.randint(0, 127)
                self.vip_to_host[vip] = host_id
        
        # Static hosts keep their addresses (simplified - use fixed addresses)
        # In reality, static hosts would have pre-assigned addresses
    
    def perform_scan(self) -> Dict[int, int]:
        """
        Perform scanning and return results
        
        Returns:
            Dict mapping host_id to number of successful scans
        """
        return self.scanner.scan(self.vip_to_host)
    
    def get_scan_statistics(self) -> dict:
        """Get scanning statistics"""
        total_scanned = len(self.scanner.scanned_addresses)
        coverage = total_scanned / self.address_space_size * 100
        
        return {
            'total_scanned': total_scanned,
            'coverage': coverage,
            'unique_addresses': len(self.scanner.scanned_addresses)
        }


if __name__ == '__main__':
    print("Testing Scanning Strategies")
    print("=" * 50)
    
    # Setup
    address_space = 65536  # 2^16
    num_hosts = 30
    scanning_rate = 16  # 16 hosts/ΔT from Table I
    
    # Create active host mapping
    active_hosts = {}
    for i in range(num_hosts):
        addr = random.randint(0, address_space - 1)
        active_hosts[addr] = i
    
    # Test each strategy
    strategies = [
        ("Local Preference", LocalPreferenceScanning(address_space, scanning_rate)),
        ("Sequential", SequentialScanning(address_space, scanning_rate)),
        ("Divide-Conquer", DivideConquerScanning(address_space, scanning_rate)),
        ("Dynamic", DynamicScanning(address_space, scanning_rate))
    ]
    
    for name, strategy in strategies:
        print(f"\n{name} Scanning:")
        
        # Simulate 10 scan periods
        total_hits = 0
        for period in range(10):
            results = strategy.scan(active_hosts)
            hits = sum(results.values())
            total_hits += hits
        
        print(f"  Total hits: {total_hits}")
        print(f"  Average hits/period: {total_hits/10:.1f}")
        print(f"  Scanned addresses: {len(strategy.scanned_addresses)}")
