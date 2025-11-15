"""
Adversarial Scanning Strategies - IMPROVED VERSION
Now tracks scanned addresses for block history in MDP
"""

import numpy as np
from typing import List, Dict, Set, Tuple
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
        # NEW: Track addresses scanned in current period
        self.current_period_scans = []
        
    def scan(self, active_hosts: Dict[int, str]) -> Dict[int, int]:
        """
        Perform scanning
        
        Args:
            active_hosts: Dict mapping address to host_id
        
        Returns:
            Dict mapping host_id to number of successful scans
        """
        raise NotImplementedError
    
    def reset_period(self):
        """Reset per-period tracking"""
        self.current_period_scans = []


class LocalPreferenceScanning(ScanningStrategy):
    """
    Local Preference Scanning Strategy
    
    Scans randomly, but prefers addresses near successfully scanned hosts
    """
    
    def __init__(self, address_space_size: int, scanning_rate: int, locality_prob: float = 0.7):
        super().__init__(address_space_size, scanning_rate)
        self.locality_prob = locality_prob
        self.successful_scans = []  # Track successful scan locations
        self.locality_range = min(256, address_space_size // 4)
        
    def scan(self, active_hosts: Dict[int, str]) -> Dict[int, int]:
        """Perform local preference scanning"""
        scan_results = {}
        self.current_period_scans = []
        
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
            self.current_period_scans.append(target_addr)
            
            # Check if hit
            if target_addr in active_hosts:
                host_id = active_hosts[target_addr]
                scan_results[host_id] = scan_results.get(host_id, 0) + 1
                self.successful_scans.append(target_addr)
                if len(self.successful_scans) > 100:
                    self.successful_scans.pop(0)
        
        return scan_results


class SequentialScanning(ScanningStrategy):
    """
    Sequential Scanning Strategy
    
    Scans addresses sequentially, periodically reselecting start address
    """
    
    def __init__(self, address_space_size: int, scanning_rate: int, reselect_period: int = 160):
        super().__init__(address_space_size, scanning_rate)
        self.current_address = random.randint(0, address_space_size - 1)
        self.reselect_period = reselect_period
        self.scan_count = 0
        
    def scan(self, active_hosts: Dict[int, str]) -> Dict[int, int]:
        """Perform sequential scanning"""
        scan_results = {}
        self.current_period_scans = []
        
        for _ in range(self.scanning_rate):
            target_addr = self.current_address
            self.scanned_addresses.add(target_addr)
            self.current_period_scans.append(target_addr)
            
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
        self.infection_prob = infection_prob
        self.controlled_hosts = set()
        
        # Each attacker has its own sequential scanner
        self.scanners = [
            SequentialScanning(address_space_size, scanning_rate // num_initial_attackers)
            for _ in range(num_initial_attackers)
        ]
    
    def scan(self, active_hosts: Dict[int, str]) -> Dict[int, int]:
        """Perform divide-conquer scanning"""
        scan_results = {}
        self.current_period_scans = []
        
        # CRITICAL FIX: Maintain constant total scanning rate
        if len(self.scanners) <= self.scanning_rate:
            rate_per_scanner = self.scanning_rate // len(self.scanners)
            
            for scanner in self.scanners:
                original_rate = scanner.scanning_rate
                scanner.scanning_rate = max(1, rate_per_scanner)
                
                results = scanner.scan(active_hosts)
                
                # Collect scanned addresses
                self.current_period_scans.extend(scanner.current_period_scans)
                
                scanner.scanning_rate = original_rate
                
                # Merge results
                for host_id, hits in results.items():
                    scan_results[host_id] = scan_results.get(host_id, 0) + hits
                    
                    # May infect scanned host
                    if random.random() < self.infection_prob:
                        if host_id not in self.controlled_hosts:
                            self.controlled_hosts.add(host_id)
                            new_scanner = SequentialScanning(
                                self.address_space_size,
                                1
                            )
                            self.scanners.append(new_scanner)
                            self.num_attackers += 1
        else:
            # Too many scanners: only activate scanning_rate number of them
            for i, scanner in enumerate(self.scanners):
                if i < self.scanning_rate:
                    original_rate = scanner.scanning_rate
                    scanner.scanning_rate = 1
                    
                    results = scanner.scan(active_hosts)
                    
                    # Collect scanned addresses
                    self.current_period_scans.extend(scanner.current_period_scans)
                    
                    scanner.scanning_rate = original_rate
                    
                    # Merge results
                    for host_id, hits in results.items():
                        scan_results[host_id] = scan_results.get(host_id, 0) + hits
                        
                        # May infect scanned host
                        if random.random() < self.infection_prob:
                            if host_id not in self.controlled_hosts:
                                self.controlled_hosts.add(host_id)
                                new_scanner = SequentialScanning(
                                    self.address_space_size,
                                    1
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
        
        self.scan_count += 1
        
        # Use current strategy
        results = self.current_strategy.scan(active_hosts)
        
        # Copy scanned addresses
        self.current_period_scans = self.current_strategy.current_period_scans.copy()
        
        return results


class NetworkScanner:
    """
    Simulate network reconnaissance with HAM defense - IMPROVED VERSION
    
    Now tracks scanned addresses for block-aware MDP
    """
    
    def __init__(self,
                 num_hosts: int,
                 num_blocks: int,
                 block_size: int,
                 scanning_strategy: ScanningStrategy):
        """
        Initialize network scanner
        
        Args:
            num_hosts: Number of hosts in network
            num_blocks: Number of IP blocks
            block_size: Size of each block  
            scanning_strategy: Scanning strategy to use
        """
        self.num_hosts = num_hosts
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.scanner = scanning_strategy
        
        # Address space based on blocks
        self.address_space_size = num_blocks * block_size
        
        # Current vIP to host mapping
        self.vip_to_host = {}
        
        # Track which hosts are currently active
        self.active_host_addresses = {}
        
    def update_address_mapping(self, allocations: np.ndarray, moving_hosts: List[int]):
        """
        Update virtual IP to host mapping after mutation
        
        Each host gets ~25 addresses to match paper's hit rates
        
        Args:
            allocations: Address block allocations (num_hosts x num_blocks)
            moving_hosts: List of moving host indices
        """
        self.vip_to_host = {}
        self.active_host_addresses = {}
        
        # Moving hosts get 25 addresses
        for host_id in moving_hosts:
            assigned_blocks = np.where(allocations[host_id] == 1)[0]
            
            if len(assigned_blocks) > 0:
                addresses_per_host = 25
                
                self.active_host_addresses[host_id] = set()
                
                for _ in range(addresses_per_host):
                    block_id = random.choice(assigned_blocks)
                    block_start = block_id * self.block_size
                    offset = random.randint(0, self.block_size - 1)
                    vip = block_start + offset
                    
                    self.vip_to_host[vip] = host_id
                    self.active_host_addresses[host_id].add(vip)
        
        # Static hosts also get 25 addresses
        static_hosts = [h for h in range(self.num_hosts) if h not in moving_hosts]
        for host_id in static_hosts:
            assigned_blocks = np.where(allocations[host_id] == 1)[0]
            
            if len(assigned_blocks) > 0:
                addresses_per_host = 25
                
                self.active_host_addresses[host_id] = set()
                
                for _ in range(addresses_per_host):
                    block_id = random.choice(assigned_blocks)
                    block_start = block_id * self.block_size
                    offset = random.randint(0, self.block_size - 1)
                    vip = block_start + offset
                    
                    self.vip_to_host[vip] = host_id
                    self.active_host_addresses[host_id].add(vip)
    
    def perform_scan(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Perform scanning and return results
        
        NEW: Returns both scan results AND scanned addresses for block history
        
        Returns:
            Tuple of:
            - Dict mapping host_id to number of successful scans
            - Dict mapping address to scan count (for MDP block history)
        """
        # Get scan results
        scan_results = self.scanner.scan(self.vip_to_host)
        
        # Track which addresses were scanned (for block history)
        scanned_addresses = {}
        for addr in self.scanner.current_period_scans:
            scanned_addresses[addr] = scanned_addresses.get(addr, 0) + 1
        
        return scan_results, scanned_addresses
    
    def get_scan_statistics(self) -> dict:
        """Get scanning statistics"""
        total_scanned = len(self.scanner.scanned_addresses)
        coverage = total_scanned / self.address_space_size * 100 if self.address_space_size > 0 else 0
        
        return {
            'total_scanned': total_scanned,
            'coverage': coverage,
            'unique_addresses': len(self.scanner.scanned_addresses),
            'active_hosts': len(self.active_host_addresses),
            'total_host_addresses': sum(len(addrs) for addrs in self.active_host_addresses.values())
        }


if __name__ == '__main__':
    print("Testing Scanning Strategies - IMPROVED VERSION")
    print("=" * 50)
    
    # Setup - using paper's parameters
    num_hosts = 30
    num_blocks = 50
    block_size = 128
    address_space = num_blocks * block_size
    scanning_rate = 16
    
    print(f"Network: {num_hosts} hosts, {num_blocks} blocks")
    print(f"Address space: {address_space} addresses")
    print(f"Scanning rate: {scanning_rate} addresses/period")
    
    # Create active host mapping
    active_hosts = {}
    for host_id in range(num_hosts):
        for _ in range(25):
            addr = random.randint(0, address_space - 1)
            active_hosts[addr] = host_id
    
    print(f"Active addresses: {len(active_hosts)}")
    
    # Test each strategy
    strategies = [
        ("Local Preference", LocalPreferenceScanning(address_space, scanning_rate)),
        ("Sequential", SequentialScanning(address_space, scanning_rate)),
    ]
    
    for name, strategy in strategies:
        print(f"\n{name} Scanning:")
        
        # Simulate 10 scan periods
        total_hits = 0
        total_scanned_addrs = 0
        
        for period in range(10):
            results = strategy.scan(active_hosts)
            hits = sum(results.values())
            total_hits += hits
            total_scanned_addrs += len(strategy.current_period_scans)
        
        print(f"  Total hits: {total_hits}")
        print(f"  Average hits/period: {total_hits/10:.1f}")
        print(f"  Scanned addresses: {len(strategy.scanned_addresses)}")
        print(f"  Avg scanned per period: {total_scanned_addrs/10:.1f}")
        print(f"  Hit rate: {total_hits / (scanning_rate * 10) * 100:.1f}%")