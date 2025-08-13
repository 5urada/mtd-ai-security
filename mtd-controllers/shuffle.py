#!/usr/bin/env python3
import subprocess
import time
import random
import threading

class EnhancedMTDController:
    def __init__(self):
        # Real target IPs
        self.targets = {
            'target1': '10.10.1.3',
            'target2': '10.10.1.4', 
            'target3': '10.10.1.5'
        }
        
        # Virtual IP pool
        self.virtual_pool = [f'10.10.1.{i}' for i in range(20, 50)]
        
        # Current mappings: virtual -> real
        self.current_mappings = {}
        
        # Attacker node for route updates
        self.attacker_ip = '10.10.1.2'
        
        # Controller interface
        self.controller_interface = 'eth1'
        
    def clear_virtual_ips(self):
        """Remove virtual IPs from controller interface"""
        for vip in self.virtual_pool:
            subprocess.run([
                'sudo', 'ip', 'addr', 'del', f'{vip}/32', 
                'dev', self.controller_interface
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    def add_virtual_ip_to_interface(self, virtual_ip):
        """Add virtual IP to controller interface so it can receive traffic"""
        subprocess.run([
            'sudo', 'ip', 'addr', 'add', f'{virtual_ip}/32',
            'dev', self.controller_interface
        ], capture_output=True)
        print(f"Added {virtual_ip} to interface {self.controller_interface}")
    
    def clear_iptables(self):
        """Clear existing NAT rules"""
        subprocess.run(['sudo', 'iptables', '-t', 'nat', '-F'], capture_output=True)
        
    def add_virtual_mapping(self, virtual_ip, real_ip):
        """Add comprehensive iptables rules for virtual -> real mapping"""
        
        # Add virtual IP to controller interface first
        self.add_virtual_ip_to_interface(virtual_ip)
        
        # DNAT rules for different protocols
        # HTTP (port 80)
        subprocess.run([
            'sudo', 'iptables', '-t', 'nat', '-A', 'PREROUTING',
            '-p', 'tcp', '--dport', '80',
            '-d', virtual_ip, '-j', 'DNAT', '--to-destination', f'{real_ip}:80'
        ])
        
        # SSH (port 22) 
        subprocess.run([
            'sudo', 'iptables', '-t', 'nat', '-A', 'PREROUTING',
            '-p', 'tcp', '--dport', '22',
            '-d', virtual_ip, '-j', 'DNAT', '--to-destination', f'{real_ip}:22'
        ])
        
        # ICMP (for ping/timing analysis)
        subprocess.run([
            'sudo', 'iptables', '-t', 'nat', '-A', 'PREROUTING',
            '-p', 'icmp',
            '-d', virtual_ip, '-j', 'DNAT', '--to-destination', real_ip
        ])
        
        # Generic DNAT for other protocols
        subprocess.run([
            'sudo', 'iptables', '-t', 'nat', '-A', 'PREROUTING',
            '-d', virtual_ip, '-j', 'DNAT', '--to-destination', real_ip
        ])
        
        # SNAT rules for return traffic
        # HTTP return traffic
        subprocess.run([
            'sudo', 'iptables', '-t', 'nat', '-A', 'POSTROUTING',
            '-p', 'tcp', '--sport', '80',
            '-s', real_ip, '-j', 'SNAT', '--to-source', f'{virtual_ip}:80'
        ])
        
        # SSH return traffic
        subprocess.run([
            'sudo', 'iptables', '-t', 'nat', '-A', 'POSTROUTING', 
            '-p', 'tcp', '--sport', '22',
            '-s', real_ip, '-j', 'SNAT', '--to-source', f'{virtual_ip}:22'
        ])
        
        # ICMP return traffic
        subprocess.run([
            'sudo', 'iptables', '-t', 'nat', '-A', 'POSTROUTING',
            '-p', 'icmp',
            '-s', real_ip, '-j', 'SNAT', '--to-source', virtual_ip
        ])
        
        # Generic SNAT
        subprocess.run([
            'sudo', 'iptables', '-t', 'nat', '-A', 'POSTROUTING',
            '-s', real_ip, '-j', 'SNAT', '--to-source', virtual_ip
        ])
        
        print(f"Mapped: {virtual_ip} -> {real_ip} (HTTP/SSH/ICMP)")
        
    def update_attacker_routes(self, virtual_ip):
        """Add route on attacker node for new virtual IP"""
        ssh_cmd = [
            'ssh', '-o', 'StrictHostKeyChecking=no',
            f'surada@{self.attacker_ip}',
            f'sudo ip route add {virtual_ip}/32 via 10.10.1.1 2>/dev/null || true'
        ]
        subprocess.run(ssh_cmd, capture_output=True)
        
    def cleanup_old_virtual_ips(self, old_mappings):
        """Remove old virtual IPs from interface"""
        for old_virtual in old_mappings.keys():
            if old_virtual not in self.current_mappings:
                # Remove from interface
                subprocess.run([
                    'sudo', 'ip', 'addr', 'del', f'{old_virtual}/32',
                    'dev', self.controller_interface
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Remove route from attacker
                ssh_cmd = [
                    'ssh', '-o', 'StrictHostKeyChecking=no',
                    f'surada@{self.attacker_ip}',
                    f'sudo ip route del {old_virtual}/32 2>/dev/null || true'
                ]
                subprocess.run(ssh_cmd, capture_output=True)
        
    def rotate_ips(self):
        """Rotate all virtual IP assignments with proper cleanup"""
        print(f"\n=== Enhanced MTD Rotation at {time.strftime('%H:%M:%S')} ===")
        
        # Store old mappings for cleanup
        old_mappings = self.current_mappings.copy()
        
        # Clear existing rules and virtual IPs
        self.clear_iptables()
        self.current_mappings.clear()
        
        # Assign new virtual IPs
        available_virtuals = self.virtual_pool.copy()
        
        for target_name, real_ip in self.targets.items():
            # Pick new virtual IP
            virtual_ip = random.choice(available_virtuals)
            available_virtuals.remove(virtual_ip)
            
            # Store mapping
            self.current_mappings[virtual_ip] = real_ip
            
            # Add comprehensive iptables rules and interface config
            self.add_virtual_mapping(virtual_ip, real_ip)
            
            # Update attacker routes
            self.update_attacker_routes(virtual_ip)
        
        # Clean up old virtual IPs and routes
        self.cleanup_old_virtual_ips(old_mappings)
        
        print("Current mappings:")
        for virtual, real in self.current_mappings.items():
            print(f"  {virtual} -> {real}")
            
    def test_connectivity(self):
        """Test connectivity to virtual IPs"""
        print("\nTesting virtual IP connectivity:")
        for virtual_ip, real_ip in self.current_mappings.items():
            # Test ping
            ping_result = subprocess.run([
                'ping', '-c', '1', '-W', '2', virtual_ip
            ], capture_output=True)
            
            ping_status = "✓" if ping_result.returncode == 0 else "✗"
            print(f"  {virtual_ip} -> {real_ip}: ping {ping_status}")
            
    def show_status(self):
        """Show current NAT table and interface status"""
        print("\nCurrent iptables NAT rules:")
        result = subprocess.run(['sudo', 'iptables', '-t', 'nat', '-L', '-n', '-v'],
                               capture_output=True, text=True)
        print(result.stdout)
        
        print("Virtual IPs on interface:")
        result = subprocess.run(['ip', 'addr', 'show', self.controller_interface],
                               capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if '10.10.1.' in line and '/32' in line:
                print(f"  {line.strip()}")
                
    def start_rotation(self, interval=30):
        """Start automatic MTD rotation with enhanced features"""
        print(f"Starting Enhanced MTD Controller with {interval}s rotation interval")
        print("Features: HTTP/SSH/ICMP support, interface management, connectivity testing")
        
        # Enable IP forwarding
        subprocess.run(['sudo', 'sysctl', '-w', 'net.ipv4.ip_forward=1'],
                      capture_output=True)
        
        # Clear any existing virtual IPs
        self.clear_virtual_ips()
        
        # Initial rotation
        self.rotate_ips()
        self.show_status()
        self.test_connectivity()
        
        # Start rotation loop
        try:
            while True:
                time.sleep(interval)
                self.rotate_ips()
                if len(self.current_mappings) > 0:
                    self.test_connectivity()
        except KeyboardInterrupt:
            print("\nStopping Enhanced MTD Controller...")
            self.clear_iptables()
            self.clear_virtual_ips()

if __name__ == "__main__":
    controller = EnhancedMTDController()
    controller.start_rotation(30)
