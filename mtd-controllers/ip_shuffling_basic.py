#!/usr/bin/env python3
import subprocess
import time
import random
import threading

class MTDController:
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
        
    def clear_iptables(self):
        """Clear existing NAT rules"""
        subprocess.run(['sudo', 'iptables', '-t', 'nat', '-F'], capture_output=True)
        
    def add_virtual_mapping(self, virtual_ip, real_ip):
        """Add iptables rules for virtual -> real mapping"""
        # Incoming: virtual -> real
        subprocess.run([
            'sudo', 'iptables', '-t', 'nat', '-A', 'PREROUTING',
            '-d', virtual_ip, '-j', 'DNAT', '--to-destination', real_ip
        ])
        
        # Outgoing: real -> virtual  
        subprocess.run([
            'sudo', 'iptables', '-t', 'nat', '-A', 'POSTROUTING',
            '-s', real_ip, '-j', 'SNAT', '--to-source', virtual_ip
        ])
        
        print(f"Mapped: {virtual_ip} -> {real_ip}")
        
    def update_attacker_routes(self, virtual_ip):
        """Add route on attacker node for new virtual IP"""
        ssh_cmd = [
            'ssh', '-o', 'StrictHostKeyChecking=no',
            f'surada@{self.attacker_ip}',
            f'sudo ip route add {virtual_ip}/32 via 10.10.1.1 2>/dev/null || true'
        ]
        subprocess.run(ssh_cmd, capture_output=True)
        
    def rotate_ips(self):
        """Rotate all virtual IP assignments"""
        print(f"\n=== MTD Rotation at {time.strftime('%H:%M:%S')} ===")
        
        # Clear existing rules
        self.clear_iptables()
        
        # Clear old mappings
        old_mappings = self.current_mappings.copy()
        self.current_mappings.clear()
        
        # Assign new virtual IPs
        available_virtuals = self.virtual_pool.copy()
        
        for target_name, real_ip in self.targets.items():
            # Pick new virtual IP
            virtual_ip = random.choice(available_virtuals)
            available_virtuals.remove(virtual_ip)
            
            # Store mapping
            self.current_mappings[virtual_ip] = real_ip
            
            # Add iptables rules
            self.add_virtual_mapping(virtual_ip, real_ip)
            
            # Update attacker routes
            self.update_attacker_routes(virtual_ip)
            
        # Clean up old routes on attacker
        for old_virtual in old_mappings.keys():
            if old_virtual not in self.current_mappings:
                ssh_cmd = [
                    'ssh', '-o', 'StrictHostKeyChecking=no',
                    f'surada@{self.attacker_ip}',
                    f'sudo ip route del {old_virtual}/32 2>/dev/null || true'
                ]
                subprocess.run(ssh_cmd, capture_output=True)
        
        print("Current mappings:")
        for virtual, real in self.current_mappings.items():
            print(f"  {virtual} -> {real}")
            
    def show_status(self):
        """Show current NAT table and routes"""
        print("\nCurrent iptables NAT rules:")
        result = subprocess.run(['sudo', 'iptables', '-t', 'nat', '-L', '-n'], 
                               capture_output=True, text=True)
        print(result.stdout)
        
    def start_rotation(self, interval=30):
        """Start automatic MTD rotation"""
        print(f"Starting MTD Controller with {interval}s rotation interval")
        
        # Enable IP forwarding
        subprocess.run(['sudo', 'sysctl', '-w', 'net.ipv4.ip_forward=1'], 
                      capture_output=True)
        
        # Initial rotation
        self.rotate_ips()
        self.show_status()
        
        # Start rotation loop
        try:
            while True:
                time.sleep(interval)
                self.rotate_ips()
        except KeyboardInterrupt:
            print("\nStopping MTD Controller...")
            self.clear_iptables()

if __name__ == "__main__":
    controller = MTDController()
    controller.start_rotation(30)