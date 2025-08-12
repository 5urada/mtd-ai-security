# Small Scale CloudLab MTD Testbed Setup Documentation

## Overview
This document describes how to set up a Moving Target Defense (MTD) testbed on CloudLab with IP shuffling capabilities. The setup includes a controller node that performs automated IP address translation and rotation, simulating real MTD environments.

## Architecture
- **Controller Node (vm0)**: MTD orchestrator with IP translation via iptables NAT
- **Attacker Node (vm1)**: Simulated attacker for testing MTD effectiveness  
- **Target Nodes (vm2-vm4)**: Protected services behind virtual IP addresses
- **Network**: All nodes connected via 10.10.1.0/24 subnet

## Prerequisites
- CloudLab account with project access
- 5 VM allocation using `small-lan` profile
- Ubuntu 20.04 or 22.04 on all nodes

## Step-by-Step Setup

### 1. CloudLab Experiment Creation
```
Profile: small-lan:43 (or similar 5-node profile)
OS Image: Ubuntu 20.04
Node Count: 5
Cluster: Utah (recommended)
```

### 2. Controller Node Setup (vm0)

#### Network Configuration
```bash
# Remove IP from physical interface
sudo ip addr flush dev eth1

# Configure IP forwarding
sudo sysctl -w net.ipv4.ip_forward=1
echo 'net.ipv4.ip_forward=1' | sudo tee -a /etc/sysctl.conf

# Set controller IP (if needed)
sudo ip addr add 10.10.1.1/24 dev eth1
```

#### Install Dependencies
```bash
sudo apt update
sudo apt install -y python3 python3-pip git vim
```

#### Create MTD Controller
```bash
mkdir ~/mtd-controller
cd ~/mtd-controller
```

Create `mtd_controller.py`:
```python
#!/usr/bin/env python3
import subprocess
import time
import random

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
        
    def rotate_ips(self):
        """Rotate all virtual IP assignments"""
        print(f"\n=== MTD Rotation at {time.strftime('%H:%M:%S')} ===")
        
        # Clear existing rules
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
            
            # Add iptables rules
            self.add_virtual_mapping(virtual_ip, real_ip)
            
        print("Current mappings:")
        for virtual, real in self.current_mappings.items():
            print(f"  {virtual} -> {real}")
            
    def start_rotation(self, interval=30):
        """Start automatic MTD rotation"""
        print(f"Starting MTD Controller with {interval}s rotation interval")
        
        # Enable IP forwarding
        subprocess.run(['sudo', 'sysctl', '-w', 'net.ipv4.ip_forward=1'], 
                      capture_output=True)
        
        # Initial rotation
        self.rotate_ips()
        
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
```

Make executable:
```bash
chmod +x mtd_controller.py
```

### 3. Attacker Node Setup (vm1)

#### Install Dependencies
```bash
sudo apt update
sudo apt install -y nmap vim
```

#### Setup Virtual IP Routes
```bash
# Add routes for entire virtual IP pool
for i in {20..49}; do
    sudo ip route add 10.10.1.$i/32 via 10.10.1.1
done
```

#### Create Attacker Testing Script
Create `attacker_setup.sh` (see previous artifact for full script):
```bash
#!/bin/bash
# Basic functions for MTD testing
setup_routes() {
    for i in {20..49}; do
        sudo ip route add 10.10.1.$i/32 via 10.10.1.1 2>/dev/null || true
    done
}

scan_virtual_ips() {
    echo "Scanning for active virtual IPs..."
    for i in {20..49}; do
        if ping -c 1 -W 1 10.10.1.$i >/dev/null 2>&1; then
            echo "  ACTIVE: 10.10.1.$i"
        fi
    done
}

# Add other functions as needed
```

### 4. Target Nodes Setup (vm2, vm3, vm4)

Target nodes require minimal setup - they just need to be reachable:

```bash
# Verify connectivity
ping -c 2 10.10.1.1  # Should reach controller
```

Optionally install services for more realistic testing:
```bash
# Example: Simple web server
sudo apt install -y apache2
sudo systemctl start apache2
```

## Operation

### Starting the MTD System

1. **On Controller (vm0)**:
```bash
cd ~/mtd-controller
sudo python3 mtd_controller.py
```

2. **On Attacker (vm1)**:
```bash
# Setup routes
./attacker_setup.sh setup

# Monitor MTD in real-time
./attacker_setup.sh monitor
```

### Testing MTD Effectiveness

From attacker node:
```bash
# Scan for active targets
./attacker_setup.sh scan

# Test specific virtual IPs
ping -c 3 10.10.1.23  # Replace with current virtual IP

# Network discovery
nmap -sn 10.10.1.20-49

# Continuous monitoring to see IP rotation
watch -n 5 'nmap -sn 10.10.1.20-49 | grep "Nmap scan report"'
```

## Key Network Mappings

- **Real Target IPs**: 10.10.1.3, 10.10.1.4, 10.10.1.5 (static)
- **Virtual IP Pool**: 10.10.1.20-49 (dynamic, rotates every 30s)
- **Controller IP**: 10.10.1.1
- **Attacker IP**: 10.10.1.2

## Verification

### Successful Setup Indicators
1. Controller shows rotation messages every 30 seconds
2. Attacker can ping virtual IPs that map to real targets
3. Virtual IPs change periodically (old ones become unreachable)
4. iptables NAT rules update automatically

### Troubleshooting

**Virtual IPs not reachable:**
```bash
# Check IP forwarding
sudo sysctl net.ipv4.ip_forward

# Verify iptables rules
sudo iptables -t nat -L -n

# Check routes on attacker
ip route show | grep "10.10.1"
```

**MTD not rotating:**
- Verify controller script is running
- Check for Python errors in controller output
- Ensure sufficient privileges for iptables commands

## Configuration Options

### Rotation Interval
Modify the interval parameter in `mtd_controller.py`:
```python
controller.start_rotation(15)  # 15-second intervals
```

### Virtual IP Pool Size
Adjust the range in `mtd_controller.py`:
```python
self.virtual_pool = [f'10.10.1.{i}' for i in range(50, 100)]  # Larger pool
```

### Target Addition
Add more targets by extending the targets dictionary:
```python
self.targets = {
    'target1': '10.10.1.3',
    'target2': '10.10.1.4', 
    'target3': '10.10.1.5',
    'target4': '10.10.1.6'  # Additional target
}
```

## Research Applications

This testbed enables research into:
- **AI Attacker Adaptation**: How quickly can automated agents discover new target locations?
- **MTD Timing**: Optimal rotation intervals vs. system overhead
- **Scanning Strategy Effectiveness**: Different reconnaissance approaches under MTD
- **Multi-Agent Coordination**: How well do coordinated attacks handle target mobility

## Cleanup

When experiment is complete:
```bash
# On controller
sudo iptables -t nat -F

# On attacker
for i in {20..49}; do sudo ip route del 10.10.1.$i/32; done
```

---

**Created**: August 2025  
**Tested**: CloudLab Utah cluster, Ubuntu 20.04  
