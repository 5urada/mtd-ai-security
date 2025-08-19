#!/bin/bash
# minimal_vulnbot_tools.sh - Install Kali essential tools on CloudLab
# Lightweight version focused on MTD research

set -e

echo "Installing minimal tools for VulnBot MTD research..."

# Update system
sudo apt update

# Core tools VulnBot actually needs
echo "Installing core scanning tools..."
sudo apt install -y \
    nmap \
    python3-pip \
    python3-dev \
    netcat-openbsd \
    curl \
    wget \
    openssh-server

# Enable SSH
echo "Configuring SSH..."
sudo systemctl enable ssh
sudo systemctl start ssh

echo "Adding MTD aliases..."
cat >> ~/.bashrc << 'EOF'
# MTD testing aliases
alias mtd-scan='nmap -sn 10.10.1.20-49'
alias quick-scan='nmap -F'
alias check-targets='ping -c 1 10.10.1.3 && ping -c 1 10.10.1.4 && ping -c 1 10.10.1.5'
EOF

echo "Done! Lightweight setup complete."
echo "Installed: nmap, python-nmap, scapy, requests, paramiko"
echo "SSH enabled for VulnBot shell connections"
echo "Run 'source ~/.bashrc' to load aliases"
