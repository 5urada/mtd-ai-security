#!/bin/bash
# attacker_setup.sh - Setup script for MTD testing from attacker node

echo "=== MTD Attacker Setup ==="

# Function to setup routes
setup_routes() {
    echo "Setting up routes for virtual IP pool (10.10.1.20-49)..."
    for i in {20..49}; do
        sudo ip route add 10.10.1.$i/32 via 10.10.1.1 2>/dev/null || true
    done
    echo "Routes installed."
}

# Function to scan for active virtual IPs
scan_virtual_ips() {
    echo "Scanning for active virtual IPs..."
    active_ips=()
    
    for i in {20..49}; do
        if ping -c 1 -W 1 10.10.1.$i >/dev/null 2>&1; then
            active_ips+=(10.10.1.$i)
            echo "  ACTIVE: 10.10.1.$i"
        fi
    done
    
    echo "Found ${#active_ips[@]} active virtual IPs: ${active_ips[*]}"
}

# Function to continuously monitor target changes
monitor_targets() {
    echo "Starting continuous monitoring (Ctrl+C to stop)..."
    echo "Time                 | Active Virtual IPs"
    echo "---------------------|----------------------------------"
    
    while true; do
        timestamp=$(date '+%H:%M:%S')
        active=""
        
        for i in {20..49}; do
            if ping -c 1 -W 1 10.10.1.$i >/dev/null 2>&1; then
                active="$active 10.10.1.$i"
            fi
        done
        
        printf "%-20s | %s\n" "$timestamp" "$active"
        sleep 5
    done
}

# Function to test specific IPs
test_ips() {
    if [ $# -eq 0 ]; then
        echo "Testing current virtual IPs from last scan..."
        scan_virtual_ips
        return
    fi
    
    echo "Testing specified IPs: $*"
    for ip in "$@"; do
        if ping -c 2 -W 2 "$ip" >/dev/null 2>&1; then
            echo "  SUCCESS: $ip is reachable"
        else
            echo "  FAILED:  $ip is not reachable"
        fi
    done
}

# Function to show current routes
show_routes() {
    echo "Current virtual IP routes:"
    ip route show | grep "10.10.1" | grep "via 10.10.1.1" | head -10
    echo "... (showing first 10)"
}

# Function to cleanup routes
cleanup_routes() {
    echo "Removing virtual IP routes..."
    for i in {20..49}; do
        sudo ip route del 10.10.1.$i/32 2>/dev/null || true
    done
    echo "Routes removed."
}

# Main menu
case "${1:-menu}" in
    "setup")
        setup_routes
        ;;
    "scan")
        scan_virtual_ips
        ;;
    "monitor")
        monitor_targets
        ;;
    "test")
        shift
        test_ips "$@"
        ;;
    "routes")
        show_routes
        ;;
    "cleanup")
        cleanup_routes
        ;;
    "menu"|*)
        echo ""
        echo "Usage: $0 [command] [args]"
        echo ""
        echo "Commands:"
        echo "  setup      - Install routes for virtual IP pool"
        echo "  scan       - Scan for currently active virtual IPs"
        echo "  monitor    - Continuously monitor target changes"
        echo "  test [ips] - Test specific IPs (or scan if none given)"
        echo "  routes     - Show current virtual IP routes"
        echo "  cleanup    - Remove all virtual IP routes"
        echo ""
        echo "Examples:"
        echo "  $0 setup"
        echo "  $0 scan"
        echo "  $0 test 10.10.1.23 10.10.1.38"
        echo "  $0 monitor"
        ;;
esac