#!/bin/bash
# Client iperf3 traffic generation with jitter

set -e

CONFIG_FILE="$FEATURE_HUNTER_RUN/config.yaml"
LOG_FILE="$FEATURE_HUNTER_RUN/capture/logs/client_iperf.log"

# Parse config (simple grep-based parsing)
VIP_START=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_FILE')); print(cfg['network']['virtual_ips'][0])")
VIP_END=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_FILE')); print(cfg['network']['virtual_ips'][-1])")
PORTS=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_FILE')); print(','.join(map(str, cfg['network']['target_ports'])))")

echo "Starting iperf3 client loop at $(date)" | tee -a "$LOG_FILE"
echo "Targeting VIP range: $VIP_START to $VIP_END" | tee -a "$LOG_FILE"  
echo "Ports: $PORTS" | tee -a "$LOG_FILE"

# Convert VIP range to array
VIP_BASE="10.10.1"
VIP_RANGE=($(seq 20 49))
PORT_ARRAY=(${PORTS//,/ })

# Function to add jitter
add_jitter() {
    local base_delay=$1
    local jitter_max=${2:-2}
    local jitter=$((RANDOM % jitter_max))
    echo $((base_delay + jitter))
}

echo "Traffic generation started. Press Ctrl+C to stop." | tee -a "$LOG_FILE"

while true; do
    # Pick random VIP and port
    vip_suffix=${VIP_RANGE[$RANDOM % ${#VIP_RANGE[@]}]}
    target_ip="${VIP_BASE}.${vip_suffix}"
    target_port=${PORT_ARRAY[$RANDOM % ${#PORT_ARRAY[@]}]}
    
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] Connecting to $target_ip:$target_port" | tee -a "$LOG_FILE"
    
    # Run iperf3 with timeout
    timeout 15s iperf3 -c "$target_ip" -p "$target_port" -t 10 -i 1 \
        >> "$LOG_FILE" 2>&1 || {
        echo "[$timestamp] iperf3 failed or timed out for $target_ip:$target_port" | tee -a "$LOG_FILE"
    }
    
    # Wait with jitter (5-15 seconds)
    delay=$(add_jitter 5 10)
    echo "[$timestamp] Waiting ${delay}s before next connection" | tee -a "$LOG_FILE"
    sleep "$delay"
done
