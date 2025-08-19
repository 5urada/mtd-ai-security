#!/bin/bash
# Client ping traffic generation with jitter

set -e

CONFIG_FILE="$FEATURE_HUNTER_RUN/config.yaml"
LOG_FILE="$FEATURE_HUNTER_RUN/capture/logs/client_ping.log"

echo "Starting ping client loop at $(date)" | tee -a "$LOG_FILE"

# VIP range from config
VIP_BASE="10.10.1"
VIP_RANGE=($(seq 20 49))

# Function to add jitter (50-200ms as configured)
add_jitter() {
    local base_delay=$1
    local jitter_ms=$((50 + RANDOM % 150))  # 50-200ms
    local jitter_sec=$(echo "scale=3; $jitter_ms/1000" | bc -l)
    echo "scale=3; $base_delay + $jitter_sec" | bc -l
}

echo "Ping traffic started. Press Ctrl+C to stop." | tee -a "$LOG_FILE"

while true; do
    # Pick random VIP
    vip_suffix=${VIP_RANGE[$RANDOM % ${#VIP_RANGE[@]}]}
    target_ip="${VIP_BASE}.${vip_suffix}"
    
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Ping with timestamp
    echo "[$timestamp] Pinging $target_ip" | tee -a "$LOG_FILE"
    ping -c 3 -W 2 "$target_ip" >> "$LOG_FILE" 2>&1 || {
        echo "[$timestamp] Ping failed for $target_ip" | tee -a "$LOG_FILE"
    }
    
    # Wait with jitter (1s base + jitter)
    delay=$(add_jitter 1.0)
    sleep "$delay"
done
