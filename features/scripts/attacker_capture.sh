#!/bin/bash
# Attacker packet capture aligned to MTD shuffle intervals

set -e

CONFIG_FILE="$FEATURE_HUNTER_RUN/config.yaml"
PCAP_DIR="$FEATURE_HUNTER_RUN/capture/pcap"
LOG_FILE="$FEATURE_HUNTER_RUN/capture/logs/capture.log"

# Parse config
SHUFFLE_INTERVAL=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_FILE')); print(cfg['network']['shuffle_interval'])")
INTERFACE=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_FILE')); print(cfg['network']['sniff_interface'])")
BPF_FILTER=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_FILE')); print(cfg['capture']['bpf_filter'])")

echo "Starting packet capture at $(date)" | tee -a "$LOG_FILE"
echo "Interface: $INTERFACE" | tee -a "$LOG_FILE"
echo "Shuffle interval: ${SHUFFLE_INTERVAL}s" | tee -a "$LOG_FILE"
echo "Filter: $BPF_FILTER" | tee -a "$LOG_FILE"

# Check if interface exists
if ! ip link show "$INTERFACE" >/dev/null 2>&1; then
    echo "ERROR: Interface $INTERFACE not found!" | tee -a "$LOG_FILE"
    echo "Available interfaces:"
    ip link show | grep -E "^[0-9]+:" | tee -a "$LOG_FILE"
    exit 1
fi

# Generate timestamp-based filename
start_time=$(date +"%Y%m%d_%H%M%S")
pcap_prefix="mtd_cap_${start_time}"

echo "Capture files will be: ${pcap_prefix}_*.pcap" | tee -a "$LOG_FILE"
echo "Starting tcpdump with ring buffer..." | tee -a "$LOG_FILE"

# Run tcpdump with ring buffer aligned to shuffle interval
# -G: rotate files every SHUFFLE_INTERVAL seconds
# -w: write to file with timestamp
# -W: keep max 20 files (configurable)
sudo tcpdump -i "$INTERFACE" \
    -G "$SHUFFLE_INTERVAL" \
    -w "${PCAP_DIR}/${pcap_prefix}_%Y%m%d_%H%M%S.pcap" \
    -W 20 \
    -Z "$(whoami)" \
    "$BPF_FILTER" \
    2>&1 | tee -a "$LOG_FILE"
