#!/bin/bash
# Epoch marker - call this from your MTD controller when shuffling occurs

EPOCHS_LOG="$FEATURE_HUNTER_RUN/capture/logs/mtd_epochs.log"

if [ -z "$FEATURE_HUNTER_RUN" ]; then
    echo "ERROR: FEATURE_HUNTER_RUN not set"
    exit 1
fi

# Create log file if it doesn't exist
mkdir -p "$(dirname "$EPOCHS_LOG")"

# Get current timestamp in seconds since epoch and human readable
epoch_time=$(date +%s)
human_time=$(date '+%Y-%m-%d %H:%M:%S')

# Log the MTD shuffle event
echo "${epoch_time},${human_time},MTD_SHUFFLE" >> "$EPOCHS_LOG"

echo "Epoch marker added: ${human_time} (${epoch_time})"
