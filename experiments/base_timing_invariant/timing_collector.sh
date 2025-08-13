#!/bin/bash
# timing_collector.sh - Collect response time data during MTD operations

LOGFILE="timing_data.csv"

# Initialize CSV file with headers
echo "timestamp,virtual_ip,response_time_ms,status" > $LOGFILE

collect_timing_data() {
    echo "Collecting timing data every 5 seconds (Ctrl+C to stop)..."
    echo "Data will be saved to: $LOGFILE"
    
    while true; do
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        
        # Scan the virtual IP range and measure response times
        for i in {20..49}; do
            virtual_ip="10.10.1.$i"
            
            # Measure ping response time
            start_time=$(date +%s.%N)
            if ping -c 1 -W 1 "$virtual_ip" >/dev/null 2>&1; then
                end_time=$(date +%s.%N)
                response_time=$(echo "($end_time - $start_time) * 1000" | bc -l)
                response_time=$(printf "%.2f" $response_time)
                status="success"
                echo "$timestamp,$virtual_ip,$response_time,$status" >> $LOGFILE
                echo "  $virtual_ip: ${response_time}ms"
            else
                echo "$timestamp,$virtual_ip,timeout,failed" >> $LOGFILE
            fi
        done
        
        echo "---"
        sleep 5
    done
}

# Check if bc is installed (needed for calculations)
if ! command -v bc &> /dev/null; then
    echo "Installing bc for calculations..."
    sudo apt install -y bc
fi

# Function to show recent data
show_recent_data() {
    echo "Last 10 successful measurements:"
    grep "success" $LOGFILE | tail -10
}

# Function to show summary
show_summary() {
    echo "=== Timing Data Summary ==="
    echo "Total measurements: $(grep -c "success" $LOGFILE)"
    echo "Active IPs found:"
    grep "success" $LOGFILE | cut -d',' -f2 | sort | uniq -c | sort -nr
}

# Main menu
case "${1:-collect}" in
    "collect")
        collect_timing_data
        ;;
    "show")
        show_recent_data
        ;;
    "summary")
        show_summary
        ;;
    *)
        echo "Usage: $0 [collect|show|summary]"
        echo "  collect  - Start collecting timing data (default)"
        echo "  show     - Show recent measurements"
        echo "  summary  - Show summary statistics"
        ;;
esac
