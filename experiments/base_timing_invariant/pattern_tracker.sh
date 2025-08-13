#!/bin/bash
# pattern_tracker.sh - Track MTD patterns over time

track_mtd_patterns() {
    echo "=== MTD Pattern Analysis ==="
    echo ""
    
    # Create a cleaner time-series view
    echo "MTD Rotation Timeline:"
    echo "Time     | Active Virtual IPs (with response times)"
    echo "---------|------------------------------------------------"
    
    # Group by timestamp and show active IPs with their response times
    grep "success" timing_data.csv | while read line; do
        timestamp=$(echo "$line" | cut -d',' -f1)
        virtual_ip=$(echo "$line" | cut -d',' -f2)
        response_time=$(echo "$line" | cut -d',' -f3)
        time_only=$(echo "$timestamp" | cut -d' ' -f2 | cut -d':' -f1,2)
        echo "$time_only $virtual_ip:$response_time"
    done | sort | awk '
    {
        time = $1
        ip_time = $2
        if (time != prev_time && prev_time != "") {
            print prev_time " | " ips
            ips = ""
        }
        ips = (ips == "" ? ip_time : ips " " ip_time)
        prev_time = time
    }
    END { if (prev_time != "") print prev_time " | " ips }'
    
    echo ""
    echo "=== Response Time Analysis ==="
    echo ""
    
    # Calculate response time statistics more robustly
    echo "Virtual IP Response Time Summary:"
    echo "IP        | Count | Avg    | Min    | Max    | Pattern"
    echo "----------|-------|--------|--------|--------|---------"
    
    # Get all unique virtual IPs that had successful responses
    for ip in $(grep "success" timing_data.csv | cut -d',' -f2 | sort | uniq); do
        # Extract response times for this IP
        times=$(grep "$ip.*success" timing_data.csv | cut -d',' -f3 | tr '\n' ' ')
        count=$(echo $times | wc -w)
        
        if [ $count -gt 0 ]; then
            # Calculate statistics using a more robust approach
            avg=$(echo $times | tr ' ' '\n' | awk '{sum+=$1; n++} END {printf "%.2f", sum/n}')
            min_val=$(echo $times | tr ' ' '\n' | sort -n | head -1)
            max_val=$(echo $times | tr ' ' '\n' | sort -n | tail -1)
            
            # Determine pattern category based on average response time
            if [ $(echo "$avg < 3.65" | bc -l) -eq 1 ]; then
                pattern="FAST"
            elif [ $(echo "$avg > 3.95" | bc -l) -eq 1 ]; then
                pattern="SLOW"
            else
                pattern="MID"
            fi
            
            printf "%-9s | %5d | %6s | %6s | %6s | %s\n" \
                   "$ip" "$count" "$avg" "$min_val" "$max_val" "$pattern"
        fi
    done | sort -k3 -n
    
    echo ""
    echo "=== Invariant Detection ==="
    echo ""
    
    # Look for timing patterns that might indicate the same underlying target
    echo "Potential Target Clusters (based on response time patterns):"
    echo ""
    
    # Group IPs by response time ranges more precisely
    echo "FAST targets (< 3.70ms average):"
    grep "success" timing_data.csv | cut -d',' -f2,3 | \
    awk -F',' '{sum[$1]+=$2; count[$1]++} END {
        for (ip in sum) {
            avg = sum[ip]/count[ip]
            if (avg < 3.70) printf "  %s (%.2f ms, %d measurements)\n", ip, avg, count[ip]
        }
    }' | sort -k2 -n
    
    echo ""
    echo "MEDIUM targets (3.70-3.90ms average):"
    grep "success" timing_data.csv | cut -d',' -f2,3 | \
    awk -F',' '{sum[$1]+=$2; count[$1]++} END {
        for (ip in sum) {
            avg = sum[ip]/count[ip]
            if (avg >= 3.70 && avg <= 3.90) printf "  %s (%.2f ms, %d measurements)\n", ip, avg, count[ip]
        }
    }' | sort -k2 -n
    
    echo ""
    echo "SLOW targets (> 3.90ms average):"
    grep "success" timing_data.csv | cut -d',' -f2,3 | \
    awk -F',' '{sum[$1]+=$2; count[$1]++} END {
        for (ip in sum) {
            avg = sum[ip]/count[ip]
            if (avg > 3.90) printf "  %s (%.2f ms, %d measurements)\n", ip, avg, count[ip]
        }
    }' | sort -k2 -n
    
    echo ""
    echo "=== Invariant Hypothesis ==="
    echo ""
    echo "Based on response time clustering, we might have:"
    echo "• Target A (FAST):   Consistently < 3.70ms response time"
    echo "• Target B (MEDIUM): Consistently 3.70-3.90ms response time" 
    echo "• Target C (SLOW):   Consistently > 3.90ms response time"
    echo ""
    echo "This suggests that despite IP shuffling, underlying network"
    echo "characteristics (distance, routing, processing time) remain"
    echo "consistent per target - a potential TIMING INVARIANT!"
}

# Check dependencies
if ! command -v bc &> /dev/null; then
    echo "Installing bc for calculations..."
    sudo apt install -y bc
fi

if [ ! -f "timing_data.csv" ]; then
    echo "Error: timing_data.csv not found!"
    exit 1
fi

track_mtd_patterns
