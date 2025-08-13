#!/bin/bash
# simple_analysis.sh - Analyze timing patterns for potential invariants

analyze_timing_patterns() {
    echo "=== MTD Timing Pattern Analysis ==="
    echo ""
    
    # Extract successful measurements only
    grep "success" timing_data.csv > successful_measurements.csv
    
    echo "1. Active Virtual IPs by Time Period:"
    echo "Time                | Active Virtual IPs"
    echo "--------------------|--------------------------------"
    
    # Group by time periods (roughly every 30 seconds due to MTD rotation)
    grep "success" timing_data.csv | cut -d',' -f1,2 | sort | uniq | \
    while read line; do
        timestamp=$(echo $line | cut -d',' -f1)
        virtual_ip=$(echo $line | cut -d',' -f2)
        time_only=$(echo $timestamp | cut -d' ' -f2)
        echo "$time_only | $virtual_ip"
    done | sort | uniq -c | head -20
    
    echo ""
    echo "2. Response Time Statistics by Virtual IP:"
    echo "Virtual IP    | Count | Avg Response | Min  | Max  | StdDev"
    echo "------------- |-------|--------------|------|------|-------"
    
    # Analyze response times per virtual IP
    for ip in $(grep "success" timing_data.csv | cut -d',' -f2 | sort | uniq); do
        response_times=$(grep "$ip.*success" timing_data.csv | cut -d',' -f3)
        count=$(echo "$response_times" | wc -l)
        
        if [ $count -gt 0 ]; then
            # Calculate statistics using awk
            stats=$(echo "$response_times" | awk '
            BEGIN { sum=0; sumsq=0; min=999; max=0; n=0 }
            { 
                if ($1+0 > 0) {  # Only process numeric values
                    sum+=$1; sumsq+=$1*$1; n++
                    if ($1 < min) min=$1
                    if ($1 > max) max=$1
                }
            }
            END { 
                if (n > 0) {
                    avg=sum/n
                    stddev=sqrt((sumsq - sum*sum/n)/(n-1))
                    printf "%.2f %.2f %.2f %.2f", avg, min, max, stddev
                } else {
                    print "0 0 0 0"
                }
            }')
            
            avg=$(echo $stats | cut -d' ' -f1)
            min_val=$(echo $stats | cut -d' ' -f2)
            max_val=$(echo $stats | cut -d' ' -f3)
            stddev=$(echo $stats | cut -d' ' -f4)
            
            printf "%-13s | %5d | %12s | %4s | %4s | %6s\n" \
                   "$ip" "$count" "$avg" "$min_val" "$max_val" "$stddev"
        fi
    done | sort -k3 -n
    
    echo ""
    echo "3. Response Time Clustering (looking for invariants):"
    echo ""
    
    # Try to cluster response times to see if there are distinct groups
    # This is a simple approach - group by response time ranges
    echo "Response Time Range | Count | Virtual IPs"
    echo "-------------------|-------|------------------------------------------"
    
    # Group response times into ranges
    for range in "3.00-3.49" "3.50-3.69" "3.70-3.79" "3.80-3.89" "3.90-4.00" "4.00-4.50"; do
        min_val=$(echo $range | cut -d'-' -f1)
        max_val=$(echo $range | cut -d'-' -f2)
        
        # Find measurements in this range
        matches=$(awk -F',' -v min=$min_val -v max=$max_val '
            $4=="success" && $3+0 >= min && $3+0 <= max { 
                print $2 "(" $3 ")"
            }' timing_data.csv)
        
        count=$(echo "$matches" | grep -c "10.10.1" 2>/dev/null || echo "0")
        
        if [ $count -gt 0 ]; then
            # Get unique IPs in this range
            unique_ips=$(echo "$matches" | cut -d'(' -f1 | sort | uniq | tr '\n' ' ')
            printf "%-18s | %5d | %s\n" "$range" "$count" "$unique_ips"
        fi
    done
    
    echo ""
    echo "4. Potential Invariant Detection:"
    echo ""
    
    # Simple heuristic: if certain virtual IPs consistently have similar response times,
    # they might be the same underlying target
    echo "Looking for virtual IPs with similar response time characteristics..."
    echo ""
    
    # Create temporary file with IP and average response time
    temp_file="/tmp/ip_avg_times.txt"
    for ip in $(grep "success" timing_data.csv | cut -d',' -f2 | sort | uniq); do
        avg_time=$(grep "$ip.*success" timing_data.csv | cut -d',' -f3 | \
                  awk '{sum+=$1; n++} END {if(n>0) printf "%.2f", sum/n; else print "0"}')
        if [ "$avg_time" != "0" ]; then
            echo "$ip $avg_time" >> $temp_file
        fi
    done
    
    # Group IPs by similar average response times (within 0.2ms)
    echo "Potential target groups (based on similar response times):"
    echo "Group | Virtual IPs | Avg Response Time"
    echo "------|-------------|------------------"
    
    group_num=1
    while read ip avg_time; do
        if [ ! -f "/tmp/processed_ips.txt" ] || ! grep -q "$ip" /tmp/processed_ips.txt 2>/dev/null; then
            # Find other IPs with similar response times (within 0.2ms)
            similar_ips=$(awk -v target=$avg_time -v current_ip=$ip '
                function abs(x) { return x < 0 ? -x : x }
                abs($2 - target) <= 0.2 { print $1 }
            ' $temp_file | tr '\n' ' ')
            
            if [ $(echo $similar_ips | wc -w) -gt 1 ]; then
                printf "%5d | %-25s | %.2f ms\n" $group_num "$similar_ips" $avg_time
                echo $similar_ips | tr ' ' '\n' >> /tmp/processed_ips.txt
                group_num=$((group_num + 1))
            fi
        fi
    done < $temp_file
    
    # Cleanup
    rm -f $temp_file /tmp/processed_ips.txt
}

# Check if timing_data.csv exists
if [ ! -f "timing_data.csv" ]; then
    echo "Error: timing_data.csv not found!"
    echo "Run the timing collector first: ./timing_collector.sh collect"
    exit 1
fi

# Run analysis
analyze_timing_patterns
