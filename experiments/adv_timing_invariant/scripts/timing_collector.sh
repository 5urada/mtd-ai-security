#!/bin/bash
# final_clean_collector.sh - Clean timing collector for MTD invariant discovery

LOGFILE="final_timing_data.csv"
PACKET_SIZES=(64 256 512)

initialize_logfile() {
    echo "timestamp,virtual_ip,packet_size,response_time_ms,status,epoch_id" > $LOGFILE
    echo "Initialized: $LOGFILE"
}

# Fast parallel discovery - only return clean IP list
fast_discovery() {
    local temp_file="/tmp/discovery_$$"
    
    # Clear temp file
    > "$temp_file"
    
    # Launch parallel pings silently
    for i in {20..49}; do
        (
            if ping -c 1 -W 2 "10.10.1.$i" >/dev/null 2>&1; then
                echo "10.10.1.$i"
            fi
        ) >> "$temp_file" &
    done
    
    # Wait for all background processes to complete
    wait
    
    # Return unique IPs only
    if [ -f "$temp_file" ]; then
        sort "$temp_file" | uniq | grep -E "^10\.10\.1\.[0-9]+$"
        rm -f "$temp_file"
    fi
}

# Measure response time for specific IP and packet size
measure_ip() {
    local virtual_ip="$1"
    local packet_size="$2" 
    local epoch_id="$3"
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S.%3N')
    local start_time=$(date +%s.%N)
    
    if ping -c 1 -s $((packet_size - 8)) -W 3 "$virtual_ip" >/dev/null 2>&1; then
        local end_time=$(date +%s.%N)
        local response_time=$(echo "($end_time - $start_time) * 1000" | bc -l)
        response_time=$(printf "%.3f" $response_time)
        
        echo "$timestamp,$virtual_ip,$packet_size,$response_time,success,$epoch_id" >> $LOGFILE
        printf "    %-12s (%3sB): %7.3fms\n" "$virtual_ip" "$packet_size" "$response_time"
        return 0
    else
        echo "$timestamp,$virtual_ip,$packet_size,timeout,failed,$epoch_id" >> $LOGFILE
        printf "    %-12s (%3sB): timeout\n" "$virtual_ip" "$packet_size"
        return 1
    fi
}

# Main timing collection loop
collect_timing() {
    local duration_minutes="$1"
    local end_time=$(($(date +%s) + duration_minutes * 60))
    local epoch_counter=1
    
    echo "=== MTD Timing Collection Started ==="
    echo "Duration: $duration_minutes minutes"
    echo "Packet sizes: ${PACKET_SIZES[*]} bytes"
    echo "Target: Find timing invariants across MTD rotations"
    echo ""
    
    while [ $(date +%s) -lt $end_time ]; do
        local cycle_start=$(date +%s.%N)
        local current_time=$(date '+%H:%M:%S')
        
        printf "[%s] Epoch %d - Fast Discovery\n" "$current_time" "$epoch_counter"
        
        # Fast parallel discovery phase
        local discovery_start=$(date +%s.%N)
        local active_ips_raw=$(fast_discovery)
        local discovery_time=$(echo "$(date +%s.%N) - $discovery_start" | bc -l)
        
        # Convert discovery results to array
        local active_ips=()
        while IFS= read -r ip; do
            if [ -n "$ip" ]; then
                active_ips+=("$ip")
            fi
        done <<< "$active_ips_raw"
        
        printf "  Discovery: %.2fs, Found %d IPs: %s\n" "$discovery_time" "${#active_ips[@]}" "${active_ips[*]}"
        
        # Measurement phase - test all packet sizes
        if [ ${#active_ips[@]} -gt 0 ]; then
            local measurement_start=$(date +%s.%N)
            
            for packet_size in "${PACKET_SIZES[@]}"; do
                echo "  Testing ${packet_size}B packets:"
                for ip in "${active_ips[@]}"; do
                    measure_ip "$ip" "$packet_size" "$epoch_counter"
                done
            done
            
            local measurement_time=$(echo "$(date +%s.%N) - $measurement_start" | bc -l)
            printf "  Measurement: %.2fs\n" "$measurement_time"
        else
            echo "  No active IPs found - possible MTD rotation in progress"
        fi
        
        # Calculate total cycle time and wait
        local total_cycle_time=$(echo "$(date +%s.%N) - $cycle_start" | bc -l)
        printf "  Epoch %d complete (%.2fs total)\n" "$epoch_counter" "$total_cycle_time"
        
        # Wait for next MTD cycle (aim for 25-second intervals)
        local wait_time=$(echo "25 - $total_cycle_time" | bc -l)
        if [ $(echo "$wait_time > 0" | bc -l) -eq 1 ]; then
            local wait_seconds=$(printf "%.0f" "$wait_time")
            printf "  Waiting %ds for next MTD rotation...\n" "$wait_seconds"
            sleep "$wait_seconds"
        else
            echo "  Cycle took longer than expected, starting next immediately"
        fi
        
        echo ""
        epoch_counter=$((epoch_counter + 1))
    done
    
    echo "=== Collection Complete ==="
    echo "Total epochs collected: $((epoch_counter - 1))"
    echo "Data saved to: $LOGFILE"
    echo ""
    echo "Run './final_clean_collector.sh stats' to analyze results"
}

# Show comprehensive statistics
show_stats() {
    if [ ! -f "$LOGFILE" ]; then
        echo "No data file found: $LOGFILE"
        echo "Run collection first: ./final_clean_collector.sh collect [minutes]"
        return
    fi
    
    echo "=== MTD Timing Analysis ==="
    echo ""
    
    # Basic counts
    local total_success=$(grep -c "success" "$LOGFILE")
    local total_failed=$(grep -c "failed" "$LOGFILE")
    local epochs=$(grep "success" "$LOGFILE" | cut -d',' -f6 | sort -n | uniq | wc -l)
    local unique_ips=$(grep "success" "$LOGFILE" | cut -d',' -f2 | sort | uniq | wc -l)
    
    echo "Collection Summary:"
    echo "  Successful measurements: $total_success"
    echo "  Failed measurements: $total_failed"
    echo "  Success rate: $(echo "scale=1; $total_success * 100 / ($total_success + $total_failed)" | bc -l)%"
    echo "  MTD epochs observed: $epochs"
    echo "  Unique virtual IPs seen: $unique_ips"
    echo "  Avg measurements per epoch: $(echo "scale=1; $total_success / $epochs" | bc -l)"
    echo ""
    
    # Response times by packet size
    echo "Response Time Analysis by Packet Size:"
    echo "Size | Count | Avg (ms) | Min (ms) | Max (ms) | Std Dev | Range"
    echo "-----|-------|----------|----------|----------|---------|-------"
    
    for size in "${PACKET_SIZES[@]}"; do
        local stats=$(grep "success.*,$size," "$LOGFILE" | cut -d',' -f4 | \
                     awk '{
                         sum+=$1; sumsq+=$1*$1; 
                         if(min=="" || $1<min) min=$1; 
                         if(max=="" || $1>max) max=$1; 
                         count++
                     } END {
                         if(count>0) {
                             avg=sum/count; 
                             if(count>1) stddev=sqrt((sumsq-sum*sum/count)/(count-1)); else stddev=0;
                             range=max-min;
                             printf "%d %.2f %.2f %.2f %.2f %.2f", count, avg, min, max, stddev, range
                         } else print "0 0 0 0 0 0"
                     }')
        
        if [ "$stats" != "0 0 0 0 0 0" ]; then
            read count avg min max stddev range <<< "$stats"
            printf "%4sB| %5s | %8s | %8s | %8s | %7s | %5s\n" "$size" "$count" "$avg" "$min" "$max" "$stddev" "$range"
        else
            printf "%4sB| %5s | %8s | %8s | %8s | %7s | %5s\n" "$size" "0" "No data" "-" "-" "-" "-"
        fi
    done
    echo ""
    
    # Virtual IP analysis - look for timing patterns
    echo "Virtual IP Timing Analysis (looking for invariants):"
    echo "Virtual IP   | Count | Avg Response | Min  | Max  | Potential Target"
    echo "-------------|-------|--------------|------|------|----------------"
    
    for ip in $(grep "success" "$LOGFILE" | cut -d',' -f2 | sort | uniq); do
        local ip_stats=$(grep "success.*$ip," "$LOGFILE" | cut -d',' -f4 | \
                        awk '{
                            sum+=$1; 
                            if(min=="" || $1<min) min=$1; 
                            if(max=="" || $1>max) max=$1; 
                            count++
                        } END {
                            if(count>0) printf "%d %.3f %.3f %.3f", count, sum/count, min, max
                            else print "0 0 0 0"
                        }')
        
        read ip_count ip_avg ip_min ip_max <<< "$ip_stats"
        
        if [ "$ip_count" != "0" ]; then
            # Classify by response time
            local target_guess=""
            if [ $(echo "$ip_avg < 0.8" | bc -l) -eq 1 ]; then
                target_guess="Target A (fast)"
            elif [ $(echo "$ip_avg > 1.0" | bc -l) -eq 1 ]; then
                target_guess="Target B (slow)"
            else
                target_guess="Target C (medium)"
            fi
            
            printf "%-12s | %5s | %12s | %4s | %4s | %s\n" "$ip" "$ip_count" "$ip_avg" "$ip_min" "$ip_max" "$target_guess"
        fi
    done
    echo ""
    
    # Epoch analysis
    echo "MTD Epoch Analysis:"
    echo "Epoch | IPs Found | Avg Response | Time Range"
    echo "------|-----------|--------------|------------"
    
    for epoch in $(grep "success" "$LOGFILE" | cut -d',' -f6 | sort -n | uniq | head -10); do
        local epoch_ips=$(grep "success.*,$epoch$" "$LOGFILE" | cut -d',' -f2 | sort | uniq | wc -l)
        local epoch_avg=$(grep "success.*,$epoch$" "$LOGFILE" | cut -d',' -f4 | \
                         awk '{sum+=$1; count++} END {if(count>0) printf "%.3f", sum/count; else print "N/A"}')
        local epoch_times=$(grep "success.*,$epoch$" "$LOGFILE" | cut -d',' -f1 | head -1 | cut -d' ' -f2 | cut -d':' -f1,2)
        
        printf "%5s | %9s | %12s | %s\n" "$epoch" "$epoch_ips" "$epoch_avg" "$epoch_times"
    done
    
    if [ $epochs -gt 10 ]; then
        echo "... (showing first 10 epochs)"
    fi
    echo ""
    
    # Invariant detection summary
    echo "=== INVARIANT DETECTION SUMMARY ==="
    echo ""
    
    # Group virtual IPs by timing characteristics
    local fast_ips=()
    local medium_ips=()
    local slow_ips=()
    
    for ip in $(grep "success" "$LOGFILE" | cut -d',' -f2 | sort | uniq); do
        local avg_time=$(grep "success.*$ip," "$LOGFILE" | cut -d',' -f4 | \
                        awk '{sum+=$1; count++} END {if(count>0) printf "%.3f", sum/count}')
        
        if [ -n "$avg_time" ]; then
            if [ $(echo "$avg_time < 0.8" | bc -l) -eq 1 ]; then
                fast_ips+=("$ip")
            elif [ $(echo "$avg_time > 1.0" | bc -l) -eq 1 ]; then
                slow_ips+=("$ip")
            else
                medium_ips+=("$ip")
            fi
        fi
    done
    
    echo "Timing-based Target Clusters:"
    echo "  Fast targets (<0.8ms):   ${#fast_ips[@]} IPs - ${fast_ips[*]}"
    echo "  Medium targets (0.8-1.0ms): ${#medium_ips[@]} IPs - ${medium_ips[*]}"
    echo "  Slow targets (>1.0ms):   ${#slow_ips[@]} IPs - ${slow_ips[*]}"
    echo ""
    
    if [ ${#fast_ips[@]} -gt 0 ] && [ ${#slow_ips[@]} -gt 0 ]; then
        echo "✅ TIMING INVARIANT DETECTED!"
        echo "   Virtual IPs cluster into distinct timing groups"
        echo "   This suggests persistent timing characteristics per target"
        echo "   despite MTD IP shuffling operations"
    else
        echo "⚠️  No clear timing separation detected"
        echo "   May need longer collection time or different measurement approach"
    fi
}

# Quick test of discovery function
test_discovery() {
    echo "=== Discovery Function Test ==="
    
    local start=$(date +%s.%N)
    local result=$(fast_discovery)
    local duration=$(echo "$(date +%s.%N) - $start" | bc -l)
    
    local ips=()
    while IFS= read -r ip; do
        if [ -n "$ip" ]; then
            ips+=("$ip")
        fi
    done <<< "$result"
    
    printf "Discovery completed in %.2f seconds\n" "$duration"
    printf "Found %d IPs: %s\n" "${#ips[@]}" "${ips[*]}"
    
    if [ ${#ips[@]} -eq 3 ]; then
        echo "✅ Perfect! Found exactly 3 IPs (as expected for MTD)"
    elif [ ${#ips[@]} -gt 3 ]; then
        echo "⚠️  Found more than 3 IPs - possible stale NAT rules"
    elif [ ${#ips[@]} -lt 3 ]; then
        echo "⚠️  Found fewer than 3 IPs - possible MTD rotation in progress"
    else
        echo "⚠️  No IPs found - check MTD controller status"
    fi
    
    # Test individual IP response times
    if [ ${#ips[@]} -gt 0 ]; then
        echo ""
        echo "Quick response time test:"
        for ip in "${ips[@]}"; do
            local start_ping=$(date +%s.%N)
            if ping -c 1 -W 2 "$ip" >/dev/null 2>&1; then
                local ping_time=$(echo "($(date +%s.%N) - $start_ping) * 1000" | bc -l)
                printf "  %s: %.2fms\n" "$ip" "$ping_time"
            else
                echo "  $ip: timeout"
            fi
        done
    fi
}

# Check dependencies
check_dependencies() {
    if ! command -v bc >/dev/null 2>&1; then
        echo "Installing bc for calculations..."
        sudo apt update && sudo apt install -y bc
    fi
}

# Main menu
case "${1:-menu}" in
    "collect")
        duration=${2:-10}
        check_dependencies
        initialize_logfile
        collect_timing "$duration"
        ;;
    "stats")
        show_stats
        ;;
    "test")
        test_discovery
        ;;
    "init")
        initialize_logfile
        ;;
    *)
        echo "MTD Timing Invariant Discovery Tool"
        echo ""
        echo "Usage: $0 [command] [options]"
        echo ""
        echo "Commands:"
        echo "  collect [minutes]  - Start timing collection (default: 10 min)"
        echo "  stats             - Analyze collected timing data"
        echo "  test              - Test discovery function"
        echo "  init              - Initialize data file"
        echo ""
        echo "Examples:"
        echo "  $0 test           # Test if discovery works"
        echo "  $0 collect 15     # Collect for 15 minutes"
        echo "  $0 stats          # Analyze results"
        echo ""
        echo "Goal: Discover timing-based invariants that persist"
        echo "      across MTD IP shuffling operations"
        ;;
esac
