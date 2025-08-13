#!/bin/bash
# fixed_parallel_collector.sh - Fixed parallel protocol collector

LOGFILE="fixed_protocol_data.csv"
DURATION_MINUTES=10

initialize_logfile() {
    echo "timestamp,virtual_ip,protocol,service_type,server_header,response_time_ms,status,epoch_id,additional_info" > $LOGFILE
    echo "Initialized: $LOGFILE"
}

# Fixed parallel discovery - only return valid IP addresses
fixed_parallel_discovery() {
    local temp_file="/tmp/clean_discovery_$$"
    > "$temp_file"
    
    # Launch parallel ping tests
    for i in {20..49}; do
        (
            if ping -c 1 -W 1 "10.10.1.$i" >/dev/null 2>&1; then
                echo "10.10.1.$i"
            fi
        ) >> "$temp_file" &
    done
    
    # Wait for all processes
    wait
    
    # Return only valid IP addresses (filter out any non-IP content)
    if [ -f "$temp_file" ]; then
        grep -E "^10\.10\.1\.[0-9]+$" "$temp_file" | sort | uniq
        rm -f "$temp_file"
    fi
}

# Simple HTTP test
test_http_simple() {
    local virtual_ip="$1"
    local epoch_id="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S.%3N')
    
    local start_time=$(date +%s.%N)
    
    # Test with longer timeout and more robust approach
    local http_response=$(timeout 5 curl -s -I "http://$virtual_ip" --connect-timeout 3 --max-time 4 2>/dev/null)
    local curl_exit=$?
    
    local end_time=$(date +%s.%N)
    local response_time=$(echo "($end_time - $start_time) * 1000" | bc -l)
    response_time=$(printf "%.3f" $response_time)
    
    if [ $curl_exit -eq 0 ] && [ -n "$http_response" ]; then
        # Extract server header
        local server_header=$(echo "$http_response" | grep -i "^Server:" | sed 's/Server: //i' | tr -d '\r\n' | head -1)
        
        # Determine service type
        local service_type="unknown"
        if echo "$server_header" | grep -qi "apache"; then
            service_type="apache"
        elif echo "$server_header" | grep -qi "nginx"; then
            service_type="nginx"
        elif [ -n "$server_header" ]; then
            service_type="http_other"
        fi
        
        echo "$timestamp,$virtual_ip,HTTP,$service_type,$server_header,$response_time,success,$epoch_id,detected" >> $LOGFILE
        printf "    %-12s HTTP: %-10s (%.3fms)\n" "$virtual_ip" "$service_type" "$response_time"
    else
        echo "$timestamp,$virtual_ip,HTTP,failed,none,timeout,failed,$epoch_id,no_response" >> $LOGFILE
        printf "    %-12s HTTP: timeout\n" "$virtual_ip"
    fi
}

# Simple SSH test
test_ssh_simple() {
    local virtual_ip="$1"
    local epoch_id="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S.%3N')
    
    local start_time=$(date +%s.%N)
    
    # Simple SSH version detection
    local ssh_version=$(timeout 3 nc -w 2 "$virtual_ip" 22 2>/dev/null | head -1 | tr -d '\r\n')
    
    local end_time=$(date +%s.%N)
    local response_time=$(echo "($end_time - $start_time) * 1000" | bc -l)
    response_time=$(printf "%.3f" $response_time)
    
    if [ -n "$ssh_version" ] && echo "$ssh_version" | grep -q "SSH-"; then
        local service_type="openssh"
        
        # Try to get banner with quick connection
        local banner_check=$(timeout 2 ssh -o ConnectTimeout=1 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$virtual_ip" 2>&1 | grep -E "SSH Server|Permission denied" | head -1)
        
        if echo "$banner_check" | grep -q "SSH Server"; then
            service_type="openssh_custom"
            ssh_version="Custom_Banner"
        fi
        
        echo "$timestamp,$virtual_ip,SSH,$service_type,$ssh_version,$response_time,success,$epoch_id,version_detected" >> $LOGFILE
        printf "    %-12s SSH:  %-10s (%.3fms)\n" "$virtual_ip" "$service_type" "$response_time"
    else
        echo "$timestamp,$virtual_ip,SSH,failed,none,timeout,failed,$epoch_id,no_ssh" >> $LOGFILE
        printf "    %-12s SSH:  timeout\n" "$virtual_ip"
    fi
}

# Test all protocols for all IPs
test_all_protocols() {
    local epoch_id="$1"
    shift
    local ips=("$@")
    
    echo "  Testing ${#ips[@]} IPs with both HTTP and SSH:"
    
    # Test HTTP first for all IPs
    for ip in "${ips[@]}"; do
        test_http_simple "$ip" "$epoch_id"
    done
    
    echo ""
    
    # Test SSH for all IPs
    for ip in "${ips[@]}"; do
        test_ssh_simple "$ip" "$epoch_id"
    done
}

# Fixed main collection loop
collect_fixed_protocol_invariants() {
    local duration_minutes="$1"
    local end_time=$(($(date +%s) + duration_minutes * 60))
    local epoch_counter=1
    
    echo "=== FIXED Protocol Invariant Collection ==="
    echo "Duration: $duration_minutes minutes"
    echo "Testing: HTTP (server detection) + SSH (banner detection)"
    echo ""
    
    while [ $(date +%s) -lt $end_time ]; do
        local cycle_start=$(date +%s.%N)
        local current_time=$(date '+%H:%M:%S')
        
        printf "[%s] Epoch %d\n" "$current_time" "$epoch_counter"
        
        # Clean discovery
        local discovery_start=$(date +%s.%N)
        local active_ips_str=$(fixed_parallel_discovery)
        local active_ips=($active_ips_str)
        local discovery_time=$(echo "$(date +%s.%N) - $discovery_start" | bc -l)
        
        printf "  Discovery: %.2fs, Found %d IPs: %s\n" "$discovery_time" "${#active_ips[@]}" "${active_ips[*]}"
        
        # Protocol testing
        if [ ${#active_ips[@]} -eq 3 ]; then
            test_all_protocols "$epoch_counter" "${active_ips[@]}"
        elif [ ${#active_ips[@]} -gt 0 ]; then
            echo "  Warning: Found ${#active_ips[@]} IPs (expected 3)"
            test_all_protocols "$epoch_counter" "${active_ips[@]}"
        else
            echo "  No active IPs found"
        fi
        
        local total_cycle_time=$(echo "$(date +%s.%N) - $cycle_start" | bc -l)
        printf "\n  Epoch %d complete (%.2fs total)\n" "$epoch_counter" "$total_cycle_time"
        
        # Wait for next cycle
        local wait_time=$(echo "25 - $total_cycle_time" | bc -l)
        if [ $(echo "$wait_time > 0" | bc -l) -eq 1 ]; then
            local wait_seconds=$(printf "%.0f" "$wait_time")
            printf "  Waiting %ds for next MTD rotation...\n\n" "$wait_seconds"
            sleep "$wait_seconds"
        else
            echo ""
        fi
        
        epoch_counter=$((epoch_counter + 1))
    done
    
    echo "=== Collection Complete ==="
}

# Quick analysis
analyze_fixed_results() {
    if [ ! -f "$LOGFILE" ]; then
        echo "No data file found: $LOGFILE"
        return
    fi
    
    echo "=== Protocol Invariant Analysis ==="
    echo ""
    
    # Basic stats
    local total_tests=$(($(wc -l < "$LOGFILE") - 1))
    local success_tests=$(grep -c "success" "$LOGFILE")
    local epochs=$(grep "success" "$LOGFILE" | cut -d',' -f8 | sort -n | uniq | wc -l)
    
    echo "Collection Summary:"
    echo "  Total tests: $total_tests"
    echo "  Successful: $success_tests"
    echo "  Success rate: $(echo "scale=1; $success_tests * 100 / $total_tests" | bc -l)%"
    echo "  Epochs: $epochs"
    echo ""
    
    # Service detection
    echo "Service Detection:"
    echo "Service Type | Count | Virtual IPs"
    echo "-------------|-------|----------------------------------"
    
    for service in apache nginx openssh openssh_custom http_other; do
        local count=$(grep -c ",$service," "$LOGFILE")
        if [ $count -gt 0 ]; then
            local ips=$(grep ",$service," "$LOGFILE" | cut -d',' -f2 | sort | uniq | tr '\n' ' ')
            printf "%-12s | %5d | %s\n" "$service" "$count" "$ips"
        fi
    done
    
    echo ""
    
    # Show recent detections
    echo "Recent Successful Detections:"
    grep "success" "$LOGFILE" | tail -10 | while IFS=',' read ts ip proto service header time status epoch info; do
        printf "  %s | %-12s | %-4s | %s\n" "$(echo $ts | cut -d' ' -f2)" "$ip" "$proto" "$service"
    done
}

# Test connectivity first
test_services() {
    echo "=== Testing Service Connectivity ==="
    
    # Check if MTD is running
    local active_ips=$(fixed_parallel_discovery)
    echo "Currently active virtual IPs: $active_ips"
    
    if [ -z "$active_ips" ]; then
        echo "ERROR: No active virtual IPs found!"
        echo "Make sure MTD controller is running"
        return 1
    fi
    
    echo ""
    echo "Testing services on active IPs:"
    
    for ip in $active_ips; do
        echo "Testing $ip:"
        
        # Test HTTP
        if curl -s -I "http://$ip" --connect-timeout 2 --max-time 3 >/dev/null 2>&1; then
            local server=$(curl -s -I "http://$ip" --connect-timeout 2 | grep -i "^Server:" | head -1)
            echo "  HTTP: ✓ $server"
        else
            echo "  HTTP: ✗ Failed"
        fi
        
        # Test SSH
        if nc -w 2 "$ip" 22 </dev/null 2>/dev/null | grep -q "SSH-"; then
            echo "  SSH:  ✓ Responding"
        else
            echo "  SSH:  ✗ Failed"
        fi
        
        echo ""
    done
}

# Main menu
case "${1:-menu}" in
    "collect")
        duration=${2:-$DURATION_MINUTES}
        initialize_logfile
        collect_fixed_protocol_invariants "$duration"
        ;;
    "analyze")
        analyze_fixed_results
        ;;
    "test")
        test_services
        ;;
    *)
        echo "Fixed Protocol Invariant Collector"
        echo ""
        echo "Usage: $0 [command] [options]"
        echo ""
        echo "Commands:"
        echo "  test              - Test service connectivity first"
        echo "  collect [minutes] - Start collection (default: 10 min)"
        echo "  analyze          - Analyze results"
        echo ""
        ;;
esac
