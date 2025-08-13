#!/bin/bash
# data_inspector.sh - Debug and fix data parsing issues

inspect_data() {
    local datafile="${1:-final_timing_data.csv}"
    
    if [ ! -f "$datafile" ]; then
        echo "Data file not found: $datafile"
        echo "Available files:"
        ls -la *.csv 2>/dev/null || echo "No CSV files found"
        return
    fi
    
    echo "=== Data File Inspection: $datafile ==="
    echo ""
    
    # Show file info
    echo "File info:"
    echo "  Size: $(wc -l < "$datafile") lines"
    echo "  Created: $(stat -c %y "$datafile" 2>/dev/null || stat -f %Sm "$datafile" 2>/dev/null || echo "Unknown")"
    echo ""
    
    # Show header
    echo "Header line:"
    head -1 "$datafile"
    echo ""
    
    # Show structure
    echo "Column structure:"
    head -1 "$datafile" | tr ',' '\n' | nl
    echo ""
    
    # Show sample successful records
    echo "Sample successful records:"
    grep "success" "$datafile" | head -5
    echo ""
    
    # Show sample failed records  
    echo "Sample failed records:"
    grep "failed" "$datafile" | head -3
    echo ""
    
    # Count records by status
    echo "Record counts:"
    echo "  Total lines: $(wc -l < "$datafile")"
    echo "  Header lines: 1"
    echo "  Data lines: $(($(wc -l < "$datafile") - 1))"
    echo "  Success records: $(grep -c "success" "$datafile" 2>/dev/null || echo "0")"
    echo "  Failed records: $(grep -c "failed" "$datafile" 2>/dev/null || echo "0")"
    echo ""
    
    # Analyze packet sizes
    echo "Packet sizes found:"
    grep "success" "$datafile" | cut -d',' -f3 | sort | uniq -c | sort -nr
    echo ""
    
    # Analyze virtual IPs
    echo "Virtual IPs found:"
    grep "success" "$datafile" | cut -d',' -f2 | sort | uniq -c | sort -nr
    echo ""
    
    # Show response time range
    echo "Response time analysis:"
    local response_times=$(grep "success" "$datafile" | cut -d',' -f4 | grep -E '^[0-9]+\.[0-9]+$')
    if [ -n "$response_times" ]; then
        echo "$response_times" | awk '
        BEGIN { min=999999; max=0; sum=0; count=0 }
        { 
            if ($1 > 0) {
                sum+=$1; count++; 
                if ($1 < min) min=$1; 
                if ($1 > max) max=$1;
            }
        }
        END { 
            if (count > 0) {
                printf "  Min: %.3fms\n", min
                printf "  Max: %.3fms\n", max  
                printf "  Avg: %.3fms\n", sum/count
                printf "  Count: %d measurements\n", count
            } else {
                print "  No valid response times found"
            }
        }'
    else
        echo "  No valid response times found in column 4"
        echo "  Sample of column 4 values:"
        grep "success" "$datafile" | cut -d',' -f4 | head -5
    fi
}

# Fixed analysis function that works with actual data format
analyze_timing_data() {
    local datafile="${1:-final_timing_data.csv}"
    
    if [ ! -f "$datafile" ]; then
        echo "Data file not found: $datafile"
        return
    fi
    
    echo "=== Corrected Timing Analysis ==="
    echo ""
    
    # Determine CSV structure from header
    local header=$(head -1 "$datafile")
    echo "Data format: $header"
    echo ""
    
    # Basic counts
    local total_lines=$(($(wc -l < "$datafile") - 1))
    local success_count=$(grep -c "success" "$datafile" 2>/dev/null || echo "0")
    local failed_count=$(grep -c "failed" "$datafile" 2>/dev/null || echo "0")
    
    echo "Collection Summary:"
    echo "  Total data records: $total_lines"
    echo "  Successful: $success_count"
    echo "  Failed: $failed_count"
    if [ $total_lines -gt 0 ]; then
        echo "  Success rate: $(echo "scale=1; $success_count * 100 / $total_lines" | bc -l)%"
    fi
    echo ""
    
    # Packet size analysis (assuming column 3)
    echo "Response Times by Packet Size:"
    echo "Size | Count | Avg (ms) | Min (ms) | Max (ms)"
    echo "-----|-------|----------|----------|----------"
    
    for packet_size in 64 256 512; do
        local size_data=$(grep "success" "$datafile" | awk -F',' -v size="$packet_size" '$3 == size {print $4}' | grep -E '^[0-9]+\.[0-9]+$')
        
        if [ -n "$size_data" ]; then
            local stats=$(echo "$size_data" | awk '
                BEGIN { sum=0; min=999999; max=0; count=0 }
                { sum+=$1; count++; if($1<min) min=$1; if($1>max) max=$1 }
                END { if(count>0) printf "%d %.2f %.2f %.2f", count, sum/count, min, max; else print "0 0 0 0" }
            ')
            
            local count=$(echo $stats | cut -d' ' -f1)
            local avg=$(echo $stats | cut -d' ' -f2)
            local min=$(echo $stats | cut -d' ' -f3)
            local max=$(echo $stats | cut -d' ' -f4)
            
            printf "%4sB| %5s | %8s | %8s | %8s\n" "$packet_size" "$count" "$avg" "$min" "$max"
        else
            printf "%4sB| %5s | %8s | %8s | %8s\n" "$packet_size" "0" "No data" "-" "-"
        fi
    done
    echo ""
    
    # Virtual IP analysis
    echo "Virtual IP Timing Patterns:"
    echo "Virtual IP   | Count | Avg (ms) | Min (ms) | Max (ms) | Classification"
    echo "-------------|-------|----------|----------|----------|---------------"
    
    for ip in $(grep "success" "$datafile" | cut -d',' -f2 | sort | uniq); do
        local ip_times=$(grep "success.*$ip," "$datafile" | cut -d',' -f4 | grep -E '^[0-9]+\.[0-9]+$')
        
        if [ -n "$ip_times" ]; then
            local ip_stats=$(echo "$ip_times" | awk '
                BEGIN { sum=0; min=999999; max=0; count=0 }
                { sum+=$1; count++; if($1<min) min=$1; if($1>max) max=$1 }
                END { if(count>0) printf "%d %.3f %.3f %.3f", count, sum/count, min, max; else print "0 0 0 0" }
            ')
            
            local count=$(echo $ip_stats | cut -d' ' -f1)
            local avg=$(echo $ip_stats | cut -d' ' -f2)
            local min=$(echo $ip_stats | cut -d' ' -f3)
            local max=$(echo $ip_stats | cut -d' ' -f4)
            
            # Classify by timing
            local classification=""
            if [ $(echo "$avg < 3.7" | bc -l) -eq 1 ]; then
                classification="FAST"
            elif [ $(echo "$avg > 4.1" | bc -l) -eq 1 ]; then
                classification="SLOW" 
            else
                classification="MEDIUM"
            fi
            
            printf "%-12s | %5s | %8s | %8s | %8s | %s\n" "$ip" "$count" "$avg" "$min" "$max" "$classification"
        fi
    done
    echo ""
    
    # Look for timing invariants
    echo "=== INVARIANT ANALYSIS ==="
    echo ""
    
    # Group IPs by timing characteristics
    local fast_ips=0
    local medium_ips=0 
    local slow_ips=0
    
    for ip in $(grep "success" "$datafile" | cut -d',' -f2 | sort | uniq); do
        local avg_time=$(grep "success.*$ip," "$datafile" | cut -d',' -f4 | grep -E '^[0-9]+\.[0-9]+$' | \
                        awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
        
        if [ $(echo "$avg_time > 0" | bc -l) -eq 1 ]; then
            if [ $(echo "$avg_time < 3.7" | bc -l) -eq 1 ]; then
                fast_ips=$((fast_ips + 1))
            elif [ $(echo "$avg_time > 4.1" | bc -l) -eq 1 ]; then
                slow_ips=$((slow_ips + 1))
            else
                medium_ips=$((medium_ips + 1))
            fi
        fi
    done
    
    echo "Timing Clusters:"
    echo "  Fast IPs (<3.7ms): $fast_ips"
    echo "  Medium IPs (3.7-4.1ms): $medium_ips" 
    echo "  Slow IPs (>4.1ms): $slow_ips"
    echo ""
    
    if [ $fast_ips -gt 0 ] && [ $slow_ips -gt 0 ]; then
        echo "✅ TIMING INVARIANT DETECTED!"
        echo "   Clear separation between fast and slow response groups"
        echo "   This suggests persistent timing characteristics across MTD rotations"
    elif [ $medium_ips -gt 0 ] && [ $((fast_ips + slow_ips)) -gt 0 ]; then
        echo "⚠️  PARTIAL TIMING PATTERNS DETECTED"
        echo "   Some timing separation visible but not strongly clustered"
    else
        echo "⚠️  NO CLEAR TIMING SEPARATION"
        echo "   All virtual IPs show similar response times"
    fi
}

case "${1:-inspect}" in
    "inspect")
        datafile="${2:-final_timing_data.csv}"
        inspect_data "$datafile"
        ;;
    "analyze")
        datafile="${2:-final_timing_data.csv}"
        analyze_timing_data "$datafile"
        ;;
    "all")
        datafile="${2:-final_timing_data.csv}"
        inspect_data "$datafile"
        echo ""
        analyze_timing_data "$datafile"
        ;;
    *)
        echo "Data Inspector and Analyzer"
        echo ""
        echo "Usage: $0 [command] [datafile]"
        echo ""
        echo "Commands:"
        echo "  inspect [file]  - Inspect data file structure (default: final_timing_data.csv)"
        echo "  analyze [file]  - Analyze timing patterns with corrected parsing" 
        echo "  all [file]      - Both inspect and analyze (default)"
        echo ""
        echo "Examples:"
        echo "  $0 inspect                    # Inspect default file"
        echo "  $0 analyze timing_data.csv    # Analyze specific file"
        echo "  $0 all                        # Full analysis of default file"
        ;;
esac
