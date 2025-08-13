#!/bin/bash
# detailed_timing_analysis.sh - Look for subtle timing patterns

analyze_subtle_patterns() {
    local datafile="${1:-final_timing_data.csv}"
    
    if [ ! -f "$datafile" ]; then
        echo "Data file not found: $datafile"
        return
    fi
    
    echo "=== DETAILED TIMING PATTERN ANALYSIS ==="
    echo ""
    
    # First, let's see ALL virtual IPs and their timing stats
    echo "Complete Virtual IP Analysis (all IPs with >3 measurements):"
    echo "Virtual IP   | Count | Avg (ms) | StdDev | Min (ms) | Max (ms) | Range"
    echo "-------------|-------|----------|--------|----------|----------|-------"
    
    for ip in $(grep "success" "$datafile" | cut -d',' -f2 | sort | uniq); do
        local ip_times=$(grep "success.*$ip," "$datafile" | cut -d',' -f4 | grep -E '^[0-9]+\.[0-9]+$')
        local count=$(echo "$ip_times" | wc -l)
        
        if [ "$count" -ge 3 ]; then
            local stats=$(echo "$ip_times" | awk '
                BEGIN { sum=0; sumsq=0; min=999; max=0; count=0 }
                { 
                    sum+=$1; sumsq+=$1*$1; count++; 
                    if($1<min) min=$1; if($1>max) max=$1 
                }
                END { 
                    if(count>0) {
                        avg=sum/count; 
                        stddev = count>1 ? sqrt((sumsq-sum*sum/count)/(count-1)) : 0;
                        range=max-min;
                        printf "%.3f %.3f %.3f %.3f %.3f", avg, stddev, min, max, range
                    }
                }')
            
            local avg=$(echo $stats | cut -d' ' -f1)
            local stddev=$(echo $stats | cut -d' ' -f2) 
            local min=$(echo $stats | cut -d' ' -f3)
            local max=$(echo $stats | cut -d' ' -f4)
            local range=$(echo $stats | cut -d' ' -f5)
            
            printf "%-12s | %5d | %8s | %6s | %8s | %8s | %5s\n" "$ip" "$count" "$avg" "$stddev" "$min" "$max" "$range"
        fi
    done | sort -k3 -n  # Sort by average response time
    echo ""
    
    # Look for packet size effects per virtual IP
    echo "Packet Size Effects Analysis:"
    echo "Virtual IP   | 64B Avg | 256B Avg | 512B Avg | Size Effect"
    echo "-------------|---------|----------|----------|------------"
    
    for ip in $(grep "success" "$datafile" | cut -d',' -f2 | sort | uniq); do
        local count_64=$(grep "success.*$ip,64," "$datafile" | wc -l)
        local count_256=$(grep "success.*$ip,256," "$datafile" | wc -l)
        local count_512=$(grep "success.*$ip,512," "$datafile" | wc -l)
        
        if [ "$count_64" -ge 1 ] && [ "$count_256" -ge 1 ] && [ "$count_512" -ge 1 ]; then
            local avg_64=$(grep "success.*$ip,64," "$datafile" | cut -d',' -f4 | awk '{sum+=$1; count++} END {printf "%.3f", sum/count}')
            local avg_256=$(grep "success.*$ip,256," "$datafile" | cut -d',' -f4 | awk '{sum+=$1; count++} END {printf "%.3f", sum/count}')
            local avg_512=$(grep "success.*$ip,512," "$datafile" | cut -d',' -f4 | awk '{sum+=$1; count++} END {printf "%.3f", sum/count}')
            
            # Calculate size effect (difference between max and min packet size averages)
            local size_effect=$(echo "$avg_64 $avg_256 $avg_512" | awk '{
                min=$1; max=$1;
                if($2<min) min=$2; if($2>max) max=$2;
                if($3<min) min=$3; if($3>max) max=$3;
                printf "%.3f", max-min
            }')
            
            printf "%-12s | %7s | %8s | %8s | %s\n" "$ip" "$avg_64" "$avg_256" "$avg_512" "$size_effect"
        fi
    done | sort -k5 -nr  # Sort by size effect
    echo ""
    
    # Temporal analysis - look for timing changes over epochs
    echo "Temporal Pattern Analysis (first 10 epochs):"
    echo "Epoch | IPs Count | Avg Time | StdDev | Min    | Max    | Characteristics"
    echo "------|-----------|----------|--------|--------|--------|----------------"
    
    for epoch in $(grep "success" "$datafile" | cut -d',' -f6 | sort -n | uniq | head -10); do
        local epoch_times=$(grep "success.*,$epoch$" "$datafile" | cut -d',' -f4)
        local epoch_ips=$(grep "success.*,$epoch$" "$datafile" | cut -d',' -f2 | sort | uniq | wc -l)
        
        if [ -n "$epoch_times" ]; then
            local epoch_stats=$(echo "$epoch_times" | awk '
                BEGIN { sum=0; sumsq=0; min=999; max=0; count=0 }
                { 
                    sum+=$1; sumsq+=$1*$1; count++; 
                    if($1<min) min=$1; if($1>max) max=$1 
                }
                END { 
                    if(count>0) {
                        avg=sum/count; 
                        stddev = count>1 ? sqrt((sumsq-sum*sum/count)/(count-1)) : 0;
                        printf "%.3f %.3f %.3f %.3f", avg, stddev, min, max
                    }
                }')
            
            local avg=$(echo $epoch_stats | cut -d' ' -f1)
            local stddev=$(echo $epoch_stats | cut -d' ' -f2)
            local min=$(echo $epoch_stats | cut -d' ' -f3)
            local max=$(echo $epoch_stats | cut -d' ' -f4)
            
            # Characterize the epoch
            local characteristics=""
            if [ $(echo "$avg < 3.85" | bc -l) -eq 1 ]; then
                characteristics="Fast epoch"
            elif [ $(echo "$avg > 4.05" | bc -l) -eq 1 ]; then
                characteristics="Slow epoch"
            else
                characteristics="Normal"
            fi
            
            printf "%5s | %9s | %8s | %6s | %6s | %6s | %s\n" "$epoch" "$epoch_ips" "$avg" "$stddev" "$min" "$max" "$characteristics"
        fi
    done
    echo ""
    
    # Statistical clustering analysis
    echo "=== STATISTICAL CLUSTERING ANALYSIS ==="
    echo ""
    
    # Calculate overall statistics
    local all_times=$(grep "success" "$datafile" | cut -d',' -f4)
    local overall_stats=$(echo "$all_times" | awk '
        BEGIN { sum=0; sumsq=0; min=999; max=0; count=0 }
        { sum+=$1; sumsq+=$1*$1; count++; if($1<min) min=$1; if($1>max) max=$1 }
        END { 
            if(count>0) {
                avg=sum/count; 
                stddev = sqrt((sumsq-sum*sum/count)/(count-1));
                printf "%.3f %.3f %.3f %.3f %d", avg, stddev, min, max, count
            }
        }')
    
    local overall_avg=$(echo $overall_stats | cut -d' ' -f1)
    local overall_stddev=$(echo $overall_stats | cut -d' ' -f2)
    local overall_min=$(echo $overall_stats | cut -d' ' -f3)
    local overall_max=$(echo $overall_stats | cut -d' ' -f4)
    local overall_count=$(echo $overall_stats | cut -d' ' -f5)
    
    echo "Overall Timing Statistics:"
    echo "  Total measurements: $overall_count"
    echo "  Average: ${overall_avg}ms"
    echo "  Standard deviation: ${overall_stddev}ms"
    echo "  Range: ${overall_min}ms - ${overall_max}ms"
    echo "  Coefficient of variation: $(echo "scale=3; $overall_stddev / $overall_avg * 100" | bc -l)%"
    echo ""
    
    # Look for IPs that are consistently faster or slower than average
    echo "Virtual IPs Relative to Overall Average (${overall_avg}ms):"
    echo "Virtual IP   | Avg Time | Deviation | Consistency | Classification"
    echo "-------------|----------|-----------|-------------|---------------"
    
    for ip in $(grep "success" "$datafile" | cut -d',' -f2 | sort | uniq); do
        local ip_times=$(grep "success.*$ip," "$datafile" | cut -d',' -f4)
        local count=$(echo "$ip_times" | wc -l)
        
        if [ "$count" -ge 3 ]; then
            local ip_avg=$(echo "$ip_times" | awk '{sum+=$1; count++} END {printf "%.3f", sum/count}')
            local deviation=$(echo "$ip_avg - $overall_avg" | bc -l)
            local abs_deviation=$(echo "$deviation" | awk '{print ($1<0) ? -$1 : $1}')
            
            # Calculate consistency (how often this IP is on the same side of average)
            local consistency=$(echo "$ip_times" | awk -v avg="$overall_avg" '
                BEGIN { above=0; below=0; total=0 }
                { 
                    total++; 
                    if($1 > avg) above++; else below++; 
                }
                END { 
                    max_side = (above > below) ? above : below;
                    printf "%.1f", max_side/total*100
                }')
            
            # Classify based on deviation and consistency
            local classification=""
            if [ $(echo "$abs_deviation > 0.05" | bc -l) -eq 1 ] && [ $(echo "$consistency > 70" | bc -l) -eq 1 ]; then
                if [ $(echo "$deviation > 0" | bc -l) -eq 1 ]; then
                    classification="CONSISTENTLY SLOW"
                else
                    classification="CONSISTENTLY FAST"
                fi
            elif [ $(echo "$abs_deviation > 0.05" | bc -l) -eq 1 ]; then
                classification="SOMEWHAT DIFFERENT"
            else
                classification="AVERAGE"
            fi
            
            printf "%-12s | %8s | %+8.3f | %10s%% | %s\n" "$ip" "$ip_avg" "$deviation" "$consistency" "$classification"
        fi
    done | sort -k3 -n
    echo ""
    
    # Final assessment
    echo "=== FINAL ASSESSMENT ==="
    echo ""
    
    # Count IPs in each category
    local fast_consistent=0
    local slow_consistent=0
    local different=0
    local average=0
    
    for ip in $(grep "success" "$datafile" | cut -d',' -f2 | sort | uniq); do
        local ip_times=$(grep "success.*$ip," "$datafile" | cut -d',' -f4)
        local count=$(echo "$ip_times" | wc -l)
        
        if [ "$count" -ge 3 ]; then
            local ip_avg=$(echo "$ip_times" | awk '{sum+=$1; count++} END {print sum/count}')
            local deviation=$(echo "$ip_avg - $overall_avg" | bc -l)
            local abs_deviation=$(echo "$deviation" | awk '{print ($1<0) ? -$1 : $1}')
            
            local consistency=$(echo "$ip_times" | awk -v avg="$overall_avg" '
                BEGIN { above=0; below=0; total=0 }
                { total++; if($1 > avg) above++; else below++; }
                END { max_side = (above > below) ? above : below; print max_side/total*100 }')
            
            if [ $(echo "$abs_deviation > 0.05" | bc -l) -eq 1 ] && [ $(echo "$consistency > 70" | bc -l) -eq 1 ]; then
                if [ $(echo "$deviation > 0" | bc -l) -eq 1 ]; then
                    slow_consistent=$((slow_consistent + 1))
                else
                    fast_consistent=$((fast_consistent + 1))
                fi
            elif [ $(echo "$abs_deviation > 0.05" | bc -l) -eq 1 ]; then
                different=$((different + 1))
            else
                average=$((average + 1))
            fi
        fi
    done
    
    echo "Virtual IP Classification Summary:"
    echo "  Consistently Fast (>70% below avg): $fast_consistent IPs"
    echo "  Consistently Slow (>70% above avg): $slow_consistent IPs"
    echo "  Somewhat Different: $different IPs"
    echo "  Average/Neutral: $average IPs"
    echo ""
    
    if [ $fast_consistent -gt 0 ] && [ $slow_consistent -gt 0 ]; then
        echo "✅ SUBTLE TIMING INVARIANT DETECTED!"
        echo "   While timing differences are small (<0.5ms), there are"
        echo "   consistently faster and slower virtual IP groups."
        echo "   This suggests underlying infrastructure differences persist"
        echo "   across MTD operations despite IP address changes."
        echo ""
        echo "🔍 RESEARCH IMPLICATIONS:"
        echo "   - Timing-based re-identification may be possible with sufficient sampling"
        echo "   - MTD provides some protection by making differences very subtle"
        echo "   - Attackers would need extended observation periods"
    elif [ $((fast_consistent + slow_consistent + different)) -gt $average ]; then
        echo "⚠️ WEAK TIMING PATTERNS DETECTED"
        echo "   Some timing differences exist but are not strongly consistent"
        echo "   May indicate partial timing-based fingerprinting possibility"
    else
        echo "✅ STRONG MTD PROTECTION OBSERVED"
        echo "   No significant timing patterns detected across virtual IPs"
        echo "   This suggests effective timing normalization by the MTD system"
        echo "   or very similar underlying infrastructure characteristics"
    fi
}

case "${1:-analyze}" in
    "analyze")
        datafile="${2:-final_timing_data.csv}"
        analyze_subtle_patterns "$datafile"
        ;;
    *)
        echo "Detailed Timing Pattern Analysis"
        echo ""
        echo "Usage: $0 analyze [datafile]"
        echo ""
        echo "Performs deep statistical analysis to detect subtle timing patterns"
        echo "that might indicate persistent characteristics across MTD rotations."
        echo ""
        echo "Default file: final_timing_data.csv"
        ;;
esac
