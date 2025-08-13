#!/bin/bash
# invariant_validation.sh - Validate discovered timing invariants against actual mappings

echo "=== TIMING INVARIANT VALIDATION ==="
echo ""

# Create mapping from the MTD controller logs
# Target 10.10.1.3 (Real Target A):
target_a_ips="10.10.1.34 10.10.1.42 10.10.1.21 10.10.1.47 10.10.1.26 10.10.1.24 10.10.1.45 10.10.1.23 10.10.1.31 10.10.1.29"

# Target 10.10.1.4 (Real Target B):  
target_b_ips="10.10.1.22 10.10.1.45 10.10.1.20 10.10.1.30 10.10.1.38 10.10.1.33 10.10.1.43 10.10.1.25 10.10.1.28 10.10.1.40 10.10.1.37 10.10.1.49 10.10.1.34 10.10.1.39"

# Target 10.10.1.5 (Real Target C):
target_c_ips="10.10.1.44 10.10.1.34 10.10.1.43 10.10.1.20 10.10.1.39 10.10.1.47 10.10.1.41 10.10.1.25 10.10.1.37 10.10.1.30 10.10.1.24 10.10.1.28"

echo "Analyzing response times by ACTUAL target assignment:"
echo ""

analyze_target() {
    local target_name="$1"
    local real_ip="$2"
    local virtual_ips="$3"
    
    echo "=== $target_name (Real IP: $real_ip) ==="
    echo "Virtual IP  | Response Times (ms) | Average"
    echo "------------|--------------------|---------"
    
    total_sum=0
    total_count=0
    all_times=""
    
    for vip in $virtual_ips; do
        # Get response times for this virtual IP from timing_data.csv
        times=$(grep "$vip.*success" timing_data.csv 2>/dev/null | cut -d',' -f3 | tr '\n' ' ')
        if [ -n "$times" ]; then
            count=$(echo $times | wc -w)
            if [ $count -gt 0 ]; then
                avg=$(echo $times | tr ' ' '\n' | awk '{sum+=$1; n++} END {if(n>0) printf "%.2f", sum/n}')
                printf "%-11s | %-18s | %s\n" "$vip" "$times" "$avg"
                
                # Add to overall statistics
                for time in $times; do
                    total_sum=$(echo "$total_sum + $time" | bc -l)
                    total_count=$((total_count + 1))
                    all_times="$all_times $time"
                done
            fi
        fi
    done
    
    if [ $total_count -gt 0 ]; then
        overall_avg=$(echo "scale=2; $total_sum / $total_count" | bc -l)
        min_time=$(echo $all_times | tr ' ' '\n' | grep -v '^$' | sort -n | head -1)
        max_time=$(echo $all_times | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1)
        
        echo "------------|--------------------|---------"
        echo "OVERALL     | Count: $total_count           | $overall_avg"
        echo "            | Range: $min_time - $max_time    |"
        
        # Calculate standard deviation
        if [ $total_count -gt 1 ]; then
            stddev=$(echo $all_times | tr ' ' '\n' | grep -v '^$' | awk -v avg=$overall_avg '
                {sum += ($1 - avg)^2} 
                END {if(NR>1) printf "%.2f", sqrt(sum/(NR-1)); else print "0.00"}')
            echo "            | Std Dev: $stddev        |"
        fi
    else
        echo "No timing data found for this target"
    fi
    echo ""
}

# Analyze each target
analyze_target "TARGET A" "10.10.1.3" "$target_a_ips"
analyze_target "TARGET B" "10.10.1.4" "$target_b_ips" 
analyze_target "TARGET C" "10.10.1.5" "$target_c_ips"

echo "=== INVARIANT DISCOVERY RESULTS ==="
echo ""
echo "Summary of timing characteristics by real target:"
echo ""

# Calculate overall averages for each real target
for target in A B C; do
    case $target in
        A) real_ip="10.10.1.3"; vips="$target_a_ips" ;;
        B) real_ip="10.10.1.4"; vips="$target_b_ips" ;;
        C) real_ip="10.10.1.5"; vips="$target_c_ips" ;;
    esac
    
    all_times=""
    for vip in $vips; do
        times=$(grep "$vip.*success" timing_data.csv 2>/dev/null | cut -d',' -f3)
        if [ -n "$times" ]; then
            all_times="$all_times $times"
        fi
    done
    
    if [ -n "$all_times" ]; then
        avg=$(echo $all_times | tr ' ' '\n' | grep -v '^$' | awk '{sum+=$1; n++} END {if(n>0) printf "%.2f", sum/n}')
        count=$(echo $all_times | wc -w)
        echo "TARGET $target ($real_ip): $avg ms average ($count measurements)"
    fi
done

echo ""
echo "=== CONCLUSION ==="
echo ""
echo "If the timing averages are clearly separated between targets,"
echo "we have discovered a TIMING-BASED INVARIANT that persists"
echo "across MTD IP shuffling operations!"
echo ""
echo "This proves that despite network address randomization,"
echo "underlying infrastructure characteristics create persistent"
echo "timing signatures that can be used for target re-identification."
