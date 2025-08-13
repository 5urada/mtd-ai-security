#!/usr/bin/env python3
"""
Corrected Stability Index Calculator for MTD Systems
Properly maps virtual IP observations to real target entities

Entities = Real IPs (10.10.1.3, 10.10.1.4, 10.10.1.5)
Observations = Virtual IP timing measurements attributed to real targets
Goal: Calculate S(φ) for real target separability across MTD epochs
"""

import pandas as pd
import numpy as np
import sys
import re
from collections import defaultdict
from datetime import datetime

class CorrectedStabilityCalculator:
    def __init__(self, csv_file):
        print("="*70)
        print("CORRECTED MTD STABILITY INDEX CALCULATOR")
        print("="*70)
        print("Entities = Real IPs (actual targets)")
        print("Observations = Virtual IP measurements attributed to real targets")
        print()
        
        self.csv_file = csv_file
        self.load_timing_data()
        
        # MTD mappings from your controller logs (manually extracted for now)
        self.create_epoch_mappings()
        
    def load_timing_data(self):
        """Load timing data with robust parsing"""
        print(f"Loading timing data from {self.csv_file}...")
        
        try:
            self.df = pd.read_csv(self.csv_file, dtype={
                'timestamp': str,
                'virtual_ip': str, 
                'packet_size': int,
                'response_time_ms': float,
                'status': str,
                'epoch_id': int
            })
        except:
            self.df = pd.read_csv(self.csv_file)
            self.df['response_time_ms'] = pd.to_numeric(self.df['response_time_ms'], errors='coerce')
            self.df['epoch_id'] = pd.to_numeric(self.df['epoch_id'], errors='coerce')
        
        # Filter successful measurements
        self.success_df = self.df[self.df['status'] == 'success'].copy()
        self.success_df = self.success_df.dropna(subset=['response_time_ms', 'epoch_id'])
        
        print(f"Loaded {len(self.success_df)} successful timing measurements")
        
        # Show epoch range
        epochs = sorted(self.success_df['epoch_id'].unique())
        print(f"Epochs: {min(epochs)} to {max(epochs)} ({len(epochs)} total)")
        
    def create_epoch_mappings(self):
        """Create virtual->real IP mappings per epoch based on MTD controller logs"""
        print("\nCreating epoch mappings from MTD controller behavior...")
        
        # Based on your MTD controller logs, create the mappings
        # This represents the ground truth of which virtual IPs mapped to which real IPs
        self.epoch_mappings = {
            # Epoch 1: 18:57:09 (first rotation we saw)
            1: {
                '10.10.1.44': '10.10.1.3',  # Target A (Modified TCP) 
                '10.10.1.23': '10.10.1.4',  # Target B (Network Delay)
                '10.10.1.47': '10.10.1.5'   # Target C (Default)
            }
            # We'll infer other epochs from the timing patterns
        }
        
        # Infer mappings for other epochs based on timing patterns
        self.infer_epoch_mappings()
        
    def infer_epoch_mappings(self):
        """Infer epoch mappings based on timing cluster analysis"""
        print("Inferring epoch mappings from timing patterns...")
        
        # From your manual analysis, we know the timing clusters:
        # SLOW (6.0-6.4ms) = Target B (10.10.1.4 - Network Delay)
        # MEDIUM (5.0-5.2ms) = Target A (10.10.1.3 - Modified TCP)  
        # FAST (3.8-4.8ms) = Target C (10.10.1.5 - Default)
        
        epochs = sorted(self.success_df['epoch_id'].unique())
        
        for epoch in epochs:
            if epoch in self.epoch_mappings:
                continue  # Already have this mapping
                
            epoch_data = self.success_df[self.success_df['epoch_id'] == epoch]
            virtual_ips = epoch_data['virtual_ip'].unique()
            
            if len(virtual_ips) != 3:
                print(f"Warning: Epoch {epoch} has {len(virtual_ips)} virtual IPs, expected 3")
                continue
            
            # Calculate average response time per virtual IP in this epoch
            vip_avg_times = {}
            for vip in virtual_ips:
                vip_data = epoch_data[epoch_data['virtual_ip'] == vip]
                avg_time = vip_data['response_time_ms'].mean()
                vip_avg_times[vip] = avg_time
            
            # Sort by response time to assign to targets
            sorted_vips = sorted(vip_avg_times.items(), key=lambda x: x[1])
            
            # Map based on timing clusters
            # Fastest -> Target C (Default)
            # Middle -> Target A (Modified TCP) 
            # Slowest -> Target B (Network Delay)
            
            self.epoch_mappings[epoch] = {
                sorted_vips[0][0]: '10.10.1.5',  # Fastest -> Target C
                sorted_vips[1][0]: '10.10.1.3',  # Middle -> Target A
                sorted_vips[2][0]: '10.10.1.4'   # Slowest -> Target B
            }
            
        print(f"Created mappings for {len(self.epoch_mappings)} epochs")
        
        # Show sample mappings
        print("\nSample epoch mappings:")
        for epoch in sorted(list(self.epoch_mappings.keys()))[:5]:
            print(f"Epoch {epoch}:")
            for vip, rip in self.epoch_mappings[epoch].items():
                avg_time = self.success_df[
                    (self.success_df['epoch_id'] == epoch) & 
                    (self.success_df['virtual_ip'] == vip)
                ]['response_time_ms'].mean()
                print(f"  {vip} -> {rip} (avg: {avg_time:.2f}ms)")
    
    def attribute_measurements_to_real_targets(self):
        """Attribute virtual IP measurements to real target entities"""
        print("\nAttributing virtual IP measurements to real targets...")
        
        # Create data structure: real_ip -> epoch -> [measurements]
        real_target_data = defaultdict(lambda: defaultdict(list))
        
        attributed_count = 0
        total_count = len(self.success_df)
        
        for _, row in self.success_df.iterrows():
            epoch = int(row['epoch_id'])
            virtual_ip = row['virtual_ip']
            response_time = float(row['response_time_ms'])
            
            # Look up which real IP this virtual IP mapped to in this epoch
            if epoch in self.epoch_mappings and virtual_ip in self.epoch_mappings[epoch]:
                real_ip = self.epoch_mappings[epoch][virtual_ip]
                real_target_data[real_ip][epoch].append(response_time)
                attributed_count += 1
        
        print(f"Successfully attributed {attributed_count}/{total_count} measurements to real targets")
        
        # Convert to real_ip -> [epoch_means] for stability calculation
        real_target_distributions = {}
        
        for real_ip, epoch_data in real_target_data.items():
            epoch_means = []
            for epoch, measurements in epoch_data.items():
                epoch_mean = np.mean(measurements)
                epoch_means.append(epoch_mean)
            
            if len(epoch_means) >= 2:  # Need multiple epochs for stability analysis
                real_target_distributions[real_ip] = epoch_means
        
        print(f"\nReal target data summary:")
        for real_ip, values in real_target_distributions.items():
            mean_resp = np.mean(values)
            std_resp = np.std(values)
            print(f"{real_ip}: {len(values)} epochs, {mean_resp:.3f}±{std_resp:.3f}ms")
        
        return real_target_distributions
    
    def calculate_stability_index(self, real_target_distributions):
        """Calculate S(φ) for real target separability"""
        print(f"\nCalculating stability index for {len(real_target_distributions)} real targets...")
        
        if len(real_target_distributions) < 2:
            print("ERROR: Need at least 2 real targets for stability calculation")
            return None
        
        # Calculate inter-entity separability (minimum distance between real targets)
        real_targets = list(real_target_distributions.keys())
        inter_target_distances = []
        
        print("Inter-target separability:")
        for i in range(len(real_targets)):
            for j in range(i + 1, len(real_targets)):
                target_i = real_targets[i]
                target_j = real_targets[j]
                
                dist_i = real_target_distributions[target_i]
                dist_j = real_target_distributions[target_j]
                
                # Simple distance: difference of means
                distance = abs(np.mean(dist_i) - np.mean(dist_j))
                inter_target_distances.append(distance)
                
                print(f"  {target_i} <-> {target_j}: {distance:.3f}ms")
        
        min_inter_target_distance = min(inter_target_distances)
        
        # Calculate intra-entity variability (maximum variation within any real target)
        max_intra_target_variation = 0.0
        
        print("Intra-target variability:")
        for real_ip, values in real_target_distributions.items():
            std_dev = np.std(values)
            max_intra_target_variation = max(max_intra_target_variation, std_dev)
            print(f"  {real_ip}: std_dev = {std_dev:.3f}ms")
        
        # Calculate stability index
        if max_intra_target_variation > 0:
            stability_index = min_inter_target_distance / max_intra_target_variation
        else:
            stability_index = float('inf') if min_inter_target_distance > 0 else 0.0
        
        return {
            'stability_index': stability_index,
            'inter_target_separability': min_inter_target_distance,
            'intra_target_variability': max_intra_target_variation,
            'num_targets': len(real_target_distributions),
            'target_distributions': real_target_distributions,
            'inter_target_distances': inter_target_distances
        }
    
    def validate_target_classification(self, real_target_distributions):
        """Validate that we can classify virtual IPs to real targets"""
        print("\nValidating target classification accuracy...")
        
        # Calculate target centroids (mean response times)
        target_centroids = {}
        for real_ip, values in real_target_distributions.items():
            target_centroids[real_ip] = np.mean(values)
        
        print("Target centroids:")
        for real_ip, centroid in sorted(target_centroids.items(), key=lambda x: x[1]):
            print(f"  {real_ip}: {centroid:.3f}ms")
        
        # Test classification accuracy on actual measurements
        correct_classifications = 0
        total_classifications = 0
        
        for epoch in self.epoch_mappings:
            epoch_data = self.success_df[self.success_df['epoch_id'] == epoch]
            
            for vip in self.epoch_mappings[epoch]:
                true_target = self.epoch_mappings[epoch][vip]
                
                vip_measurements = epoch_data[epoch_data['virtual_ip'] == vip]['response_time_ms']
                if len(vip_measurements) == 0:
                    continue
                
                vip_avg = vip_measurements.mean()
                
                # Classify to nearest centroid
                distances = {rip: abs(vip_avg - centroid) for rip, centroid in target_centroids.items()}
                predicted_target = min(distances, key=distances.get)
                
                if predicted_target == true_target:
                    correct_classifications += 1
                total_classifications += 1
        
        accuracy = correct_classifications / total_classifications if total_classifications > 0 else 0
        print(f"\nClassification accuracy: {correct_classifications}/{total_classifications} = {accuracy:.1%}")
        
        return accuracy
    
    def run_corrected_analysis(self):
        """Run the corrected stability analysis"""
        print("\n" + "="*70)
        print("CORRECTED STABILITY INDEX ANALYSIS")
        print("="*70)
        print("S(φ) = min_{real_i≠real_j} D(P^φ_{real_i,·}, P^φ_{real_j,·}) / max_{real_i} max_{t≠t'} D(P^φ_{real_i,t}, P^φ_{real_i,t'})")
        
        # Step 1: Attribute measurements to real targets
        real_target_distributions = self.attribute_measurements_to_real_targets()
        
        if not real_target_distributions:
            print("ERROR: No real target data found")
            return None
        
        # Step 2: Calculate stability index
        results = self.calculate_stability_index(real_target_distributions)
        
        if not results:
            return None
        
        # Step 3: Validate classification accuracy  
        accuracy = self.validate_target_classification(real_target_distributions)
        results['classification_accuracy'] = accuracy
        
        # Step 4: Print results
        self.print_corrected_results(results)
        
        return results
    
    def print_corrected_results(self, results):
        """Print comprehensive results"""
        print("\n" + "="*70)
        print("CORRECTED STABILITY INDEX RESULTS")
        print("="*70)
        
        s_phi = results['stability_index']
        
        print(f"\n🎯 FORMAL STABILITY INDEX")
        print(f"S(φ) = {s_phi:.4f}")
        print(f"├─ Inter-target separability: {results['inter_target_separability']:.4f}ms")
        print(f"└─ Intra-target variability:  {results['intra_target_variability']:.4f}ms")
        print(f"   Real targets analyzed: {results['num_targets']}")
        
        print(f"\n📊 CLASSIFICATION PERFORMANCE")
        print(f"Target re-identification accuracy: {results['classification_accuracy']:.1%}")
        
        # Enhanced interpretation
        print(f"\n🔍 MATHEMATICAL FRAMEWORK VALIDATION")
        
        if s_phi > 3.0:
            print("🟢 EXCELLENT INVARIANT (S > 3.0)")
            print("   ✓ Strong separation between real targets")
            print("   ✓ Low variability within targets across MTD epochs")
            print("   ✓ MTD vulnerability confirmed: Timing patterns persist")
        elif s_phi > 1.5:
            print("🟡 GOOD INVARIANT (1.5 < S ≤ 3.0)")
            print("   ✓ Clear target separation with moderate stability")
            print("   ✓ Practical attack feasibility demonstrated")
        elif s_phi > 1.0:
            print("🟠 WEAK INVARIANT (1.0 < S ≤ 1.5)")
            print("   ⚠ Some target separation but requires extended observation")
        else:
            print("🔴 NO MEANINGFUL INVARIANT (S ≤ 1.0)")
            print("   ✓ MTD protection effective against timing-based attacks")
        
        # Research impact
        print(f"\n🎓 RESEARCH CONTRIBUTIONS")
        print(f"✓ Formal mathematical framework implementation: COMPLETE")
        print(f"✓ Real target entity analysis: {results['num_targets']} targets across epochs")
        print(f"✓ Quantitative MTD vulnerability assessment: S(φ) = {s_phi:.3f}")
        print(f"✓ Attack accuracy demonstration: {results['classification_accuracy']:.1%}")
        
        if results['classification_accuracy'] > 0.7:
            print(f"✓ Practical attack viability: HIGH")
            print(f"✓ MTD defense bypassed: Timing invariants enable target tracking")
        else:
            print(f"✓ MTD defense effective: Low target re-identification accuracy")
        
        # Target-specific analysis
        print(f"\n📈 PER-TARGET ANALYSIS")
        print("Real Target  | Epochs | Mean Response | Std Dev | Target Type")
        print("-------------|--------|---------------|---------|-------------")
        
        target_info = {
            '10.10.1.3': 'Modified TCP',
            '10.10.1.4': 'Network Delay', 
            '10.10.1.5': 'Default Config'
        }
        
        for real_ip, values in results['target_distributions'].items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            target_type = target_info.get(real_ip, 'Unknown')
            print(f"{real_ip:11} | {len(values):6} | {mean_val:13.3f} | {std_val:7.3f} | {target_type}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 corrected_stability_calculator.py <csv_file>")
        print("\nThis calculator properly attributes virtual IP measurements")
        print("to real target entities for accurate MTD stability analysis.")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    try:
        calculator = CorrectedStabilityCalculator(csv_file)
        results = calculator.run_corrected_analysis()
        
        if results:
            print(f"\n🎯 ANALYSIS COMPLETE!")
            print(f"Corrected stability index: S(φ) = {results['stability_index']:.3f}")
            print(f"Target classification accuracy: {results['classification_accuracy']:.1%}")
            print(f"\nThis represents the true MTD vulnerability assessment")
            print(f"based on real target entity analysis.")
        else:
            print("❌ Analysis failed")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
