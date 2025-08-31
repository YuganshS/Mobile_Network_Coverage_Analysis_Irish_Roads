"""
Spatial Analysis for Irish Highways.
Uses main analysis engine and displays the results.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import warnings

from main_analysis import main_analysis

warnings.filterwarnings('ignore')

class spatial_analyzer:
   
    
    # This method initializes the spatial analyzer.
    def __init__(self):
        
        self.main_analyzer = main_analysis()
        self.output_dir = Path("../results/spatial_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Spatial Analyzer Initialized")
        print("Using Main Analysis Engine - No Code Duplication")
        
        
    # This method runs the complete spatial analysis using the main analysis engine.
    def run_complete_analysis(self) -> Dict:
        
        print("\n" + "="*60)
        print("SPATIAL ANALYSIS")
        print("="*60)
        print("M7 Dublin-Limerick & N40 Cork Ring Road")
       
        
        
        results = self.main_analyzer.analyze_both_roads()
        
        self._display_results(results)
        
        self._save_results(results)
        
        return results
    

    # This method displays the results of the spatial analysis.
    def _display_results(self, results: Dict):
        
        print("\nSPATIAL ANALYSIS RESULTS")
        print("="*50)
        
        # 1. Network Coverage Analysis 
        print("\n1. Network Coverage Analysis")
        total_points = 0
        total_covered = 0
        
        for road_id in ['m7', 'n40']:
            if road_id in results:
                road_data = results[road_id]
                coverage_gaps = road_data.get('coverage_gaps', {})
                if coverage_gaps:
                    road_name = coverage_gaps.get('road_name', road_id.upper())
                    coverage_data = coverage_gaps.get('coverage_gaps', {})
                    road_total_points = coverage_gaps.get('total_points', 0)
                    total_points += road_total_points
                    
                    print(f"\n{road_name} Network Coverage:")
                    print(f"  Locations Checked: {road_total_points:,}")
                    
                            
                    unique_covered_locations = set()
                    
                    for network, network_data in coverage_data.items():
                        covered_points = network_data.get('covered_points', 0)
                        unique_covered_locations.add(covered_points)
                    
                    if unique_covered_locations:
                        road_covered_points = max(unique_covered_locations)
                    else:
                        road_covered_points = 0
                    
                    total_covered += road_covered_points
                    
                for network, network_data in coverage_data.items():
                    coverage_pct = network_data.get('coverage_percentage', 0)
                    covered_points = network_data.get('covered_points', 0)
                    no_coverage_points = network_data.get('no_coverage_points', 0)
                        
                    print(f"  {network}: {coverage_pct:.1f}% coverage ({covered_points:,} locations covered, {no_coverage_points:,} no signal areas)")
        

        print(f"\nOverall Network Coverage:")
        print(f"  Total Locations Analyzed: {total_points:,}")
        print(f"  Locations with Coverage: {total_covered:,}")
        print(f"  Overall Coverage Rate: {(total_covered/total_points*100):.1f}%" if total_points > 0 else "  Overall Coverage Rate: 0.0%")
        
        # 2. Network Provider Comparison
        print("\n2. Network Provider Comparison")
        for road_id in ['m7', 'n40']:
            if road_id in results:
                road_data = results[road_id]
                operator_comp = road_data.get('operator_comparison', {})
                if operator_comp:
                    road_name = operator_comp.get('road_name', road_id.upper())
                    performance_corr = operator_comp.get('performance_correlation', {})
                if performance_corr:
                    correlation_strength = performance_corr.get('correlation_strength', 'Unknown')
                    coverage_diff = performance_corr.get('coverage_difference', 0)
                    print(f"{road_name}: Provider performance: {correlation_strength} (Coverage difference: {coverage_diff:.1f}%)")
        
        # 3. Technology Coverage 
        print("\n3. Technology Coverage")
        for road_id in ['m7', 'n40']:
            if road_id in results:
                road_data = results[road_id]
                tech_distribution = road_data.get('technology_distribution', {})
                if tech_distribution:
                    road_name = tech_distribution.get('road_name', road_id.upper())
                    tech_transition = tech_distribution.get('technology_transition', {})
                    
                    print(f"{road_name}: Technology availability: \"4G-5G Overlap\" for both providers")
                    
                    
                for operator, transition_data in tech_transition.items():
                    transition_pct = transition_data.get('transition_percentage', 0)
                    overlap_pct = transition_data.get('overlap_percentage', 0)
                    print(f"{road_name}: 5G-only areas: {transition_pct:.1f}%, 4G-5G overlap: {overlap_pct:.1f}%")
                    
                
                    tech_evolution = tech_distribution.get('technology_evolution', {})
                if tech_evolution:
                    evolution_ratio = tech_evolution.get('evolution_ratio', 0)
                    print(f"{road_name}: 5G coverage level: {evolution_ratio:.2f} (5G Mature)")
        
        # 4. Clustering Analysis 
        print("\n4. Clustering Analysis")
        for road_id in ['m7', 'n40']:
            if road_id in results:
                road_data = results[road_id]
                clustering_analysis = road_data.get('clustering', {})
                if clustering_analysis:
                    road_name = clustering_analysis.get('road_name', road_id.upper())
                    total_points = clustering_analysis.get('total_points', 0)
                
                print(f"\n{road_name} Coverage Patterns:")
                print(f"  Locations Analyzed: {total_points:,}")
                
                # K-Means results
                kmeans = clustering_analysis.get('kmeans_clustering', {})
                if kmeans:
                    n_clusters = kmeans.get('n_clusters', 0)
                    silhouette = kmeans.get('silhouette_score', 0)
                    cluster_stats = kmeans.get('cluster_statistics', {})
                    
                    print(f"  K-Means Clustering: {n_clusters} coverage groups (Pattern quality: {silhouette:.3f})")
                    
                    # Show detailed cluster statistics
                    for cluster_id, stats in cluster_stats.items():
                        size = stats.get('size', 0)
                        avg_quality = stats.get('avg_coverage_quality', 0)
                        percentage = stats.get('percentage', 0)
                        quality_desc = "Very Good" if avg_quality >= 3.5 else "Good" if avg_quality >= 2.5 else "Fair" if avg_quality >= 1.5 else "Fringe"
                        print(f"    Coverage Group {cluster_id}: {size:,} locations ({percentage:.1f}%) - Avg Quality: {avg_quality:.2f} ({quality_desc})")
                
                # DBSCAN results
                dbscan = clustering_analysis.get('dbscan_clustering', {})
                if dbscan:
                    n_clusters = dbscan.get('n_clusters', 0)
                    noise_points = dbscan.get('noise_points', 0)
                    silhouette = dbscan.get('silhouette_score', 0)
                    cluster_stats = dbscan.get('cluster_statistics', {})
                    
                    print(f"  DBSCAN Clustering: {n_clusters} coverage groups, {noise_points:,} isolated locations (Pattern quality: {silhouette:.3f})")
                    
                    # Show detailed cluster statistics
                    for cluster_id, stats in cluster_stats.items():
                        size = stats.get('size', 0)
                        avg_quality = stats.get('avg_coverage_quality', 0)
                        percentage = stats.get('percentage', 0)
                        quality_desc = "Very Good" if avg_quality >= 3.5 else "Good" if avg_quality >= 2.5 else "Fair" if avg_quality >= 1.5 else "Fringe"
                        print(f"    Coverage Group {cluster_id}: {size:,} locations ({percentage:.1f}%) - Avg Quality: {avg_quality:.2f} ({quality_desc})")
        
        # 5. Backup Coverage
        print("\n5. Backup Coverage")
        for road_id in ['m7', 'n40']:
            if road_id in results:
                road_data = results[road_id]
                backup_coverage = road_data.get('backup_coverage', {})
                if backup_coverage:
                    road_name = backup_coverage.get('road_name', road_id.upper())
                    total_points = backup_coverage.get('total_points', 0)
                
                print(f"\n{road_name} Backup Coverage Analysis:")
                print(f"  Locations Analyzed: {total_points:,}")
                
                # Operator backup coverage
                op_backup = backup_coverage.get('operator_backup', {})
                for operator, op_data in op_backup.items():
                    coverage_pct = op_data.get('coverage_percentage', 0)
                    covered_points = op_data.get('total_covered_points', 0)
                    print(f"  {operator.title()} Backup: {coverage_pct:.1f}% coverage ({covered_points:,} locations)")
                
                # Technology backup coverage
                tech_backup = backup_coverage.get('technology_backup', {})
                for tech, tech_data in tech_backup.items():
                    coverage_pct = tech_data.get('coverage_percentage', 0)
                    covered_points = tech_data.get('total_covered_points', 0)
                    print(f"  {tech.upper()} Backup: {coverage_pct:.1f}% coverage ({covered_points:,} locations)")
                
                # Coverage overlap analysis
                coverage_overlap = backup_coverage.get('coverage_overlap', {})
                if coverage_overlap:
                    print(f"  Coverage Overlap Analysis:")
                    for op1, overlaps in coverage_overlap.items():
                        for op2, overlap_data in overlaps.items():
                            if op1 != op2:
                                overlap_count = overlap_data.get('overlap_count', 0)
                                overlap_pct = overlap_data.get('overlap_percentage', 0)
                                print(f"    {op1.title()} ↔ {op2.title()}: {overlap_count:,} locations ({overlap_pct:.1f}%)")
                
                # Network Reliability analysis
                redundancy = backup_coverage.get('redundancy_analysis', {})
                if redundancy:
                    print(f"  Network Reliability Analysis:")
                    for redundancy_type, data in redundancy.items():
                        if isinstance(data, dict):
                            count = data.get('count', 0)
                            percentage = data.get('percentage', 0)
                            type_desc = redundancy_type.replace('_', ' ').title()
                            print(f"    {type_desc}: {count:,} locations ({percentage:.1f}%)")
        
        # 6. Prediction Accuracy
        print("\n6. Model Accuracy")
        for road_id in ['m7', 'n40']:
            if road_id in results:
                road_data = results[road_id]
                model_comp = road_data.get('model_comparison', {})
                if model_comp and model_comp.get('model_comparison'):
                    road_name = model_comp.get('road_name', road_id.upper())
                
                # Show model vs real data comparison
                for network, network_data in model_comp.get('model_comparison', {}).items():
                    real = network_data.get('real_coverage', 0)
                    model = network_data.get('model_coverage', 0)
                    accuracy = network_data.get('accuracy', 0)
                    print(f"{road_name} {network}: Real {real:.1f}% vs Model {model:.1f}% (Accuracy: {accuracy:.1f}%)")
                
                # Show model agreement (difference between models)
                accuracy_metrics = model_comp.get('accuracy_metrics', {})
                if accuracy_metrics:
                    hata_accuracy = accuracy_metrics.get('hata_model', {}).get('overall_accuracy', 0)
                    gpp_accuracy = accuracy_metrics.get('3gpp_model', {}).get('overall_accuracy', 0)
                    if hata_accuracy > 0 and gpp_accuracy > 0:
                        model_agreement = 100 - abs(hata_accuracy - gpp_accuracy)
                        print(f"{road_name} Model Agreement: {model_agreement:.1f}% (Hata: {hata_accuracy:.1f}%, 3GPP: {gpp_accuracy:.1f}%)")
        
        print("\n" + "="*50)
        print("SPATIAL ANALYSIS COMPLETE")
        print("Results saved to output files")
        print("="*50)
    
    def _save_results(self, results: Dict):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        results_file = self.output_dir / f"spatial_analysis_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {str(k): convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            converted_results = convert_numpy(results)
            json.dump(converted_results, f, indent=2, default=str)
        
        print(f" Results saved: {results_file}")
        
        # Save summary report
        summary_file = self.output_dir / f"spatial_analysis_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("SPATIAL ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Method: Main Analysis Engine\n")
            f.write(f"Roads Analyzed: M7, N40\n")
            f.write("="*50 + "\n\n")
            
            # Add road summaries
            for road_id in ['m7', 'n40']:
                if road_id in results:
                    road_data = results[road_id]
                    coverage_gaps = road_data.get('coverage_gaps', {})
                    if coverage_gaps:
                        f.write(f"{coverage_gaps.get('road_name', road_id.upper())}:\n")
                        f.write(f"  Total Points: {coverage_gaps.get('total_points', 0)}\n")
                        f.write(f"  Networks Analyzed: {len(coverage_gaps.get('coverage_gaps', {}))}\n\n")
        
        print(f"Summary saved: {summary_file}")


#This method is the main function that runs the spatial analysis.
def main():
   
    analyzer = spatial_analyzer()
    results = analyzer.run_complete_analysis()
    
    print("\nSpatial Analysis Complete!")
    

if __name__ == "__main__":
    main()
