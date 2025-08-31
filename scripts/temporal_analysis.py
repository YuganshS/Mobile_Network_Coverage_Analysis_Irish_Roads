
"""
Temporal Analysis for Irish Motorway Traffic
Uses main analysis engine to display the results.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import warnings

from main_analysis import main_analysis

warnings.filterwarnings('ignore')


class temporal_analyzer:
    
    
    def __init__(self):
        
        self.main_engine = main_analysis()
        self.output_dir = Path("../results/temporal_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Temporal Analyzer Initialized")
       

    # This method runs the temporal analysis using the main analysis engine.
    def run_temporal_analysis(self) -> Dict:
        
        print("\n" + "="*60)
        print("TEMPORAL ANALYSIS")
        print("="*60)
        print("M7 Dublin-Limerick & N40 Cork Ring Road")
        
       
        
        temporal_results = self.main_engine.run_temporal_analysis(traffic_data=None)
        
        self.main_engine.display_temporal_results(temporal_results)
        
        self.save_temporal_results(temporal_results)
        
        return temporal_results
    
    def save_temporal_results(self, results: Dict):
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_file = self.output_dir / f"temporal_analysis_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            
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
        
        
        summary_file = self.output_dir / f"temporal_analysis_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("TEMPORAL ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            
            
            if 'detailed_weekday_weekend' in results:
                f.write("Weekday/Weekend Analysis: Available\n")
            if 'morning_evening_peaks' in results:
                f.write("Peak Hour Analysis: Available\n")
            if 'multiple_peak_hours' in results:
                f.write("Multiple Peak Hours Analysis: Available\n")
        
        print(f"Summary saved: {summary_file}")

def main():
    
    analyzer = temporal_analyzer()
    results = analyzer.run_temporal_analysis()
    
    print("\nTemporal Analysis Complete!")
    print("Results ready")
    print("Analysis complete")

if __name__ == "__main__":
    main()
