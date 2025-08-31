import json
import math
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# File Paths
M7_POINTS_PATH = Path('../data/m7_coverage_points/m7_coverage_points.json')
BASESTATION_CSV = Path("../data/m7_verified_basestations_locations.csv")
MAX_DISTANCE = 20

# Hata Model & Network Parameters (ComReg Verified)
basestation_height = 30.0  # Effective Base Station antenna height (meters) - Irish planning guidelines
mobile_height = 1.5   # Mobile Station antenna height (meters) - Standard value
power_4g = 67.0  # ComReg verified: 67 dBm/5MHz for 1800 MHz LTE (2024)
frequency_4g= 1800  # Typical 4G band 

# RSRP thresholds for coverage quality (ComReg 2021 verified)
coverage_thresholds = {
    -85: 'Very Good',    # dBm - ComReg/Plum report 2021
    -95: 'Good',         # dBm - ComReg/Plum report 2021
    -105: 'Fair',        # dBm - ComReg/Plum report 2021
    -115: 'Fringe'       # dBm - ComReg/Plum report 2021
}

OPERATOR_INFO = {
    "Vodafone": {"dir": "Vodafone"},
    "Three": {"dir": "Three"}
}

# Load all basestations from CSV
def load_basestations():
    all_stations = {}
    
    for op_name in ["vodafone", "three"]:
        key = f"{op_name}_4g"
        all_stations[key] = []
    
    if BASESTATION_CSV.exists():
        df = pd.read_csv(BASESTATION_CSV)
        for _, row in df.iterrows():
            operator = row['OperatorName'].lower()
            lat = float(row['Latitude'])
            lon = float(row['Longitude'])
            
            
            key = f"{operator}_4g"
            if key in all_stations:
                all_stations[key].append({
                    'lat': lat,
                    'lon': lon,
                    'site_id': row['SiteID']
                })
    
    return all_stations

# Cost 231 Hata Model
def cost231_hata(freq_mhz, d_km, h_bs, h_ms):
    if d_km == 0: return 0.0
    
    a_hm = (1.1 * math.log10(freq_mhz) - 0.7) * h_ms - (1.56 * math.log10(freq_mhz) - 0.8)
    
    path_loss = (46.3 + 33.9 * math.log10(freq_mhz) - 13.82 * math.log10(h_bs) - a_hm +
                 (44.9 - 6.55 * math.log10(h_bs)) * math.log10(d_km) + 0) # C_m = 0 for suburban
    return path_loss

# Get RSRP from EIRP and path loss.
def get_rsrp(eirp_dbm, path_loss_db):
    
    return eirp_dbm - path_loss_db

# Classify RSRP into a quality label.
def classify_rsrp(rsrp):
    
    for threshold, label in sorted(coverage_thresholds.items(), reverse=True):
        if rsrp >= threshold:
            return label
    return 'No coverage'


# Calculate distance between two points in km.
def haversine_distance(lat1, lon1, lat2, lon2):
    
    R = 6371  # Radius of Earth in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon / 2) * math.sin(dLon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Main Execution
def main():
    print(" Starting the M7 Hata Model")
    
    print(" Loading M7 sample points...")
    with open(M7_POINTS_PATH, 'r') as f:
        m7_points = json.load(f)
    print(f"   ...loaded {len(m7_points)} points.")
    
    print(" Loading all basestation data...")
    basestations = load_basestations()
    for name, station_list in basestations.items():
        print(f"   ...loaded {len(station_list)} stations for {name}.")
    
    print("\n Calculating modeled coverage for each point on the M7...")
    results = []
    try:
        for point in tqdm(m7_points, desc="Modeling M7 Coverage"):
            point_data = {'lat': point['lat'], 'lon': point['lon'], 'modeled_coverage': {}}
            
            for tech_key, stations in basestations.items():
                best_rsrp = -999
                
                # Use 4G parameters 
                freq = frequency_4g
                eirp = power_4g
                
                for station in stations:
                    dist_km = haversine_distance(point['lat'], point['lon'], station['lat'], station['lon'])
                    if 0 < dist_km <= MAX_DISTANCE:
                        path_loss = cost231_hata(freq, dist_km, basestation_height, mobile_height)
                        rsrp = get_rsrp(eirp, path_loss)
                        if rsrp > best_rsrp:
                            best_rsrp = rsrp
                
                point_data['modeled_coverage'][tech_key] = classify_rsrp(best_rsrp)
            results.append(point_data)
    except Exception as e:
        print(f"\n An error occurred during calculation: {e}")
        return
    print("\n Saving results...")

    output_dir = Path("../data/Hata Cost 231")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / 'm7_hata_model_coverage.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Hata model coverage data saved to {output_json}")
    


if __name__ == "__main__":
    main() 