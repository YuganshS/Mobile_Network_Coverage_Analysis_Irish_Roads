"""
M7 Coverage Download Script - This script downloads the coverage data for the M7 motorway.
Every 50 meters along the M7 motorway, the script will download the coverage data for the 
operators and technologies.

The script will save the coverage data to a JSON file.

The script will also save a checkpoint every 100 points.


"""

import os
import json
import time
import requests
import random
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Configuration
M7_GEOJSON_FILE_PATH = '../data/ireland_highways.geojson'
RESULTS_FILE = Path("../data/m7_coverage_points")
api_url = "https://coveragemap.comreg.ie"
sample_distance = 50  # Distance between each collection point in meters
operators = ["vodafone", "three"]
technologies = ["4g", "5g"]


retry_config = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)


class Download_M7_coverage:
    
    def __init__(self):
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_config)
        self.session.mount("https://", adapter)
        
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': '*/*',
            'Referer': f"{api_url}/map"
        })
        
        self.check_connection()

    #This method checks if the connection to the ComReg API is successful.
    def check_connection(self):
        try:
            print("Testing ComReg connection...")
            response = self.session.get(f"{api_url}/map", timeout=10)
            response.raise_for_status()
            print("ComReg connection successful")
        except Exception as e:
            print(f"ComReg connection failed: {e}")
            raise
    
    #This method gets the coverage level for a given point.
    def get_coverage_level(self, lat: float, lon: float, operator: str, tech: str) -> str:
        params = {
            'lat': lat,
            'lon': lon,
            'operator': operator,
            'technology': tech,
        }
        
        level_url = f"{api_url}/comreg-map/api/legend/level"
        
        try:
            time.sleep(random.uniform(0.2, 0.7))
            
            response = self.session.get(level_url, params=params, timeout=15)
            response.raise_for_status()
            
            if not response.text.strip():
                return "No Coverage"
            
            data = response.json()
            
            if not isinstance(data, dict):
                return "No Coverage"
            
            result = data.get('label', 'No Coverage')
            return result
            
        except requests.exceptions.Timeout:
            return "No Coverage"
        except requests.exceptions.RequestException:
            return "No Coverage"
        except json.JSONDecodeError:
            return "No Coverage"
        except Exception:
            return "No Coverage"
    
    #This method converts the M7 geometry to a LineString.
    def convert_to_linestring(self) -> Optional[LineString]:
        try:
            print(f"Loading M7 geometry from: {M7_GEOJSON_FILE_PATH}")
            road_data = gpd.read_file(M7_GEOJSON_FILE_PATH)
            
            if road_data.empty:
                print("M7 GeoJSON file is empty")
                return None
            
            print(f"Loaded {len(road_data)} features")
            
            road_feature = None
            for idx, row in road_data.iterrows():
                if 'M7' in str(row.get('name', '')) or 'm7' in str(row.get('name', '')).lower():
                    road_feature = row
                    break
            
            if road_feature is None:
                print("No M7 feature found, using first feature")
                road_feature = road_data.iloc[0]
            
            geom = road_feature.geometry
            
            if isinstance(geom, MultiLineString):
                return linemerge(geom)
            return geom
            
        except Exception as e:
            print(f"Error loading M7 geometry: {e}")
            return None
    
    #This method creates sample points along the M7 geometry.
    def create_sample_points(self, line: LineString) -> List[Dict]:
        sample_points = []
        
        projected_line = gpd.GeoSeries([line], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        
        distance_covered = 0
        total_length = projected_line.length
        
        print(f"M7 total length: {total_length:.0f} meters")
        print(f"Generating points every {sample_distance} meters")
        
        while distance_covered < total_length:
            projected_point = projected_line.interpolate(distance_covered)
            
            coordinate_point = gpd.GeoSeries([projected_point], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
            
            sample_points.append({
                'lat': coordinate_point.y, 
                'lon': coordinate_point.x,
                'distance_meters': distance_covered
            })
            
            distance_covered += sample_distance
        
        end_coordinate_point = gpd.GeoSeries([projected_line.boundary.geoms[1]], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
        sample_points.append({
            'lat': end_coordinate_point.y, 
            'lon': end_coordinate_point.x,
            'distance_meters': total_length
        })
        
        print(f"Generated {len(sample_points)} sampling points")
        return sample_points
    
    #This method runs the script.
    def run(self):
        print("Starting M7 Coverage Download")
        print("=" * 50)
        
        road_line = self.convert_to_linestring()
        if not road_line:
            print("Failed to load M7 geometry")
            return
        
        sample_points = self.create_sample_points(road_line)
        
        RESULTS_FILE.mkdir(parents=True, exist_ok=True)
        output_file = RESULTS_FILE / "m7_coverage_points.json"
        
        results = []
        start_index = 0
        
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    existing_results = json.load(f)
                    results = existing_results
                    start_index = len(existing_results)
                    print(f"Found existing data: {start_index} points")
                    print(f"Resuming from point {start_index + 1}")
            except Exception as e:
                print(f"Error reading existing data: {e}")
                print("Starting fresh")
                results = []
                start_index = 0
        
        print(f"Starting  API calls for {len(sample_points) - start_index} points...")
        
        
        for i in tqdm(range(start_index, len(sample_points)), desc="Downloading M7 Coverage"):
            point = sample_points[i]
            
            point_data = {
                'point_id': i,
                'lat': point['lat'],
                'lon': point['lon'],
                'distance_meters': point['distance_meters'],
                'coverage': {}
            }
            
            for operator in operators:
                for tech in technologies:
                    level = self.get_coverage_level(point['lat'], point['lon'], operator, tech)
                    point_data['coverage'][f'{operator}_{tech}'] = level
            
            if i >= len(results):
                results.append(point_data)
            else:
                results[i] = point_data
            
            if (i + 1) % 100 == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Checkpoint saved: {i + 1}/{len(sample_points)} points")
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Download complete! Saved {len(results)} points to {output_file}")

if __name__ == "__main__":
    try:
        downloader = Download_M7_coverage()
        downloader.run()
    except Exception as e:
        print(f"Error: {e}")
