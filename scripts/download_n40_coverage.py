import os
import json
import time
import requests
import random
from pathlib import Path
from typing import List, Dict, Optional
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Configuration
CORK_GEOJSON_PATH = '../data/cork_ring_road.geojson'
OUTPUT_DIR = Path("../data/cork_ring_coverage_points")
BASE_URL = "https://coveragemap.comreg.ie"
sample_distance = 50

retry_config = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
)

operators = ["vodafone", "three"]
technologies = ["4g", "5g"]

class Download_N40_coverage:
    def __init__(self):
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_config)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': '*/*',
            'Referer': f"{BASE_URL}/map"
        })
        self.initialize_session()

    def initialize_session(self):
        try:
            print("Initializing session...")
            response = self.session.get(f"{BASE_URL}/map")
            response.raise_for_status()
            print("Session initialized successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error initializing session: {e}")
            raise

    def get_coverage_level(self, lat: float, lon: float, operator: str, tech: str) -> str:
        params = {
            'lat': lat,
            'lon': lon,
            'operator': operator,
            'technology': tech,
        }
        level_url = f"{BASE_URL}/comreg-map/api/legend/level"
        try:
            time.sleep(random.uniform(0.2, 0.7))
            response = self.session.get(level_url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get('label', 'No Coverage')
        except requests.exceptions.RequestException as e:
            return "No Coverage"
        except json.JSONDecodeError:
            return "No Coverage"

    def load_road_segments(self) -> List[LineString]:
        try:
            print(f"Loading Cork Ring Road geometry from {CORK_GEOJSON_PATH}...")
            road_data = gpd.read_file(CORK_GEOJSON_PATH)
            
            if road_data.empty:
                print("GeoJSON file is empty.")
                return []
            
            # Get all LineString geometries
            road_segments = []
            for idx, row in road_data.iterrows():
                geom = row.geometry
                if geom.geom_type == "LineString":
                    road_segments.append(geom)
                elif geom.geom_type == "MultiLineString":
                    for line in geom.geoms:
                        road_segments.append(line)
            
            print(f"Found {len(road_segments)} LineString segments.")
            return road_segments
                    
        except Exception as e:
            print(f"Could not load or parse {CORK_GEOJSON_PATH}: {e}")
            return []

    def create_sample_points(self, segments: List[LineString]) -> List[Dict]:
        sample_points = []
        point_id = 0
        
        for segment_number, line in enumerate(segments):
            print(f"Processing segment {segment_number + 1}/{len(segments)}")
            
            projected_line = gpd.GeoSeries([line], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
            
            if projected_line.length < 100:
                print(f"Skipping segment {segment_number + 1} (too short: {projected_line.length:.1f}m)")
                continue
            
            distance_covered = 0
            while distance_covered < projected_line.length:
                projected_point = projected_line.interpolate(distance_covered)
                coordinate_point = gpd.GeoSeries([projected_point], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
                sample_points.append({
                    'point_id': point_id,
                    'lat': coordinate_point.y, 
                    'lon': coordinate_point.x,
                    'segment_id': segment_number
                })
                point_id += 1
                distance_covered += sample_distance
            
            last_point_wgs84 = gpd.GeoSeries([projected_line.boundary.geoms[1]], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
            sample_points.append({
                'point_id': point_id,
                'lat': last_point_wgs84.y, 
                'lon': last_point_wgs84.x,
                'segment_id': segment_number
            })
            point_id += 1
        
        print(f"Generated {len(sample_points)} total sample points from {len(segments)} segments.")
        return sample_points

    def run(self):
        segments = self.load_road_segments()
        if not segments:
            print("Could not retrieve N40 segments.")
            return
        
        sample_points = self.create_sample_points(segments)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        results = []
        print("Starting download of coverage data for all N40 sample points.")
        
        for point in tqdm(sample_points, desc="Querying Cork Points"):
            point_data = {
                'point_id': point['point_id'], 
                'lat': point['lat'], 
                'lon': point['lon'], 
                'segment_id': point['segment_id'],
                'coverage': {}
            }
            
            for op in operators:
                for tech in technologies:
                    level = self.get_coverage_level(point['lat'], point['lon'], op, tech)
                    point_data['coverage'][f'{op}_{tech}'] = level
            
            results.append(point_data)
        
        output_file = OUTPUT_DIR / "cork_ring_coverage_points.json"
        print(f"Download complete. Saving all data to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Sucessfully collected coverage data for {len(results)} points.")

if __name__ == "__main__":
    downloader = Download_N40_coverage()
    downloader.run() 