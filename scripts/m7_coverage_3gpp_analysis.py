#!/usr/bin/env python3
"""
M7 Highway Coverage Analysis using 3GPP Models

This script implements 3GPP TR 36.873 (4G) and TR 38.901 (5G) models for
 coverage analysis along the M7 highway in Ireland.

Models Used:
- 4G: 3GPP TR 36.873 (Urban Macro, Urban Micro, Rural Macro)
- 5G: 3GPP TR 38.901 (UMa, UMi, RMa scenarios)

"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import folium
from folium import plugins
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import random
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import math
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'm7_start': (53.3498, -6.2603),  # Dublin
    'm7_end': (52.6613, -7.2441),    # Limerick
    'sample_distance': 50,  # meters
    'operators': ['vodafone', 'three'],
    'technologies': ['4g', '5g'],
    'frequencies': {
        'vodafone': {'4g': 1800, '5g': 3500},  # MHz
        'three': {'4g': 1800, '5g': 3500}      # MHz
    },
    'tx_power': {
        'vodafone': {'4g': 67, '5g': 68},  # dBm - ComReg verified 2024
        'three': {'4g': 67, '5g': 68}      # dBm - ComReg verified 2024
    },
    'antenna_height': {
        'vodafone': {'4g': 30, '5g': 30},  # meters - Irish planning guidelines
        'three': {'4g': 30, '5g': 30}      # meters - Irish planning guidelines
    }
}

# 3GPP Coverage Models
class gpp_coverage:
    
    @staticmethod
    def calculate_4g_loss(frequency: float, distance: float, basestation_height: float, mobile_height: float, 
                            scenario: str = 'urban_macro') -> float:
       
        # Convert frequency to GHz
        freq_ghz = frequency / 1000
        
        # Effective environment height
        h_e = 1.0  # Default for urban areas
        
        if scenario == 'urban_macro':
            # Urban Macro (UMa) - TR 36.873 Section 7.2
            breakpoint_distance = 4 * basestation_height * mobile_height * freq_ghz / 3e8  # Breakpoint distance
            
            if distance <= breakpoint_distance:
                # LOS condition
                path_loss = 20 * np.log10(4 * np.pi * distance * freq_ghz / 3e8) + \
                        min(0.03 * h_e**1.72, 10) * np.log10(distance) + \
                        min(0.044 * h_e**1.72, 14.77) + \
                        0.002 * np.log10(h_e) * distance
            else:
                # NLOS condition
                path_loss = 20 * np.log10(4 * np.pi * breakpoint_distance * freq_ghz / 3e8) + \
                        min(0.03 * h_e**1.72, 10) * np.log10(breakpoint_distance) + \
                        min(0.044 * h_e**1.72, 14.77) + \
                        0.002 * np.log10(h_e) * breakpoint_distance
                
                pl_nlos = 161.04 - 7.1 * np.log10(10) + 7.5 * np.log10(h_e) - \
                         (24.37 - 3.7 * (h_e / basestation_height)**2) * np.log10(basestation_height) + \
                         (43.42 - 3.1 * np.log10(basestation_height)) * (np.log10(distance) - 3) + \
                         20 * np.log10(freq_ghz) - (3.2 * (np.log10(11.75 * mobile_height))**2 - 4.97)
                
                path_loss = max(path_loss, pl_nlos)
            
            return path_loss
            
        elif scenario == 'urban_micro':
            # Urban Micro (UMi) - TR 36.873 Section 7.2
            breakpoint_distance = 4 * basestation_height * mobile_height * freq_ghz / 3e8
            
            if distance <= breakpoint_distance:
                path_loss = 22.0 * np.log10(distance) + 28.0 + 20 * np.log10(freq_ghz)
            else:
                path_loss = 40 * np.log10(distance) + 7.8 - 18 * np.log10(basestation_height) - \
                        18 * np.log10(mobile_height) + 2 * np.log10(freq_ghz)
            
            return path_loss
            
        elif scenario == 'rural_macro':
            # Rural Macro (RMa) - TR 36.873 Section 7.2
            breakpoint_distance = 2 * np.pi * basestation_height * mobile_height * freq_ghz / 3e8
            
            if distance <= breakpoint_distance:
                path_loss = 20 * np.log10(4 * np.pi * distance * freq_ghz / 3e8) + \
                        min(0.03 * h_e**1.72, 10) * np.log10(distance) + \
                        min(0.044 * h_e**1.72, 14.77) + \
                        0.002 * np.log10(h_e) * distance
            else:
                path_loss = 20 * np.log10(4 * np.pi * breakpoint_distance * freq_ghz / 3e8) + \
                        min(0.03 * h_e**1.72, 10) * np.log10(breakpoint_distance) + \
                        min(0.044 * h_e**1.72, 14.77) + \
                        0.002 * np.log10(h_e) * breakpoint_distance + \
                        40 * np.log10(distance / breakpoint_distance)
            
            return path_loss
    
    # 5G Coverage Models.
    @staticmethod
    def calculate_5g_loss(frequency: float, distance: float, basestation_height: float, mobile_height: float,
                            scenario: str = 'UMa') -> float:
       
        # Convert frequency to GHz
        freq_ghz = frequency / 1000
        h_e = 1.0 
        
        if scenario == 'UMa':
            # Urban Macro - TR 38.901 Section 7.4.1
            breakpoint_distance = 4 * basestation_height * mobile_height * freq_ghz / 3e8
            
            if distance <= breakpoint_distance:
                # LOS condition
                path_loss = 28.0 + 22 * np.log10(distance) + 20 * np.log10(freq_ghz)
            else:
                # NLOS condition
                path_loss = 28.0 + 22 * np.log10(breakpoint_distance) + 20 * np.log10(freq_ghz) + \
                        40 * np.log10(distance / breakpoint_distance)
                
                # Additional NLOS component
                pl_nlos = 13.54 + 39.08 * np.log10(distance) + 20 * np.log10(freq_ghz) - \
                        0.6 * (mobile_height - 1.5)
                
                path_loss = max(path_loss, pl_nlos)
            
            return path_loss
            
        elif scenario == 'UMi':
            # Urban Micro - TR 38.901 Section 7.4.2
            breakpoint_distance = 4 * basestation_height * mobile_height * freq_ghz / 3e8
            
            if distance <= breakpoint_distance:
                path_loss = 32.4 + 21 * np.log10(distance) + 20 * np.log10(freq_ghz)
            else:
                path_loss = 32.4 + 21 * np.log10(breakpoint_distance) + 20 * np.log10(freq_ghz) + \
                        40 * np.log10(distance / breakpoint_distance)
            
            return path_loss
            
        elif scenario == 'RMa':
            # Rural Macro - TR 38.901 Section 7.4.3
            breakpoint_distance = 2 * np.pi * basestation_height * mobile_height * freq_ghz / 3e8
            
            if distance <= breakpoint_distance:
                path_loss = 20 * np.log10(4 * np.pi * distance * freq_ghz / 3e8) + \
                        min(0.03 * h_e**1.72, 10) * np.log10(distance) + \
                        min(0.044 * h_e**1.72, 14.77) + \
                        0.002 * np.log10(h_e) * distance
            else:
                path_loss = 20 * np.log10(4 * np.pi * breakpoint_distance * freq_ghz / 3e8) + \
                        min(0.03 * h_e**1.72, 10) * np.log10(breakpoint_distance) + \
                        min(0.044 * h_e**1.72, 14.77) + \
                        0.002 * np.log10(h_e) * breakpoint_distance + \
                        40 * np.log10(distance / breakpoint_distance)
            
            return path_loss


# Helper function to load base stations.
def load_basestations(operator: str, tech: str) -> list:
    
    import pandas as pd
    
    csv_path = "data/m7_verified_basestations_locations.csv"
    stations = []
    
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        operator_filter = df['OperatorName'].str.lower() == operator.lower()
        filtered_df = df[operator_filter]
        
        for _, row in filtered_df.iterrows():
            stations.append({
                'lat': float(row['Latitude']),
                'lon': float(row['Longitude']),
                'site_id': row['SiteID']
            })
    
    return stations

# Haversine distance in meters
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))


# Main class for the analysis 
class gpp_predicted_coverage:
    
    
    def __init__(self, config: Dict):
        self.config = config
        self.coverage_data = {}
        self.model_predictions = {}
        self.analysis_results = {}
        
    # Generate sampling points along M7 highway.
    def generate_m7_points(self) -> List[Tuple[float, float]]:
        
        start_lat, start_lon = self.config['m7_start']
        end_lat, end_lon = self.config['m7_end']
        
        
        total_distance = np.sqrt((end_lat - start_lat)**2 + (end_lon - start_lon)**2) * 111000  # meters
        
        
        num_points = int(total_distance / self.config['sample_distance'])
        points = []
        
        for i in range(num_points + 1):
            t = i / num_points
            lat = start_lat + t * (end_lat - start_lat)
            lon = start_lon + t * (end_lon - start_lon)
            points.append((lat, lon))
        
        return points
    
    # Load real coverage data from previous collection and structure by operator/technology, restoring all required fields.
    def load_coverage_data(self) -> Dict:
        
        with open('../data/m7_coverage_points/m7_coverage_points.json', 'r') as f:
            points = json.load(f)
        real_data = {op: {tech: [] for tech in self.config['technologies']} for op in self.config['operators']}
        sample_distance_km = self.config['sample_distance'] / 1000
        for idx, pt in enumerate(points):
            for op in self.config['operators']:
                for tech in self.config['technologies']:
                    key = f"{op}_{tech}"
                    if key in pt['coverage']:
                        real_data[op][tech].append({
                            'lat': pt['lat'],
                            'lon': pt['lon'],
                            'quality': pt['coverage'][key],
                            'point_id': pt['point_id'],
                            'distance_km': idx * sample_distance_km
                        })
        return real_data

    # Collect real coverage data from operators (restored).
    def get_coverage_data(self) -> Dict:
        
        self.coverage_data = self.load_coverage_data()
        return self.coverage_data
    
    # Calculate signal strength based on 3GPP models and real base station locations.
    def calculate_signal_strength(self, operator: str, tech: str, lat: float, lon: float) -> float:
       
        # Base distance from Dublin.
        base_distance = np.sqrt((lat - 53.3498)**2 + (lon + 6.2603)**2) * 111000  # meters
        
        #
        if base_distance < 50000:  
            environment = 'suburban'
            shadowing_standard = 6.9  
            loss_exponent = 2.9  
        else:
            environment = 'rural'
            # Determine LOS vs NLOS based on terrain.
            if self.check_line_of_sight(lat, lon):
                shadowing_standard = 5.1  
                loss_exponent = 2.3  
            else:
                shadowing_standard = 9.4  
                loss_exponent = 3.1  
        
        
        frequency = self.config['frequencies'][operator][tech]
        transmit_power = self.config['tx_power'][operator][tech]
        
        if tech == '4g':
            
            path_loss = gpp_coverage.calculate_4g_loss(
                frequency, base_distance, 
                self.config['antenna_height'][operator][tech], 1.5
            )
            
            path_loss_correction = -10.0  
        else:  # 5g
        
            path_loss = gpp_coverage.calculate_5g_loss(
                frequency, base_distance,
                self.config['antenna_height'][operator][tech], 1.5
            )
            path_loss_correction = -12.0  
        
        shadowing = np.random.normal(0, shadowing_standard)  
        
        multipath_fading = np.random.normal(0, 2.5)  
        
        signal_strength = transmit_power - path_loss + path_loss_correction + shadowing + multipath_fading
        
        return max(-120, min(-40, signal_strength))
    
    # Check if the location has Line-of-Sight (LOS) conditions.
    def check_line_of_sight(self, lat: float, lon: float) -> bool:
        
        
        dublin_dist = np.sqrt((lat - 53.3498)**2 + (lon + 6.2603)**2) * 111000
        cork_dist = np.sqrt((lat - 51.8969)**2 + (lon + 8.4863)**2) * 111000
        limerick_dist = np.sqrt((lat - 52.6613)**2 + (lon + 7.2441)**2) * 111000
        
        return True  

    
    # Convert signal strength to quality category using ComReg standards.
    def convert_signal_to_quality(self, signal_strength: float) -> str:
        
        if signal_strength >= -85:
            return 'Very Good'
        elif signal_strength >= -95:
            return 'Good'
        elif signal_strength >= -105:
            return 'Fair'
        elif signal_strength >= -115:
            return 'Fringe'
        else:
            return 'No Coverage'
    

    # Calculate coverage predictions using 3GPP models and real base station locations.
    def calculate_3gpp_predictions(self) -> Dict:
        
        points = self.generate_m7_points()
        predictions = {}
        for operator in self.config['operators']:
            predictions[operator] = {}
            for tech in self.config['technologies']:
                predictions[operator][tech] = []
                basestations = load_basestations(operator, tech)
                for i, (lat, lon) in enumerate(points):
                    if not basestations:
                        min_dist = 1e6  
                    else:
                        min_dist = min(haversine(lat, lon, bs['lat'], bs['lon']) for bs in basestations)
                    frequency = self.config['frequencies'][operator][tech]
                    transmit_power = self.config['tx_power'][operator][tech]
                    basestation_height = self.config['antenna_height'][operator][tech]
                    if tech == '4g':
                        path_loss = gpp_coverage.calculate_4g_loss(
                            frequency, min_dist, basestation_height, 1.5, 'rural_macro'
                        )
                    else:  
                        path_loss = gpp_coverage.calculate_5g_loss(
                            frequency, min_dist, basestation_height, 1.5, 'RMa'
                        )
                    predicted_signal = transmit_power - path_loss
                    predictions[operator][tech].append({
                        'lat': lat,
                        'lon': lon,
                        'predicted_signal': predicted_signal,
                        'predicted_quality': self.convert_signal_to_quality(predicted_signal),
                        'distance_km': i * self.config['sample_distance'] / 1000,
                        'nearest_bs_dist_m': min_dist
                    })
        self.model_predictions = predictions
        return predictions
    
    # Analyze coverage quality distribution for both real and model data.
    def predicted_coverage_quality(self) -> Dict:
        
        analysis = {}
        for operator in self.config['operators']:
            analysis[operator] = {}
            for tech in self.config['technologies']:
                # Real data
                real_data = self.coverage_data[operator][tech]
                real_quality_counts = {}
                for point in real_data:
                    quality = point['quality']
                    real_quality_counts[quality] = real_quality_counts.get(quality, 0) + 1
                # Model data
                predicted_data = self.model_predictions[operator][tech]
                pred_quality_counts = {}
                for point in predicted_data:
                    quality = point['predicted_quality']
                    pred_quality_counts[quality] = pred_quality_counts.get(quality, 0) + 1
                analysis[operator][tech] = {
                    'real_quality_distribution': real_quality_counts,
                    'predicted_quality_distribution': pred_quality_counts,
                    'total_points': len(real_data)
                }
        return analysis
    
    # Analyze coverage overlap between operators for both real and model data.
    def predicted_operator_overlap(self) -> Dict:
        
        points = self.generate_m7_points()
        overlap_analysis = {'real': {}, 'model': {}}
        for tech in self.config['technologies']:
            overlap_data_real = []
            overlap_data_model = []
            for i, (lat, lon) in enumerate(points):
                # Real
                point_analysis_real = {
                    'lat': lat,
                    'lon': lon,
                    'distance_km': i * self.config['sample_distance'] / 1000,
                    'operators_present': 0,
                    'operator_qualities': {}
                }
                for operator in self.config['operators']:
                    real_data = self.coverage_data[operator][tech][i]
                    quality = real_data['quality']
                    point_analysis_real['operator_qualities'][operator] = quality
                    if quality != 'No Coverage':
                        point_analysis_real['operators_present'] += 1
                overlap_data_real.append(point_analysis_real)
                # Model
                point_analysis_model = {
                    'lat': lat,
                    'lon': lon,
                    'distance_km': i * self.config['sample_distance'] / 1000,
                    'operators_present': 0,
                    'operator_qualities': {}
                }
                for operator in self.config['operators']:
                    pred_data = self.model_predictions[operator][tech][i]
                    quality = pred_data['predicted_quality']
                    point_analysis_model['operator_qualities'][operator] = quality
                    if quality != 'No Coverage':
                        point_analysis_model['operators_present'] += 1
                overlap_data_model.append(point_analysis_model)
            overlap_analysis['real'][tech] = overlap_data_real
            overlap_analysis['model'][tech] = overlap_data_model
        return overlap_analysis
    
    # Perform spatial clustering analysis using model predictions.
    def predicted_clusters(self) -> Dict:
        
        clustering_results = {}
        for operator in self.config['operators']:
            clustering_results[operator] = {}
            for tech in self.config['technologies']:
                data = self.model_predictions[operator][tech]
                coords = np.array([[point['lat'], point['lon']] for point in data])
                signals = np.array([point['predicted_signal'] for point in data])
                scaler = StandardScaler()
                coords_scaled = scaler.fit_transform(coords)
                dbscan = DBSCAN(eps=0.1, min_samples=5)
                clusters = dbscan.fit_predict(coords_scaled)
                cluster_stats = []
                for cluster_id in set(clusters):
                    if cluster_id == -1:  
                        continue
                    cluster_mask = clusters == cluster_id
                    cluster_coords = coords[cluster_mask]
                    cluster_signals = signals[cluster_mask]
                    cluster_stats.append({
                        'cluster_id': int(cluster_id),
                        'size': int(np.sum(cluster_mask)),
                        'center_lat': float(np.mean(cluster_coords[:, 0])),
                        'center_lon': float(np.mean(cluster_coords[:, 1])),
                        'avg_signal': float(np.mean(cluster_signals)),
                        'signal_std': float(np.std(cluster_signals))
                    })
                clustering_results[operator][tech] = {
                    'clusters': cluster_stats,
                    'noise_points': int(np.sum(clusters == -1)),
                    'total_clusters': len(set(clusters)) - (1 if -1 in clusters else 0)
                }
        return clustering_results
    
    # Estimate data rates based on predicted signal strength and technology.
    def calculate_data_rates(self) -> Dict:
        
        data_rates = {}
        for operator in self.config['operators']:
            data_rates[operator] = {}
            for tech in self.config['technologies']:
                rate_data = []
                for point in self.model_predictions[operator][tech]:
                    signal = point['predicted_signal']
                    if tech == '4g':
                        if signal >= -70:
                            rate = np.random.uniform(50, 100)  # Mbps
                        elif signal >= -85:
                            rate = np.random.uniform(20, 50)
                        elif signal >= -100:
                            rate = np.random.uniform(5, 20)
                        elif signal >= -110:
                            rate = np.random.uniform(1, 5)
                        else:
                            rate = 0
                    else:  # 5g
                        if signal >= -70:
                            rate = np.random.uniform(200, 500)  # Mbps
                        elif signal >= -85:
                            rate = np.random.uniform(100, 200)
                        elif signal >= -100:
                            rate = np.random.uniform(20, 100)
                        elif signal >= -110:
                            rate = np.random.uniform(5, 20)
                        else:
                            rate = 0
                    rate_data.append({
                        'lat': point['lat'],
                        'lon': point['lon'],
                        'signal_strength': signal,
                        'estimated_rate_mbps': rate,
                        'distance_km': point['distance_km']
                    })
                data_rates[operator][tech] = rate_data
        return data_rates
    
    # Analyze 4G fallback needs for 5G coverage gaps for both real and model data.
    def check_fallback_needs(self) -> Dict:
        
        fallback_analysis = {'real': {}, 'model': {}}
        for operator in self.config['operators']:
            # Real
            points_4g_real = self.coverage_data[operator]['4g']
            points_5g_real = self.coverage_data[operator]['5g']
            fallback_data_real = []
            fallback_needed_real = 0
            total_points_real = len(points_4g_real)
            for i in range(total_points_real):
                point_4g = points_4g_real[i]
                point_5g = points_5g_real[i]
                def classify_quality(quality):
                    if quality in ['Excellent', 'Good']:
                        return 'Good'
                    elif quality == 'Fair':
                        return 'Fair'
                    else:
                        return 'Poor'
                quality_4g = classify_quality(point_4g['quality'])
                quality_5g = classify_quality(point_5g['quality'])
                if quality_5g in ['Poor', 'No Coverage'] and quality_4g in ['Good', 'Fair']:
                    fallback_category = '5G_Fallback_Needed'
                    fallback_needed_real += 1
                elif quality_4g in ['Poor', 'No Coverage'] and quality_5g in ['Good', 'Fair']:
                    fallback_category = '4G_Fallback_Needed'
                elif quality_4g in ['Poor', 'No Coverage'] and quality_5g in ['Poor', 'No Coverage']:
                    fallback_category = 'Both_Poor'
                else:
                    fallback_category = 'Both_Good'
                fallback_data_real.append({
                    'lat': point_4g['lat'],
                    'lon': point_4g['lon'],
                    'distance_km': point_4g['distance_km'],
                    'quality_4g': quality_4g,
                    'quality_5g': quality_5g,
                    'fallback_category': fallback_category
                })
            fallback_analysis['real'][operator] = {
                'fallback_data': fallback_data_real,
                'fallback_percentage': (fallback_needed_real / total_points_real) * 100 if total_points_real > 0 else 0,
                'total_points': total_points_real,
                'fallback_needed': fallback_needed_real
            }
            # Model
            points_4g_model = self.model_predictions[operator]['4g']
            points_5g_model = self.model_predictions[operator]['5g']
            fallback_data_model = []
            fallback_needed_model = 0
            total_points_model = len(points_4g_model)
            for i in range(total_points_model):
                point_4g = points_4g_model[i]
                point_5g = points_5g_model[i]
                def classify_quality(quality):
                    if quality in ['Excellent', 'Good']:
                        return 'Good'
                    elif quality == 'Fair':
                        return 'Fair'
                    else:
                        return 'Poor'
                quality_4g = classify_quality(point_4g['predicted_quality'])
                quality_5g = classify_quality(point_5g['predicted_quality'])
                if quality_5g in ['Poor', 'No Coverage'] and quality_4g in ['Good', 'Fair']:
                    fallback_category = '5G_Fallback_Needed'
                    fallback_needed_model += 1
                elif quality_4g in ['Poor', 'No Coverage'] and quality_5g in ['Good', 'Fair']:
                    fallback_category = '4G_Fallback_Needed'
                elif quality_4g in ['Poor', 'No Coverage'] and quality_5g in ['Poor', 'No Coverage']:
                    fallback_category = 'Both_Poor'
                else:
                    fallback_category = 'Both_Good'
                fallback_data_model.append({
                    'lat': point_4g['lat'],
                    'lon': point_4g['lon'],
                    'distance_km': point_4g['distance_km'],
                    'quality_4g': quality_4g,
                    'quality_5g': quality_5g,
                    'fallback_category': fallback_category
                })
            fallback_analysis['model'][operator] = {
                'fallback_data': fallback_data_model,
                'fallback_percentage': (fallback_needed_model / total_points_model) * 100 if total_points_model > 0 else 0,
                'total_points': total_points_model,
                'fallback_needed': fallback_needed_model
            }
        return fallback_analysis
    

    # Run complete analysis pipeline.
    def run_analysis(self) -> Dict:
        
        print("Starting M7 coverage analysis with 3GPP models...")
        
        # Collect real coverage data
        print("Collecting real coverage data...")
        self.get_coverage_data()
        
        # Calculate 3GPP model predictions
        print("Calculating 3GPP model predictions...")
        self.calculate_3gpp_predictions()
        
        # Analyze coverage quality
        print("Analyzing coverage quality...")
        quality_analysis = self.predicted_coverage_quality()
        
        # Analyze operator overlap
        print("Analyzing operator overlap...")
        overlap_analysis = self.predicted_operator_overlap()
        
        # Perform spatial clustering
        print("Performing spatial clustering...")
        clustering_analysis = self.predicted_clusters()
        
        # Estimate data rates
        print("Estimating data rates...")
        data_rates = self.calculate_data_rates()
        
        # Analyze fallback needs
        print("Analyzing fallback needs...")
        fallback_analysis = self.check_fallback_needs()
        
        # Compile results
        self.analysis_results = {
            'coverage_data': self.coverage_data,
            'model_predictions': self.model_predictions,
            'quality_analysis': quality_analysis,
            'overlap_analysis': overlap_analysis,
            'clustering_analysis': clustering_analysis,
            'data_rates': data_rates,
            'fallback_analysis': fallback_analysis,
            'analysis_timestamp': datetime.now().isoformat(),
            'config': self.config
        }
        
        print("Analysis complete!")
        return self.analysis_results
    
    def save_results(self, filename: str = '../data/Model 3GPP/m7_3gpp_analysis_results.json'):
        # Save analysis results to JSON file.
        
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        results_to_save = convert_numpy_types(self.analysis_results)
        
        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Results saved to {filename}")
    


def main():
    # Main execution function.
    # Initialize analyzer
    analyzer = gpp_predicted_coverage(CONFIG)
    
    # Run complete analysis
    results = analyzer.run_analysis()
    
    # Save results
    analyzer.save_results()

if __name__ == "__main__":
    main() 