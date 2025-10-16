# Mobile Network Coverage Analysis for Vehicular Traffic on Irish Roads

# Name - Yugansh Suryavanshi


## Project Overview
This research project analyzes mobile network coverage performance along Ireland's major transportation corridors, specifically the M7 Dublin-Limerick motorway and N40 Cork Ring Road. The study combines spatial analysis of real ComReg coverage data with temporal analysis of TII traffic patterns to understand network performance in transportation environments.


## Project Structure

Research_Project/
   scripts/                     # Python analysis scripts
        main_analysis.py         # Core analysis engine
        spatial_analysis.py      # Spatial coverage analysis
        temporal_analysis.py     # Temporal traffic analysis
        model_m7_coverage_hata.py # Hata model implementation
        m7_coverage_3gpp_analysis.py # 3GPP model analysis
        tii_data_processor.py    # TII traffic data processing
        download_m7_coverage.py  # M7 coverage data download
        download_n40_coverage.py # N40 coverage data download
   data/                        # Data files and datasets
        ComReg Coverage Data/    # Real coverage measurements
        RAW_TII/                 # Raw TII traffic data
        temporal_analysis/       # Processed temporal data
        Hata Cost 231/           # Hata model results
        Model 3GPP/              # 3GPP model results
        *.geojson                # Geographic data files
   results/                     # Analysis outputs
       spatial_analysis/        # Spatial analysis results
       temporal_analysis/       # Temporal analysis results
   
    venv/                        # Python virtual environment
    thesis/                      # LaTeX thesis files


## Setup Instructions

### Prerequisites
- Python 3.10.2 or higher


### Installation
1. **Clone or download the project**
2. **Install dependencies:**
   ```bash
   pip install pandas numpy geopandas scikit-learn matplotlib seaborn requests beautifulsoup4
   ```

## Usage

### Running the Complete Analysis
```bash
# Navigate to scripts directory
cd scripts
python main_analysis.py
python spatial_analysis.py
python temporal_analysis.py
```

### Individual Scripts
```bash

python download_m7_coverage.py
python download_n40_coverage.py
python tii_data_processor.py
python model_m7_coverage_hata.py
python m7_coverage_3gpp_analysis.py
```


## Data Sources
- **ComReg:** Real coverage measurements and base station data
- **TII:** Traffic volume and pattern data
- **OpenStreetMap:** Geographic road network data
- **3GPP:** Standardized propagation models

## Data Files Note
This repository excludes large data files to maintain reasonable repository size:
- **Excluded:** `data/RAW_TII/` folder containing 84 Excel files with raw traffic data
- **Excluded:** PDF files including `scripts/124101779.pdf` and thesis figures
- **Included:** Processed data files, analysis results, and smaller datasets

To obtain the complete dataset, the raw TII traffic data files would need to be downloaded separately from the original TII data sources.

## Technologies Used
- **Python Libraries:** pandas, numpy, geopandas, scikit-learn, matplotlib
- **Analysis Methods:** K-means clustering, DBSCAN, spatial analysis
- **Models:** Hata Cost-231, 3GPP TR 38.901
- **Data Formats:** JSON, CSV, GeoJSON

## Output Files
- **Spatial Analysis:** Coverage maps, clustering results, model comparisons
- **Temporal Analysis:** Traffic patterns, seasonal variations, capacity assessments
- **Thesis Visualizations:** Charts and figures for academic presentation

#

