# Spotify Unwrapped Analytics

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-GradientBoosting-green)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-lightgrey)

Analyze your Spotify streaming history and predict your top tracks and artists using Python and LightGBM.

---

## Project Overview

This project processes a personal Spotify StreamingHistory dataset to:

1. Clean and engineer features from raw streaming data.
2. Explore your top tracks and artists through visualizations.
3. Build predictive models using **LightGBM** to forecast future top tracks and artists.
4. Export CSVs of actual and predicted top tracks/artists for easy sharing.

This replicates the concept of Spotify's annual “Wrapped” feature, but built as a data analytics personal project.

---

## Features

- **Data Cleaning:** Remove short plays and normalize track/artist names.
- **Exploratory Data Analysis:** Top tracks and artists by plays and listening minutes.
- **Visualization:** Bar charts for top 10 tracks and artists (actual & predicted).
- **Predictive Modeling:** LightGBM regression to predict top tracks/artists.
- **CSV Exports:**  
  - `top_tracks_actual.csv`  
  - `top_artists_actual.csv`  
  - `top_tracks_predicted.csv`  
  - `top_artists_predicted.csv`

---

## Folder Structure

Spotify-Unwrapped/
├── data/ # Place your Spotify StreamingHistory JSON files here
├── src/
│ └── analysis.py # Main script for cleaning, analysis, modeling, and visualization
├── results/ # Generated CSV outputs
├── images/ # Optional: save charts/screenshots
├── README.md
└── .gitignore

yaml
Copy code

---

## Data

This project uses your **Spotify Streaming History**, which is **not included in this repository** to protect privacy.  

To run the analysis:

1. Go to [Spotify Account → Download Your Data](https://www.spotify.com/us/account/privacy/).  
2. Request your **“Streaming History”** (JSON format).  
3. Place the downloaded JSON files into the `data/` folder in the project root:  

Spotify-Unwrapped/
├── data/
│ ├── StreamingHistory0.json
│ ├── StreamingHistory1.json
│ └── ...

yaml
Copy code

> Make sure your JSON files follow the naming pattern `StreamingHistory*.json` so the analysis script can read them automatically.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/spotify-unwrapped-analytics.git
cd spotify-unwrapped-analytics
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
Install dependencies:

bash
Copy code
pip install pandas matplotlib numpy lightgbm
Usage
Place your Spotify StreamingHistory JSON files in the data/ folder.

Run the analysis script:

bash
Copy code
python src/analysis.py
View results in results/ folder:

top_tracks_actual.csv – Your actual top tracks

top_artists_actual.csv – Your actual top artists

top_tracks_predicted.csv – Predicted top tracks

top_artists_predicted.csv – Predicted top artists

Visualizations will automatically appear as bar charts during runtime.

Note: You must download your own Spotify data and place it in the data/ folder before running analysis.py.

Example Visualizations
(Optional: add screenshots of your charts here)

Top 10 Actual Tracks

Top 10 Predicted Tracks

Top 10 Actual Artists

Top 10 Predicted Artists

Precision Evaluation
The model calculates Precision@10 to compare predicted top tracks/artists with actual top tracks/artists.

python
Copy code
Precision@10 for Tracks: 0.8
Precision@10 for Artists: 0.9
License
This project is for educational and portfolio purposes. No license restrictions.

Acknowledgements
Inspired by Spotify Wrapped.

Built with Python, Pandas, Matplotlib, NumPy, and LightGBM.
