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

