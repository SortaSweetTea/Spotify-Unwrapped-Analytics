import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
import os

# -----------------------------
# Step 1: Load Spotify history
# -----------------------------
files = glob.glob("../data/StreamingHistory*.json")
dfs = [pd.read_json(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# -----------------------------
# Step 2: Parse time & Step 3: Convert ms to minutes
# -----------------------------
df['endTime'] = pd.to_datetime(df['endTime'])
df['minutes'] = df['msPlayed'] / 60000

# -----------------------------
# Step 4: Sort by time
# -----------------------------
df = df.sort_values('endTime').reset_index(drop=True)
print("First 5 rows of your streaming history:")
print(df.head())

# -----------------------------
# Step 5: Clean & engineer features
# -----------------------------
df = df[df['msPlayed'] >= 30_000]  # Drop plays shorter than 30 seconds
df['artist_clean'] = df['artistName'].str.lower().str.strip()
df['track_clean'] = (df['trackName']
                     .str.replace(r"\(.*\)|\[.*\]", "", regex=True)
                     .str.replace(r"feat\..*", "", regex=True)
                     .str.lower().str.strip())
last_date = df['endTime'].max()
df['days_since_last_play'] = (last_date - df['endTime']).dt.days

# -----------------------------
# Step 6: Explore & aggregate
# -----------------------------
# Actual top 10 tracks using original track names
top_tracks_actual = (
    df.groupby('trackName')
      .agg(total_minutes=('minutes','sum'),
           plays=('trackName','size'))
      .sort_values('plays', ascending=False)
      .head(10)
      .reset_index()
)
print("\nTop 10 Tracks by Plays (Actual Names):")
print(top_tracks_actual)

# Actual top 10 artists using original artist names
top_artists_actual = (
    df.groupby('artistName')
      .agg(total_minutes=('minutes','sum'),
           plays=('artistName','size'))
      .sort_values('plays', ascending=False)
      .head(10)
      .reset_index()
)
print("\nTop 10 Artists by Plays (Actual Names):")
print(top_artists_actual)

# For modeling: use cleaned columns
top_artists = (df.groupby('artist_clean')
                .agg(total_minutes=('minutes','sum'),
                     plays=('artist_clean','size'))
                .sort_values('plays', ascending=False)
                .head(10))

top_tracks = (df.groupby('track_clean')
                .agg(total_minutes=('minutes','sum'),
                     plays=('track_clean','size'))
                .sort_values('plays', ascending=False)
                .head(10))

# -----------------------------
# Step 7: Visualizations
# -----------------------------
top_artists['plays'].plot(kind='bar', color='skyblue')
plt.title("Top 10 Artists by Plays (Cleaned for Modeling)")
plt.ylabel("Number of Plays")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

top_tracks['plays'].plot(kind='bar', color='lightgreen')
plt.title("Top 10 Tracks by Plays (Cleaned for Modeling)")
plt.ylabel("Number of Plays")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# -----------------------------
# Step 8: LightGBM prediction for tracks
# -----------------------------
cutoff = last_date - pd.Timedelta(days=30)
train = df[df['endTime'] < cutoff]
holdout = df[df['endTime'] >= cutoff]

train_agg_tracks = train.groupby('track_clean').agg(
    total_minutes=('minutes','sum'),
    plays=('track_clean','size')
).reset_index()

holdout_agg_tracks = holdout.groupby('track_clean').agg(
    target_plays=('track_clean','size')
).reset_index()

data_tracks = train_agg_tracks.merge(holdout_agg_tracks, on='track_clean', how='left').fillna(0)

X_tracks = data_tracks[['total_minutes','plays']]
y_tracks = np.log1p(data_tracks['target_plays'])

dtrain_tracks = lgb.Dataset(X_tracks, label=y_tracks)
params = {'objective': 'regression', 'metric': 'rmse'}
model_tracks = lgb.train(params, dtrain_tracks, num_boost_round=100)

data_tracks['predicted_plays'] = np.expm1(model_tracks.predict(X_tracks))

print("\nTop 10 Predicted Tracks:")
print(data_tracks.sort_values('predicted_plays', ascending=False).head(10))

# -----------------------------
# Step 9: LightGBM prediction for artists
# -----------------------------
train_agg_artists = train.groupby('artist_clean').agg(
    total_minutes=('minutes','sum'),
    plays=('artist_clean','size')
).reset_index()

holdout_agg_artists = holdout.groupby('artist_clean').agg(
    target_plays=('artist_clean','size')
).reset_index()

data_artists = train_agg_artists.merge(holdout_agg_artists, on='artist_clean', how='left').fillna(0)

X_artists = data_artists[['total_minutes','plays']]
y_artists = np.log1p(data_artists['target_plays'])

dtrain_artists = lgb.Dataset(X_artists, label=y_artists)
model_artists = lgb.train(params, dtrain_artists, num_boost_round=100)

data_artists['predicted_plays'] = np.expm1(model_artists.predict(X_artists))

print("\nTop 10 Predicted Artists:")
print(data_artists.sort_values('predicted_plays', ascending=False).head(10))

# -----------------------------
# Step 10: Evaluate Precision@10 for tracks & artists
# -----------------------------
def precision_at_k(df, k=10):
    true_top = set(df.sort_values('target_plays', ascending=False).head(k).index)
    pred_top = set(df.sort_values('predicted_plays', ascending=False).head(k).index)
    return len(true_top & pred_top) / k

print("\nPrecision@10 for Tracks:", precision_at_k(data_tracks))
print("Precision@10 for Artists:", precision_at_k(data_artists))

# -----------------------------
# Step 11: Visualize predicted top tracks and artists
# -----------------------------
data_tracks_sorted = data_tracks.sort_values('predicted_plays', ascending=False).head(10)
data_tracks_sorted.set_index('track_clean')['predicted_plays'].plot(
    kind='bar', color='orange'
)
plt.title("Top 10 Predicted Tracks")
plt.ylabel("Predicted Plays")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

data_artists_sorted = data_artists.sort_values('predicted_plays', ascending=False).head(10)
data_artists_sorted.set_index('artist_clean')['predicted_plays'].plot(
    kind='bar', color='purple'
)
plt.title("Top 10 Predicted Artists")
plt.ylabel("Predicted Plays")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# -----------------------------
# Step 12: Export results to CSVs with readable names
# -----------------------------
os.makedirs("../results", exist_ok=True)

# Actual top tracks (original song names)
top_tracks_actual.to_csv("../results/top_tracks_actual.csv", index=False)

# Actual top artists (original artist names)
top_artists_actual.to_csv("../results/top_artists_actual.csv", index=False)

# Predicted top tracks
data_tracks_sorted_export = data_tracks_sorted.copy()
data_tracks_sorted_export = data_tracks_sorted_export.rename(
    columns={
        'track_clean': 'Song Name',
        'total_minutes': 'Total Minutes',
        'plays': 'Plays',
        'predicted_plays': 'Predicted Plays'
    }
)
data_tracks_sorted_export.to_csv("../results/top_tracks_predicted.csv", index=False)

# Predicted top artists
data_artists_sorted_export = data_artists_sorted.copy()
data_artists_sorted_export = data_artists_sorted_export.rename(
    columns={
        'artist_clean': 'Artist Name',
        'total_minutes': 'Total Minutes',
        'plays': 'Plays',
        'predicted_plays': 'Predicted Plays'
    }
)
data_artists_sorted_export.to_csv("../results/top_artists_predicted.csv", index=False)

print("All CSVs exported to ../results/")
