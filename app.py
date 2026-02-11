# 🚕 Taxi Pickup Hotspot Discovery Dashboard
# DBSCAN • Slider Sampling • eps Plot • Best Plot

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# APP TITLE

st.set_page_config(page_title="Taxi Hotspot Discovery", layout="wide")

st.title("🚕 Taxi Pickup Hotspot Discovery Dashboard")

st.markdown(
"""
This dashboard applies DBSCAN clustering to discover taxi pickup hotspots,
ignore random pickups, and analyze demand density.
"""
)

# LOAD DATASET (DIRECT)

DATA_PATH = "taxi_sample.csv"

try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
except:
    df = pd.read_csv(DATA_PATH, encoding="latin1")

st.success("Dataset loaded successfully.")

# FEATURE SELECTION

X = df[['pickup_latitude', 'pickup_longitude']].dropna()

# SCALING

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SAMPLE SIZE SLIDER

st.sidebar.header("Sampling Control")

sample_size = st.sidebar.slider(
    "Select Sample Size",
    1000,
    min(20000, len(X_scaled)),
    min(10000, len(X_scaled)),
    step=1000
)

idx = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[idx]

st.write(f"Clustering on {sample_size} pickup points.")

# DBSCAN EXPERIMENTS

eps_values = [0.2, 0.3, 0.5]

results = {}
sil_scores = {}

st.subheader("DBSCAN Experiment Evaluation")

for eps in eps_values:

    model = DBSCAN(eps=eps, min_samples=5, algorithm="ball_tree")
    labels = model.fit_predict(X_sample)

    results[eps] = labels

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    noise_ratio = n_noise / len(labels)

    st.write(f"### eps = {eps}")
    st.write("Clusters:", n_clusters)
    st.write("Noise Points:", n_noise)
    st.write("Noise Ratio:", round(noise_ratio, 3))

    # Silhouette Score
    if n_clusters > 1:
        X_non_noise = X_sample[labels != -1]
        labels_non_noise = labels[labels != -1]

        score = silhouette_score(X_non_noise, labels_non_noise)
        sil_scores[eps] = score

        st.write("Silhouette Score:", round(score, 3))
    else:
        sil_scores[eps] = None
        st.write("Silhouette Score: Not Applicable")

    st.markdown("---")

# EPS SELECTION → CLUSTER PLOT

selected_eps = st.selectbox(
    "Select eps to visualize clusters",
    eps_values
)

def plot_clusters(X, labels, title):

    fig, ax = plt.subplots(figsize=(7,5))

    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap='rainbow',
        s=10
    )

    ax.scatter(
        X[labels == -1, 0],
        X[labels == -1, 1],
        color='black',
        s=10,
        label='Noise'
    )

    ax.set_title(title)
    ax.set_xlabel("Pickup Latitude (scaled)")
    ax.set_ylabel("Pickup Longitude (scaled)")
    ax.legend()

    st.pyplot(fig)

st.subheader(f"Cluster Plot — eps = {selected_eps}")

plot_clusters(
    X_sample,
    results[selected_eps],
    f"DBSCAN Clusters (eps = {selected_eps})"
)

# BEST EPS SELECTION + PLOT

best_eps = None
best_score = -1

for eps, score in sil_scores.items():

    if score is not None and score > best_score:
        best_score = score
        best_eps = eps

st.subheader(f"🏆 Best eps Identified = {best_eps}")

plot_clusters(
    X_sample,
    results[best_eps],
    f"Best DBSCAN Clusters (eps = {best_eps})"
)

# BUSINESS INSIGHT

st.info(
    "Clusters represent high-demand pickup hotspots. Noise points indicate rare or random pickups. "
    "These insights help optimize driver allocation and surge pricing."
)
