"""
Automated peak clustering into 急冲/缓冲/震荡 types.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from datetime import timedelta

# Load 1-minute data (60-day test period)
print("Loading 1-minute RTM data...")
df = pd.read_csv('data/rtm_prices_1min.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Filter to 60-day test period
test_start = pd.Timestamp('2024-01-01')
test_end = pd.Timestamp('2024-03-01')
df = df[(df['timestamp'] >= test_start) & (df['timestamp'] < test_end)].copy()

print(f"Test period: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
print(f"Total rows: {len(df):,}")

# Extract peaks (one per day)
print("\nExtracting daily peaks...")
peaks = []

for date, group in df.groupby(df['timestamp'].dt.date):
    peak_idx = group['price'].idxmax()
    peak_time = group.loc[peak_idx, 'timestamp']
    peak_price = group.loc[peak_idx, 'price']
    
    # Get baseline (20th percentile)
    baseline_price = group['price'].quantile(0.20)
    
    # Get window around peak (2 hours = 120 minutes)
    window_start = peak_time - timedelta(minutes=120)
    window_end = peak_time + timedelta(minutes=120)
    window = df[(df['timestamp'] >= window_start) & (df['timestamp'] <= window_end)].copy()
    
    if len(window) < 10:
        continue
    
    # Calculate features
    # 1. Rise phase: from baseline to peak
    rise_data = window[window['timestamp'] <= peak_time]
    rise_phase = rise_data[rise_data['price'] >= baseline_price]
    rise_time = len(rise_phase) if len(rise_phase) > 0 else 1
    
    # 2. Fall phase: from peak back to baseline level
    fall_data = window[window['timestamp'] > peak_time]
    fall_phase = fall_data[fall_data['price'] >= baseline_price]
    fall_time = len(fall_phase) if len(fall_phase) > 0 else 1
    
    # 3. Rise rate
    price_magnitude = peak_price - baseline_price
    rise_rate = price_magnitude / rise_time if rise_time > 0 else 0
    
    # 4. Velocity (1-minute price changes)
    window['velocity'] = window['price'].diff()
    max_velocity = window['velocity'].abs().max()
    
    # 5. Acceleration (change in velocity)
    window['acceleration'] = window['velocity'].diff()
    max_acceleration = window['acceleration'].abs().max()
    
    # 6. Volatility in 30-min window around peak
    peak_window_30 = window[
        (window['timestamp'] >= peak_time - timedelta(minutes=15)) &
        (window['timestamp'] <= peak_time + timedelta(minutes=15))
    ]
    volatility_30min = peak_window_30['price'].std() if len(peak_window_30) > 0 else 0
    
    peaks.append({
        'date': date,
        'peak_time': peak_time,
        'peak_price': peak_price,
        'baseline_price': baseline_price,
        'rise_time': rise_time,
        'fall_time': fall_time,
        'rise_rate': rise_rate,
        'max_velocity': max_velocity,
        'max_acceleration': max_acceleration,
        'volatility_30min': volatility_30min,
        'price_magnitude': price_magnitude
    })

peaks_df = pd.DataFrame(peaks)
print(f"Extracted {len(peaks_df)} peaks")

# STEP 2: NORMALIZE FEATURES
print("\nNormalizing features...")
feature_cols = ['rise_time', 'fall_time', 'rise_rate', 'max_velocity', 
                'max_acceleration', 'volatility_30min', 'price_magnitude']

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(peaks_df[feature_cols])
scaled_features = [f + '_scaled' for f in feature_cols]
for i, col in enumerate(scaled_features):
    peaks_df[col] = scaled_data[:, i]

# STEP 3: CLUSTER ANALYSIS
print("\nRunning K-means clustering...")
silhouette_scores = {}

for k in [3, 4]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(peaks_df[scaled_features])
    score = silhouette_score(peaks_df[scaled_features], labels)
    silhouette_scores[k] = score
    print(f"k={k}: silhouette score = {score:.3f}")

# Choose optimal k
optimal_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"\nOptimal k: {optimal_k} (silhouette score: {silhouette_scores[optimal_k]:.3f})")

# Apply optimal clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
peaks_df['cluster'] = kmeans.fit_predict(peaks_df[scaled_features])

# STEP 4: ANALYZE CLUSTERS
print("\n" + "="*70)
print("PEAK CLUSTERING ANALYSIS")
print("="*70)
print(f"Optimal k: {optimal_k} (silhouette score: {silhouette_scores[optimal_k]:.3f})")

print("\nCluster characteristics:")
cluster_stats = []

for cluster in range(optimal_k):
    cluster_data = peaks_df[peaks_df['cluster'] == cluster]
    cluster_stats.append({
        'Cluster': cluster,
        'Count': len(cluster_data),
        'Avg Rise Time': cluster_data['rise_time'].mean(),
        'Avg Fall Time': cluster_data['fall_time'].mean(),
        'Avg Velocity': cluster_data['max_velocity'].mean(),
        'Avg Volatility': cluster_data['volatility_30min'].mean(),
        'Rise Rate': cluster_data['rise_rate'].mean()
    })

stats_df = pd.DataFrame(cluster_stats)
print(stats_df.to_string(index=False))

# Map clusters to types based on characteristics
# 急冲 (Rapid Spike): high velocity, short rise time, high rise rate
# 缓冲 (Gradual Rise): low velocity, long rise time, low rise rate  
# 震荡 (Oscillating): high volatility

cluster_mapping = {}
for idx, row in stats_df.iterrows():
    cluster = row['Cluster']
    velocity = row['Avg Velocity']
    rise_time = row['Avg Rise Time']
    volatility = row['Avg Volatility']
    rise_rate = row['Rise Rate']
    
    # Decision logic
    if rise_rate > stats_df['Rise Rate'].median() and velocity > stats_df['Avg Velocity'].median():
        cluster_mapping[cluster] = ('急冲', 'Rapid Spike', 'High velocity, high rise rate')
    elif volatility > stats_df['Avg Volatility'].median():
        cluster_mapping[cluster] = ('震荡', 'Oscillating', 'High volatility')
    else:
        cluster_mapping[cluster] = ('缓冲', 'Gradual Rise', 'Moderate velocity, steady rise')

print("\nProposed mapping:")
for cluster, (chinese, english, features) in cluster_mapping.items():
    count = len(peaks_df[peaks_df['cluster'] == cluster])
    pct = count / len(peaks_df) * 100
    print(f"  Cluster {cluster} → {chinese} ({english})")
    print(f"    Defining features: {features}")
    print(f"    Count: {count} peaks ({pct:.1f}%)")

# Add type labels
peaks_df['type_chinese'] = peaks_df['cluster'].map(lambda x: cluster_mapping[x][0])
peaks_df['type_english'] = peaks_df['cluster'].map(lambda x: cluster_mapping[x][1])

# STEP 5: SHOW EXAMPLES
print("\nGenerating example plots...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(optimal_k, 5, hspace=0.3, wspace=0.3)

for row, cluster in enumerate(range(optimal_k)):
    cluster_peaks = peaks_df[peaks_df['cluster'] == cluster].head(5)
    chinese_name, english_name, _ = cluster_mapping[cluster]
    
    for col, (idx, peak) in enumerate(cluster_peaks.iterrows()):
        ax = fig.add_subplot(gs[row, col])
        
        # Get 2-hour window around peak
        peak_time = peak['peak_time']
        window_start = peak_time - timedelta(minutes=120)
        window_end = peak_time + timedelta(minutes=120)
        
        window_data = df[
            (df['timestamp'] >= window_start) & 
            (df['timestamp'] <= window_end)
        ]
        
        # Plot
        ax.plot(window_data['timestamp'], window_data['price'], 'b-', linewidth=1)
        ax.axvline(peak_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axhline(peak['baseline_price'], color='gray', linestyle=':', alpha=0.5)
        
        ax.set_title(f"{peak['date']}\n${peak['peak_price']:.2f}", fontsize=9, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3)
        
        if col == 0:
            ax.set_ylabel(f"{chinese_name}\n({english_name})", fontsize=10, fontweight='bold')

plt.suptitle('Peak Examples by Type (5 examples each)', fontsize=14, fontweight='bold', y=0.995)
plt.savefig('results/plots/peak_clustering_examples.png', dpi=300, bbox_inches='tight')
print("✓ Examples saved: results/plots/peak_clustering_examples.png")

# STEP 6: PCA VISUALIZATION
print("\nGenerating PCA plot...")
pca = PCA(n_components=2)
pca_features = pca.fit_transform(peaks_df[scaled_features])

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['red', 'blue', 'green', 'orange']

for cluster in range(optimal_k):
    cluster_data = peaks_df[peaks_df['cluster'] == cluster]
    pca_cluster = pca_features[peaks_df['cluster'] == cluster]
    chinese_name, english_name, _ = cluster_mapping[cluster]
    
    ax.scatter(pca_cluster[:, 0], pca_cluster[:, 1], 
              c=colors[cluster], label=f"{chinese_name} ({english_name})", 
              s=100, alpha=0.6, edgecolors='black')

ax.set_xlabel('PCA Component 1', fontsize=12, fontweight='bold')
ax.set_ylabel('PCA Component 2', fontsize=12, fontweight='bold')
ax.set_title('Peak Clustering - PCA 2D Projection', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/peak_clustering_pca.png', dpi=300, bbox_inches='tight')
print("✓ PCA plot saved: results/plots/peak_clustering_pca.png")

# Save classifications
peaks_df.to_csv('data/peak_classifications.csv', index=False)
print("✓ Classifications saved: data/peak_classifications.csv")

print("\n" + "="*70)
print("CLUSTERING COMPLETE")
print("="*70)
print("\nDistribution:")
for cluster in range(optimal_k):
    chinese_name, english_name, _ = cluster_mapping[cluster]
    count = len(peaks_df[peaks_df['cluster'] == cluster])
    pct = count / len(peaks_df) * 100
    print(f"  {chinese_name} ({english_name}): {count} peaks ({pct:.1f}%)")

print("\n⚠️  WARNING: Review example plots carefully!")
print("   If clusters don't make visual sense, adjust features or k value.")
print("="*70)

