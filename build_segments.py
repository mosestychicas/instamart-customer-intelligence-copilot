
import os
import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DB_PATH = "data/instamart.db"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
K = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

with sqlite3.connect(DB_PATH) as conn:
    rfm = pd.read_sql_query(
        """
        SELECT
            c.customer_id,
            CAST(julianday('2025-10-01') - julianday(
                MAX(o.order_ts)
            ) AS INTEGER) AS recency,
            COUNT(DISTINCT o.order_id) AS frequency,
            ROUND(SUM(o.order_amount), 2) AS monetary
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        WHERE o.status = 'completed'
        GROUP BY c.customer_id
        """,
        conn,
    )

# Prepare RFM features and scale them before clustering.
features = ["recency", "frequency", "monetary"]
X = rfm[features].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=K,
    n_init=10,
    random_state=RANDOM_STATE,
)
rfm["segment_id"] = kmeans.fit_predict(X_scaled)

# Review the customer count and average RFM values per cluster.
profiles = (
    rfm.groupby("segment_id")
    .agg(
        customer_count=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
    .round(2)
)

print("=" * 70)
print("CUSTOMER SEGMENT PROFILES")
print("=" * 70)
print(profiles.to_string())

# Give the cluster with the highest average spend the Premium label.
premium_segment = profiles["mean_monetary"].idxmax()

# Assign business-friendly names using each cluster's RFM profile.
segment_names = {}
for segment_id, profile in profiles.iterrows():
    if segment_id == premium_segment:
        name = "Premium"
    elif (
        profile["mean_frequency"] >= profiles["mean_frequency"].median()
        and profile["mean_recency"] <= profiles["mean_recency"].median()
    ):
        name = "Loyal"
    elif profile["mean_recency"] > profiles["mean_recency"].median():
        name = "At Risk"
    else:
        name = "Regular"

    segment_names[segment_id] = name

rfm["segment_name"] = rfm["segment_id"].map(segment_names)
rfm["is_premium"] = (rfm["segment_id"] == premium_segment).astype(int)

# PCA summary for a two-dimensional view of the scaled RFM data.
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca.fit(X_scaled)

print("\nPCA explained variance ratio:")
print(np.round(pca.explained_variance_ratio_, 3))
print(
    "Total variance explained:",
    round(pca.explained_variance_ratio_.sum(), 3),
)

output_path = os.path.join(OUTPUT_DIR, "customer_segments.csv")
rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]
].to_csv(output_path, index=False)

print(f"\nSaved customer segments to: {os.path.abspath(output_path)}")
print("\nSegment name mapping:", segment_names)
print("\nCUSTOMER SEGMENTATION COMPLETE")
