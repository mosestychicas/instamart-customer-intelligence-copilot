
from pathlib import Path
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parent
RFM_PATH = BASE_DIR / "outputs" / "rfm.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "kmeans_evaluation.csv"

print("=" * 65)
print("K-MEANS CLUSTER EVALUATION")
print("=" * 65)

rfm = pd.read_csv(RFM_PATH)

features = ["recency", "frequency", "monetary"]
X = rfm[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42,
    )

    labels = model.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)

    results.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette_score": silhouette,
    })

    print(
        f"K={k} | "
        f"Inertia={model.inertia_:.1f} | "
        f"Silhouette={silhouette:.3f}"
    )

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print("\nEvaluation table:")
print(results_df.round(3).to_string(index=False))

print(f"\nResults saved to: {OUTPUT_PATH}")
print("\nCLUSTER EVALUATION COMPLETE")
