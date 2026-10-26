
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

DATA_PATH = "outputs/lapse_dataset.csv"
SEGMENTS_PATH = "outputs/customer_segments.csv"
MODEL_PATH = "outputs/lapse_model.joblib"
RETENTION_PATH = "outputs/retention_list.csv"

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(DATA_PATH)
segments = pd.read_csv(SEGMENTS_PATH)

X = df.drop(columns=["customer_id", "lapsed"])
y = df["lapsed"]

categorical_features = ["favourite_category", "city", "gender"]
numeric_features = [
    col for col in X.columns if col not in categorical_features
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# Reproduce the same oversampling procedure using training rows only.
train_data = X_train.copy()
train_data["lapsed"] = y_train.to_numpy()

majority_count = int(train_data["lapsed"].value_counts().max())
minority_data = train_data[train_data["lapsed"] == 1]

sampled_minority = minority_data.sample(
    n=majority_count - len(minority_data),
    replace=True,
    random_state=42,
)

balanced_train = pd.concat(
    [train_data, sampled_minority],
    ignore_index=True,
).sample(frac=1, random_state=42).reset_index(drop=True)

X_balanced = balanced_train.drop(columns=["lapsed"])
y_balanced = balanced_train["lapsed"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
    )),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(C=100, max_iter=1000)),
])

model.fit(X_balanced, y_balanced)

test_predictions = model.predict(X_test)
test_f1 = f1_score(y_test, test_predictions, zero_division=0)

# Save, reload, and confirm predictions are unchanged.
joblib.dump(model, MODEL_PATH)
loaded_model = joblib.load(MODEL_PATH)

reloaded_predictions = loaded_model.predict(X_test)
predictions_match = np.array_equal(
    test_predictions, reloaded_predictions
)

if not predictions_match:
    raise AssertionError("Reloaded model predictions do not match.")

# Use predicted probabilities to rank Premium customers.
# Premium membership is joined only after model training.
premium_ids = set(
    segments.loc[segments["is_premium"] == 1, "customer_id"]
)

customer_rows = df.loc[
    df["customer_id"].isin(premium_ids)
].copy()

premium_X = customer_rows.drop(columns=["customer_id", "lapsed"])
premium_probabilities = loaded_model.predict_proba(premium_X)[:, 1]

retention = pd.DataFrame({
    "customer_id": customer_rows["customer_id"].to_numpy(),
    "lapse_probability": premium_probabilities,
    "actually_lapsed": customer_rows["lapsed"].to_numpy(),
})

retention = retention.sort_values(
    "lapse_probability",
    ascending=False,
).head(10)

retention["lapse_probability"] = retention[
    "lapse_probability"
].round(4)

retention.to_csv(RETENTION_PATH, index=False)

print("=" * 65)
print("FINAL MODEL CHECKPOINTS")
print("=" * 65)
print("Balanced training rows:", len(X_balanced))
print("Training class counts:")
print(y_balanced.value_counts().sort_index().to_string())
print("Oversampled model test F1:", round(test_f1, 4))
print("Saved model:", os.path.abspath(MODEL_PATH))
print("Reloaded predictions identical:", predictions_match)
print("Premium customers:", len(premium_ids))
print("\nTop 10 Premium customers by predicted lapse probability:")
print(retention.to_string(index=False))
print(
    "\nActually lapsed among top 10:",
    int(retention["actually_lapsed"].sum()),
)
print("Saved retention list:", os.path.abspath(RETENTION_PATH))
print("\nFINAL MODEL AND RETENTION LIST COMPLETE")
