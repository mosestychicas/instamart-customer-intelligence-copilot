
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

df = pd.read_csv("outputs/lapse_dataset.csv")

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

# Train and evaluate the original, imbalanced model.
model.fit(X_train, y_train)
original_predictions = model.predict(X_test)

# Oversample only the lapsed class in the training data.
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

# Fit a fresh pipeline so scaling and encoding are learned from
# the oversampled training data only. The test data remain unchanged.
balanced_model = clone(model)
balanced_model.fit(X_balanced, y_balanced)
balanced_predictions = balanced_model.predict(X_test)

def show_results(title, y_true, predictions):
    print("\n" + title)
    print("-" * 55)
    print("Precision:", round(
        precision_score(y_true, predictions, zero_division=0), 4
    ))
    print("Recall:", round(
        recall_score(y_true, predictions, zero_division=0), 4
    ))
    print("F1:", round(
        f1_score(y_true, predictions, zero_division=0), 4
    ))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, predictions, labels=[0, 1]))

print("=" * 65)
print("TRAINING CLASS BALANCE")
print("=" * 65)
print("Original training counts:")
print(y_train.value_counts().sort_index().to_string())
print("\nBalanced training counts:")
print(y_balanced.value_counts().sort_index().to_string())
print("\nTest set unchanged:", len(X_test), "rows")
print("Scaler frequency mean (balanced training):",
      round(
          balanced_model.named_steps["preprocessor"]
          .named_transformers_["numeric"]
          .named_steps["scaler"]
          .mean_[numeric_features.index("frequency")],
          4,
      ))
print("Original training frequency mean:",
      round(X_train["frequency"].mean(), 4))
print("Full cohort frequency mean:",
      round(X["frequency"].mean(), 4))

show_results(
    "ORIGINAL TRAINING DATA",
    y_test,
    original_predictions,
)
show_results(
    "OVERSAMPLED TRAINING DATA",
    y_test,
    balanced_predictions,
)

print("\nOVERSAMPLING COMPARISON COMPLETE")
