
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

# Recreate the training-only oversampling procedure.
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

# Train the Random Forest with the best grid-search parameters.
rf = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
    )),
])
rf.fit(X_balanced, y_balanced)

feature_names = (
    rf.named_steps["preprocessor"].get_feature_names_out()
)
importances = rf.named_steps["model"].feature_importances_

importance_table = pd.DataFrame({
    "feature": feature_names,
    "importance": importances,
}).sort_values("importance", ascending=False)

print("=" * 65)
print("RANDOM FOREST: TOP 5 FEATURE IMPORTANCES")
print("=" * 65)
print(importance_table.head(5).to_string(index=False))

# Train Logistic Regression with C=100 on the same balanced data.
lr = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("model", LogisticRegression(C=100, max_iter=1000)),
])
lr.fit(X_balanced, y_balanced)

lr_names = lr.named_steps["preprocessor"].get_feature_names_out()
lr_coefficients = lr.named_steps["model"].coef_[0]

coefficient_table = pd.DataFrame({
    "feature": lr_names,
    "coefficient": lr_coefficients,
}).sort_values("coefficient", ascending=False)

print("\n" + "=" * 65)
print("LOGISTIC REGRESSION: THREE POSITIVE AND THREE NEGATIVE COEFFICIENTS")
print("=" * 65)

print("\nLargest positive coefficients:")
print(coefficient_table.head(3).to_string(index=False))

print("\nLargest negative coefficients:")
print(
    coefficient_table.tail(3)
    .sort_values("coefficient")
    .to_string(index=False)
)

print("\nFEATURE ANALYSIS COMPLETE")
