
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATA_PATH = "outputs/lapse_dataset.csv"

df = pd.read_csv(DATA_PATH)

target = "lapsed"
drop_columns = ["customer_id", target]

X = df.drop(columns=drop_columns)
y = df[target]

categorical_features = [
    "favourite_category",
    "city",
    "gender",
]
numeric_features = [
    col for col in X.columns
    if col not in categorical_features
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# Preprocessing is fitted inside each CV training fold.
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

print("=" * 70)
print("LAPSE MODEL TRAINING")
print("=" * 70)
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# Logistic Regression grid search.
logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])

logistic_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
}

logistic_search = GridSearchCV(
    logistic_pipeline,
    logistic_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
)
logistic_search.fit(X_train, y_train)

# Random Forest grid search.
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )),
])

rf_grid = {
    "model__max_depth": [3, 5, 8],
    "model__min_samples_leaf": [5, 10, 20],
}

rf_search = GridSearchCV(
    rf_pipeline,
    rf_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
)
rf_search.fit(X_train, y_train)

models = {
    "Logistic Regression": logistic_search,
    "Random Forest": rf_search,
}

for name, search in models.items():
    predictions = search.best_estimator_.predict(X_test)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print("Best parameters:", search.best_params_)
    print("Best CV F1:", round(search.best_score_, 4))
    print("Test accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(
        "Test precision (lapsed):",
        round(precision_score(y_test, predictions, zero_division=0), 4),
    )
    print(
        "Test recall (lapsed):",
        round(recall_score(y_test, predictions, zero_division=0), 4),
    )
    print(
        "Test F1 (lapsed):",
        round(f1_score(y_test, predictions, zero_division=0), 4),
    )
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

print("\nMODEL TRAINING AND COMPARISON COMPLETE")
