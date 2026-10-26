
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

DATA_PATH = "outputs/lapse_dataset.csv"

df = pd.read_csv(DATA_PATH)

target = "lapsed"
X = df.drop(columns=["customer_id", target])
y = df[target]

# Keep the same class proportions in train and test.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# Baseline: predict every customer as not lapsed (class 0).
y_pred = pd.Series(0, index=y_test.index)

print("=" * 65)
print("LAPSE PREDICTION: BASELINE CHECKPOINTS")
print("=" * 65)
print("Full cohort:", len(df))
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Training lapsed:", int(y_train.sum()))
print("Test lapsed:", int(y_test.sum()))
print("Test lapsed rate (%):", round(y_test.mean() * 100, 2))

print("\nBaseline always predicts not lapsed")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Lapsed-class F1:", round(f1_score(y_test, y_pred, zero_division=0), 4))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nBASELINE EVALUATION COMPLETE")
