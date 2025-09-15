import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from category_encoders import TargetEncoder
from datetime import datetime


# =========================
# Custom RareCategoryGrouper
# =========================
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, rules: dict, new_label="Other"):
        self.rules = rules
        self.new_label = new_label
        self.top_categories_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col, top_n in self.rules.items():
            self.top_categories_[col] = (
                X[col].value_counts().nlargest(top_n).index
            )
        return self

    def transform(self, X):
        X = X.copy()
        for col, top_cats in self.top_categories_.items():
            X[col] = X[col].apply(
                lambda x: x if x in top_cats else self.new_label
            )
        return X


def train_and_save_model(df: pd.DataFrame, output_path: str, blob_service_client=None):
    """
    Train Logistic Regression & Random Forest with 5-fold CV,
    log metrics to blob storage, and save final Random Forest model.
    """

    # ===== Step 1: Drop high-cardinality and irrelevant columns =====
    drop_cols = [
        "Cost Center (Label)",
        "Job Title",
        "Legal Entity (Label)",
        "Position Position Title (Label)",
        "Physical Location (Location Name)",
        "Position Business Segment (Picklist Label)",
        "Business unit (Label)",
        "Work Contract (Picklist Label)",
        "Employment Classification (Label)",
        "Job Function (Label)",
        "Job Classification (Label)"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.dropna()

    # ===== Step 2: Encode target =====
    if "CC / OVC" not in df.columns:
        raise ValueError("Target column 'CC / OVC' not found in dataset.")
    df["CC / OVC"] = LabelEncoder().fit_transform(df["CC / OVC"])

    # ===== Step 3: Split features and target =====
    X = df.drop("CC / OVC", axis=1)
    y = df["CC / OVC"]

    categorical_features = X.select_dtypes(include="object").columns.tolist()

    # ===== Step 4: Build CV Setup =====
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grouping_rules = {
        "Employee Type (Label)": 2,
        "Employment Type (Label)": 2
    }

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = []

    # ===== Step 5: Cross-validation =====
    for model_name, model in models.items():
        print(f"🔄 Running CV for {model_name}...")
        pipeline = Pipeline([
            ("grouper", RareCategoryGrouper(rules=grouping_rules, new_label="Other")),
            ("encoder", TargetEncoder(cols=categorical_features)),
            ("classifier", model)
        ])

        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict(X_val)
            acc = accuracy_score(y_val, preds)
            fold_scores.append(acc)

            results.append({
                "Model": model_name,
                "Fold": fold,
                "Accuracy": acc
            })

        mean_acc = sum(fold_scores) / len(fold_scores)
        results.append({
            "Model": model_name,
            "Fold": "Mean",
            "Accuracy": mean_acc
        })
        print(f"✅ {model_name} Mean Accuracy: {mean_acc:.4f}")

    # ===== Step 6: Save metrics log =====
    results_df = pd.DataFrame(results)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{timestamp}.csv"
    local_log_path = f"/tmp/{log_filename}"
    results_df.to_csv(local_log_path, index=False)

    if blob_service_client:
        log_blob = blob_service_client.get_blob_client(container="model-store", blob=f"logs/{log_filename}")
        with open(local_log_path, "rb") as data:
            log_blob.upload_blob(data, overwrite=True)
        print(f"📝 Training metrics uploaded to blob as logs/{log_filename}")

    # ===== Step 7: Retrain best model (Random Forest) on full data =====
    final_pipeline = Pipeline([
        ("grouper", RareCategoryGrouper(rules=grouping_rules, new_label="Other")),
        ("encoder", TargetEncoder(cols=categorical_features)),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    final_pipeline.fit(X, y)

    # ===== Step 8: Save production model =====
    joblib.dump(final_pipeline, output_path)
    print(f"💾 Final Random Forest model trained on full dataset and saved to {output_path}")
