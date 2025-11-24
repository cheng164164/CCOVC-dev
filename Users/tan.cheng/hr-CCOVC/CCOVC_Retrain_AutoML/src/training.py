# src/training.py
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, rules: dict, new_label="Other"):
        self.rules = rules
        self.new_label = new_label
        self.top_categories_ = {}

    def fit(self, X, y=None):
        for col, top_n in self.rules.items():
            self.top_categories_[col] = X[col].value_counts().nlargest(top_n).index
        return self

    def transform(self, X):
        for col, top_cats in self.top_categories_.items():
            X[col] = X[col].apply(lambda x: x if x in top_cats else self.new_label)
        return X

def train_and_save_model(df: pd.DataFrame, output_path: str, blob_service_client=None):
    if "CC / OVC" not in df.columns:
        raise ValueError("Missing target column 'CC / OVC'")
    
    selected_features = [
        'Legal Entity (Label)', 'Business unit (Label)', 'Division (Label)', 'Employment Type (Label)' , 'Job Classification (Label)'
        ]
    df = df[["CC / OVC"]+ selected_features]
    df["CC / OVC"] = LabelEncoder().fit_transform(df["CC / OVC"])

    X = df.drop("CC / OVC", axis=1)
    y = df["CC / OVC"]
    cat_features = X.select_dtypes(include="object").columns.tolist()

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Automatically determine how many classes to keep based on min frequency
    min_freq = 1
    job_counts = X_train["Job Classification (Label)"].value_counts()
    top_job_classes = job_counts[job_counts >= min_freq].index.tolist()
    print("Top Job Classes to keep:", top_job_classes)

    # Grouping rules
    grouping_rules = {
        "Job Classification (Label)": len(top_job_classes),
        "Employment Type (Label)": 4,
        "Business unit (Label)": 50,
        "Legal Entity (Label)": 30,
        "Division (Label)": 25,
    }

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for model_name, model in models.items():
        print(f"🔄 Running CV for {model_name}")
        pipeline = Pipeline([
            ("grouper", RareCategoryGrouper(grouping_rules, "Other")),
            ("encoder", TargetEncoder(cols=cat_features)),
            ("classifier", model)
        ])

        scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = pipeline.predict(X.iloc[val_idx])
            acc = accuracy_score(y.iloc[val_idx], preds)
            scores.append(acc)
            results.append({"Model": model_name, "Fold": fold, "Accuracy": acc})

        mean_acc = sum(scores) / len(scores)
        results.append({"Model": model_name, "Fold": "Mean", "Accuracy": mean_acc})
        print(f"✅ {model_name} Mean Accuracy: {mean_acc:.4f}")

    # Save metrics log
    results_df = pd.DataFrame(results)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_file = f"training_log_{timestamp}.csv"
    local_log = f"/tmp/{log_file}"
    results_df.to_csv(local_log, index=False)

    if blob_service_client:
        log_blob = blob_service_client.get_blob_client(container="model-store", blob=f"logs/{log_file}")
        with open(local_log, "rb") as f:
            log_blob.upload_blob(f, overwrite=True)
        print(f"📝 Metrics uploaded: logs/{log_file}")

    # Retrain best model (Random Forest) on full train
    best_pipeline = Pipeline([
        ("grouper", RareCategoryGrouper(grouping_rules, "Other")),
        ("encoder", TargetEncoder(cols=cat_features)),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    best_pipeline.fit(X_train, y_train)
    joblib.dump(best_pipeline, output_path)
    print(f"💾 Best model saved to {output_path}")
