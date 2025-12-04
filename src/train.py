"""Train baseline models on the synthetic heart disease dataset.

Creates a pipeline with preprocessing and trains LogisticRegression and RandomForest.
Saves the best model by ROC AUC to models/best_model.joblib and writes metrics to models/metrics.txt
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
import matplotlib.pyplot as plt
try:
    import shap
    _HAS_SHAP = True
except Exception:
    shap = None
    _HAS_SHAP = False




def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_and_train(X_train, y_train, model_name="logreg"):
    # numeric and categorical column detection
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=[object]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols),
    ])

    if model_name == "logreg":
        clf = LogisticRegression(max_iter=1000)
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf),
    ])

    # cross-val as quick check
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(model, X, y):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    acc = accuracy_score(y, preds)
    report = classification_report(y, preds)
    return {"auc": float(auc), "accuracy": float(acc), "report": report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/synthetic_heart_disease_dataset.csv")
    parser.add_argument("--out", type=str, default="models")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}")
    df = load_data(data_path)

    # quick cleanup: drop rows with all nulls
    df = df.dropna(how="all")

    target = "Heart_Disease"
    if target not in df.columns:
        raise SystemExit(f"Target column {target} not found in data")

    X = df.drop(columns=[target])
    y = df[target]

    # Simple feature engineering: age group and BMI category if BMI present
    def add_features(df_):
        df = df_.copy()
        if "Age" in df.columns:
            bins = [0, 30, 45, 60, 75, 200]
            labels = ["<30", "30-44", "45-59", "60-74", "75+"]
            df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels)
        if "BMI" in df.columns:
            # underweight, normal, overweight, obese
            df["BMI_Category"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 100], labels=["Underweight", "Normal", "Overweight", "Obese"])
        return df

    X = add_features(X)

    # convert some integer-like columns if necessary
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train two models
    print("Training Logistic Regression...")
    logreg_pipe = build_and_train(X_train.copy(), y_train, model_name="logreg")
    print("Training Random Forest...")
    rf_pipe = build_and_train(X_train.copy(), y_train, model_name="rf")

    # Evaluate on test set
    print("Evaluating models on test set...")
    logreg_metrics = evaluate(logreg_pipe, X_test, y_test)
    rf_metrics = evaluate(rf_pipe, X_test, y_test)

    # cross-validation scores (AUC) as additional check
    try:
        logreg_cv = cross_val_score(logreg_pipe, X_train, y_train, cv=5, scoring="roc_auc")
        rf_cv = cross_val_score(rf_pipe, X_train, y_train, cv=5, scoring="roc_auc")
    except Exception:
        logreg_cv = None
        rf_cv = None


    # choose best by AUC
    best_model = logreg_pipe
    best_metrics = logreg_metrics
    best_name = "logistic_regression"
    if rf_metrics["auc"] > logreg_metrics["auc"]:
        best_model = rf_pipe
        best_metrics = rf_metrics
        best_name = "random_forest"

    # save model
    model_path = out_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)

    # SHAP explainability for best model (only tree or linear supported well)
    try:
        expl_path = out_dir / "shap_summary.png"
        # prepare a small sample for SHAP (use test set)
        X_test_sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
        # need to apply preprocessing to get numeric array for shap
        preprocessor = best_model.named_steps["preprocessor"]
        clf = best_model.named_steps["clf"]
        X_trans = preprocessor.transform(X_test_sample)

        if hasattr(clf, "feature_importances_") or "RandomForest" in clf.__class__.__name__:
            expl = shap.TreeExplainer(clf)
            shap_values = expl.shap_values(X_trans)
            # shap summary plot
            plt.figure(figsize=(8, 6))
            try:
                shap.summary_plot(shap_values, X_trans, show=False)
                plt.tight_layout()
                plt.savefig(expl_path, dpi=150)
                plt.close()
            except Exception:
                # fallback: feature importance
                importances = clf.feature_importances_
                plt.bar(range(len(importances)), importances)
                plt.title("Feature importances (preprocessed features)")
                plt.savefig(expl_path, dpi=150)
                plt.close()
        else:
            # try KernelExplainer (slower) or skip
            try:
                expl = shap.KernelExplainer(clf.predict_proba, X_trans[:50])
                shap_values = expl.shap_values(X_trans[:100])
                plt.figure(figsize=(8, 6))
                shap.summary_plot(shap_values, X_trans[:100], show=False)
                plt.tight_layout()
                plt.savefig(expl_path, dpi=150)
                plt.close()
            except Exception:
                pass
    except Exception:
        pass

    # write metrics
    metrics_path = out_dir / "metrics.txt"
    with metrics_path.open("w") as f:
        f.write(f"best_model: {best_name}\n")
        f.write(f"logistic_regression_auc: {logreg_metrics['auc']:.4f}\n")
        f.write(f"random_forest_auc: {rf_metrics['auc']:.4f}\n\n")
        f.write("--- logistic classification report ---\n")
        f.write(logreg_metrics["report"] + "\n")
        f.write("--- random forest classification report ---\n")
        f.write(rf_metrics["report"] + "\n")

    print(f"Saved best model ({best_name}) to {model_path}")
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
