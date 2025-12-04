"""Model monitoring and drift detection system."""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats


class ModelMonitor:
    """Monitor model performance and detect data drift."""
    
    def __init__(self, baseline_path: str = "models/baseline_stats.json"):
        self.baseline_path = Path(baseline_path)
        self.baseline = self._load_baseline()
    
    def _load_baseline(self) -> dict:
        """Load baseline statistics from file."""
        if self.baseline_path.exists():
            with open(self.baseline_path) as f:
                return json.load(f)
        return {}
    
    def save_baseline(self, X: pd.DataFrame):
        """Save baseline statistics for drift detection."""
        baseline = {
            "timestamp": datetime.now().isoformat(),
            "features": {},
            "target_distribution": None
        }
        
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                baseline["features"][col] = {
                    "type": "numeric",
                    "mean": float(X[col].mean()),
                    "std": float(X[col].std()),
                    "min": float(X[col].min()),
                    "max": float(X[col].max()),
                    "q1": float(X[col].quantile(0.25)),
                    "q3": float(X[col].quantile(0.75))
                }
            else:
                baseline["features"][col] = {
                    "type": "categorical",
                    "unique_values": int(X[col].nunique()),
                    "top_value": str(X[col].mode()[0]) if len(X[col].mode()) > 0 else None
                }
        
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_path, 'w') as f:
            json.dump(baseline, f, indent=2)
    
    def detect_drift(self, X_new: pd.DataFrame, threshold: float = 0.05) -> dict:
        """
        Detect data drift using statistical tests.
        
        Returns:
            Dictionary with drift detection results
        """
        if not self.baseline:
            return {"status": "no_baseline", "drifts": []}
        
        drifts = []
        baseline_features = self.baseline.get("features", {})
        
        for col in X_new.columns:
            if col not in baseline_features:
                continue
            
            baseline_info = baseline_features[col]
            
            if baseline_info["type"] == "numeric":
                # KS test for numeric features
                ks_stat, p_value = stats.ks_2samp(
                    X_new[col].dropna(),
                    np.random.normal(
                        loc=baseline_info["mean"],
                        scale=baseline_info["std"],
                        size=len(X_new)
                    )
                )
                
                if p_value < threshold:
                    drifts.append({
                        "feature": col,
                        "type": "numeric",
                        "test": "ks_test",
                        "p_value": float(p_value),
                        "current_mean": float(X_new[col].mean()),
                        "baseline_mean": baseline_info["mean"]
                    })
        
        return {
            "status": "drift_detected" if drifts else "no_drift",
            "timestamp": datetime.now().isoformat(),
            "drifts": drifts,
            "threshold": threshold
        }


def check_prediction_quality(predictions: np.ndarray, probabilities: np.ndarray) -> dict:
    """Check prediction quality metrics."""
    return {
        "prediction_distribution": {
            "class_0": int(np.sum(predictions == 0)),
            "class_1": int(np.sum(predictions == 1))
        },
        "confidence_stats": {
            "mean_confidence": float(np.max(probabilities, axis=1).mean()),
            "min_confidence": float(np.max(probabilities, axis=1).min()),
            "low_confidence_count": int(np.sum(np.max(probabilities, axis=1) < 0.6))
        }
    }
