"""Fairness and bias detection for the heart disease prediction model."""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple


class FairnessAnalyzer:
    """Analyze model predictions for fairness across demographic groups."""
    
    PROTECTED_ATTRIBUTES = ['Gender', 'Age_Group']
    FAIRNESS_THRESHOLD = 0.8  # 80% rule for fairness
    
    def __init__(self):
        self.fairness_reports = {}
    
    def analyze_demographic_parity(
        self, 
        y_pred: np.ndarray, 
        y_true: np.ndarray,
        protected_attr: pd.Series,
        group_name: str = "demographic"
    ) -> Dict:
        """
        Analyze demographic parity: P(pred=1|Group A) ≈ P(pred=1|Group B)
        
        A model has demographic parity if positive prediction rates are 
        equal across demographic groups.
        """
        groups = protected_attr.unique()
        group_stats = {}
        
        for group in groups:
            mask = protected_attr == group
            group_pred_rate = np.mean(y_pred[mask])
            group_stats[str(group)] = {
                "positive_rate": float(group_pred_rate),
                "sample_size": int(np.sum(mask))
            }
        
        # Check 80% rule
        rates = [v["positive_rate"] for v in group_stats.values()]
        min_rate = min(rates) if rates else 0
        max_rate = max(rates) if rates else 1
        
        disparate_impact = min_rate / max_rate if max_rate > 0 else 1.0
        
        return {
            "metric": "demographic_parity",
            "attribute": group_name,
            "group_stats": group_stats,
            "disparate_impact_ratio": float(disparate_impact),
            "is_fair": disparate_impact >= self.FAIRNESS_THRESHOLD,
            "threshold": self.FAIRNESS_THRESHOLD
        }
    
    def analyze_equal_opportunity(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        protected_attr: pd.Series,
        group_name: str = "demographic"
    ) -> Dict:
        """
        Analyze equal opportunity: TPR(Group A) ≈ TPR(Group B)
        
        True positive rate (recall) should be equal across groups.
        This ensures the model has equal sensitivity for positive class.
        """
        groups = protected_attr.unique()
        group_stats = {}
        
        for group in groups:
            mask = protected_attr == group
            group_true = y_true[mask]
            group_pred = y_pred[mask]
            
            # Calculate TPR only for positive class
            positive_mask = group_true == 1
            if positive_mask.sum() > 0:
                tpr = np.mean(group_pred[positive_mask] == 1)
            else:
                tpr = np.nan
            
            group_stats[str(group)] = {
                "tpr": float(tpr) if not np.isnan(tpr) else None,
                "positive_samples": int(positive_mask.sum())
            }
        
        # Check fairness
        tprs = [v["tpr"] for v in group_stats.values() if v["tpr"] is not None]
        if len(tprs) > 1:
            tpr_disparity = max(tprs) - min(tprs)
            is_fair = tpr_disparity < 0.1  # TPR difference < 10%
        else:
            tpr_disparity = 0
            is_fair = True
        
        return {
            "metric": "equal_opportunity",
            "attribute": group_name,
            "group_stats": group_stats,
            "tpr_disparity": float(tpr_disparity),
            "is_fair": is_fair,
            "tolerance": 0.1
        }
    
    def analyze_calibration(
        self,
        y_pred_proba: np.ndarray,
        y_true: np.ndarray,
        protected_attr: pd.Series,
        group_name: str = "demographic"
    ) -> Dict:
        """
        Analyze calibration: P(Y=1|score=s) should be equal across groups
        
        Calibration measures if predicted probabilities match actual outcomes
        across demographic groups.
        """
        groups = protected_attr.unique()
        group_stats = {}
        
        # Bin predictions into deciles
        bins = np.linspace(0, 1, 11)
        
        for group in groups:
            mask = protected_attr == group
            group_proba = y_pred_proba[mask]
            group_true = y_true[mask]
            
            # Calculate calibration error for this group
            calibration_errors = []
            for i in range(len(bins) - 1):
                bin_mask = (group_proba >= bins[i]) & (group_proba < bins[i+1])
                if bin_mask.sum() > 0:
                    avg_pred = group_proba[bin_mask].mean()
                    avg_true = group_true[bin_mask].mean()
                    error = abs(avg_pred - avg_true)
                    calibration_errors.append(error)
            
            mean_calibration_error = np.mean(calibration_errors) if calibration_errors else np.nan
            
            group_stats[str(group)] = {
                "calibration_error": float(mean_calibration_error) if not np.isnan(mean_calibration_error) else None,
                "sample_size": int(mask.sum())
            }
        
        return {
            "metric": "calibration",
            "attribute": group_name,
            "group_stats": group_stats
        }
    
    def comprehensive_fairness_audit(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        X_demographics: pd.DataFrame
    ) -> Dict:
        """Run comprehensive fairness analysis across all protected attributes."""
        
        results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_samples": len(y_true),
            "metrics": {}
        }
        
        for attr in self.PROTECTED_ATTRIBUTES:
            if attr not in X_demographics.columns:
                continue
            
            # Demographic parity
            dp_result = self.analyze_demographic_parity(
                y_pred, y_true, X_demographics[attr], attr
            )
            
            # Equal opportunity
            eo_result = self.analyze_equal_opportunity(
                y_pred, y_true, X_demographics[attr], attr
            )
            
            # Calibration
            calib_result = self.analyze_calibration(
                y_pred_proba, y_true, X_demographics[attr], attr
            )
            
            results["metrics"][attr] = {
                "demographic_parity": dp_result,
                "equal_opportunity": eo_result,
                "calibration": calib_result
            }
        
        # Overall fairness assessment
        all_fair = all(
            results["metrics"][attr]["demographic_parity"]["is_fair"]
            and results["metrics"][attr]["equal_opportunity"]["is_fair"]
            for attr in results["metrics"]
        )
        
        results["overall_assessment"] = {
            "is_fair": all_fair,
            "recommendations": self._generate_recommendations(results)
        }
        
        return results
    
    def _generate_recommendations(self, audit_results: Dict) -> List[str]:
        """Generate fairness improvement recommendations."""
        recommendations = []
        
        for attr, metrics in audit_results["metrics"].items():
            if not metrics["demographic_parity"]["is_fair"]:
                disparate_impact = metrics["demographic_parity"]["disparate_impact_ratio"]
                recommendations.append(
                    f"⚠️  Demographic parity issue for {attr}: "
                    f"disparate impact ratio = {disparate_impact:.2%} (< 80%)"
                )
            
            if not metrics["equal_opportunity"]["is_fair"]:
                tpr_disparity = metrics["equal_opportunity"]["tpr_disparity"]
                recommendations.append(
                    f"⚠️  Equal opportunity issue for {attr}: "
                    f"TPR disparity = {tpr_disparity:.2%}"
                )
        
        if not recommendations:
            recommendations.append("✓ Model appears fair across demographic groups")
        
        recommendations.append("\n📋 Recommendations:")
        recommendations.append("• Monitor fairness metrics continuously in production")
        recommendations.append("• Collect feedback from diverse demographic groups")
        recommendations.append("• Consider retraining with balanced sampling if biases detected")
        recommendations.append("• Use model explanations (SHAP) to understand disparities")
        
        return recommendations


def generate_fairness_report(audit_results: Dict) -> str:
    """Generate a human-readable fairness report."""
    
    report = """
╔═══════════════════════════════════════════════════════════════════╗
║                   FAIRNESS AUDIT REPORT                          ║
║              Heart Disease Prediction Model                       ║
╚═══════════════════════════════════════════════════════════════════╝

TIMESTAMP: {timestamp}
SAMPLES ANALYZED: {total_samples}

""".format(**audit_results)
    
    if audit_results.get("overall_assessment", {}).get("is_fair"):
        report += "✓ OVERALL ASSESSMENT: FAIR\n\n"
    else:
        report += "⚠️  OVERALL ASSESSMENT: POTENTIAL BIAS DETECTED\n\n"
    
    for attr, metrics in audit_results.get("metrics", {}).items():
        report += f"\n{'─' * 60}\n"
        report += f"Protected Attribute: {attr}\n"
        report += f"{'─' * 60}\n"
        
        # Demographic Parity
        dp = metrics["demographic_parity"]
        report += f"\n1. DEMOGRAPHIC PARITY\n"
        report += f"   Status: {'✓ PASS' if dp['is_fair'] else '✗ FAIL'}\n"
        report += f"   Disparate Impact Ratio: {dp['disparate_impact_ratio']:.2%}\n"
        report += f"   Threshold: {dp['threshold']:.0%}\n"
        for group, stats in dp["group_stats"].items():
            report += f"   - {group}: {stats['positive_rate']:.2%} positive predictions "
            report += f"(n={stats['sample_size']})\n"
        
        # Equal Opportunity
        eo = metrics["equal_opportunity"]
        report += f"\n2. EQUAL OPPORTUNITY\n"
        report += f"   Status: {'✓ PASS' if eo['is_fair'] else '✗ FAIL'}\n"
        report += f"   TPR Disparity: {eo['tpr_disparity']:.2%}\n"
        for group, stats in eo["group_stats"].items():
            if stats['tpr'] is not None:
                report += f"   - {group}: TPR = {stats['tpr']:.2%} "
                report += f"(n_positive={stats['positive_samples']})\n"
    
    report += f"\n{'═' * 60}\n"
    report += "RECOMMENDATIONS:\n"
    report += f"{'═' * 60}\n"
    for rec in audit_results.get("overall_assessment", {}).get("recommendations", []):
        report += f"{rec}\n"
    
    return report
