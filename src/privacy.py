"""Privacy and data protection utilities for the heart disease prediction system."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import hashlib
from pathlib import Path
import json


class PrivacyManager:
    """Manage privacy-preserving operations on patient data."""
    
    # Personally Identifiable Information (PII) patterns
    PII_FIELDS = ['name', 'email', 'phone', 'address', 'ssn', 'medical_id', 'patient_id']
    
    # Quasi-identifiers that could re-identify with external data
    QUASI_IDENTIFIERS = ['Age', 'Gender', 'Zip_Code', 'Race', 'DOB']
    
    def __init__(self, min_k_anonymity: int = 5):
        """
        Initialize privacy manager.
        
        Args:
            min_k_anonymity: Minimum group size for k-anonymity (default 5)
        """
        self.min_k_anonymity = min_k_anonymity
    
    def anonymize_dataframe(self, df: pd.DataFrame, hash_seed: str = None) -> pd.DataFrame:
        """
        Anonymize a dataframe by removing and hashing sensitive information.
        
        Args:
            df: Input dataframe
            hash_seed: Seed for consistent hashing
            
        Returns:
            Anonymized dataframe
        """
        df_anon = df.copy()
        
        # Remove explicit PII fields if present
        for field in self.PII_FIELDS:
            if field in df_anon.columns:
                df_anon = df_anon.drop(columns=[field])
        
        return df_anon
    
    def age_generalization(self, age: int, bin_size: int = 5) -> str:
        """Generalize age into age groups for privacy."""
        lower = (age // bin_size) * bin_size
        upper = lower + bin_size
        return f"{lower}-{upper}"
    
    def check_k_anonymity(self, df: pd.DataFrame, quasi_identifiers: List[str]) -> Dict:
        """
        Check k-anonymity of a dataset.
        
        k-anonymity ensures each combination of quasi-identifiers 
        appears in at least k records.
        
        Returns:
            Dictionary with k-anonymity metrics
        """
        # Filter to only existing columns
        qi = [col for col in quasi_identifiers if col in df.columns]
        
        if not qi:
            return {"k_anonymity": len(df), "status": "no_quasi_identifiers"}
        
        # Group by quasi-identifiers and count
        group_sizes = df.groupby(qi).size()
        k = group_sizes.min()
        
        violations = (group_sizes < self.min_k_anonymity).sum()
        
        return {
            "k_anonymity": int(k),
            "min_required": self.min_k_anonymity,
            "is_compliant": k >= self.min_k_anonymity,
            "total_groups": len(group_sizes),
            "groups_with_violations": int(violations),
            "smallest_group_size": int(group_sizes.min()),
            "largest_group_size": int(group_sizes.max()),
            "median_group_size": int(group_sizes.median())
        }
    
    def differential_privacy_check(self, df: pd.DataFrame, epsilon: float = 1.0) -> Dict:
        """
        Check if data could benefit from differential privacy.
        
        Returns:
            Assessment and recommendations
        """
        return {
            "method": "differential_privacy",
            "epsilon": epsilon,
            "status": "recommended" if len(df) > 1000 else "optional",
            "description": "Differential privacy adds controlled noise to query results "
                          "to prevent individual record identification",
            "when_to_use": [
                "When sharing aggregated statistics",
                "When publishing research results",
                "When data contains sensitive health information"
            ]
        }
    
    def data_minimization_check(self, df: pd.DataFrame) -> Dict:
        """
        Check if dataset follows data minimization principles.
        
        Data minimization: collect only necessary data for the stated purpose.
        """
        features_by_importance = {
            "critical": [
                "Age", "Gender", "Systolic_BP", "Diastolic_BP", "Heart_Rate",
                "Blood_Sugar_Fasting", "Cholesterol_Total"
            ],
            "important": [
                "Smoking", "Alcohol_Intake", "Physical_Activity", "Diet", "Stress_Level"
            ],
            "optional": [
                "Weight", "Height", "BMI"
            ]
        }
        
        present_features = {
            "critical": [f for f in features_by_importance["critical"] if f in df.columns],
            "important": [f for f in features_by_importance["important"] if f in df.columns],
            "optional": [f for f in features_by_importance["optional"] if f in df.columns]
        }
        
        return {
            "principle": "data_minimization",
            "total_features": len(df.columns),
            "features_by_importance": present_features,
            "assessment": "Good" if len(present_features["optional"]) > 0 else "Excellent",
            "recommendation": "Remove optional features if not used in model"
        }


class DataProtectionPolicy:
    """Define and manage data protection policies."""
    
    def __init__(self):
        self.policies = {}
    
    def create_policy(self, policy_name: str, policy_dict: Dict) -> None:
        """Create a data protection policy."""
        self.policies[policy_name] = {
            "name": policy_name,
            "created": pd.Timestamp.now().isoformat(),
            "content": policy_dict
        }
    
    def get_gdpr_compliance_checklist(self) -> Dict:
        """Get GDPR compliance checklist for healthcare data."""
        return {
            "lawful_basis": {
                "description": "Ensure lawful basis for processing health data",
                "options": [
                    "Explicit consent from data subject",
                    "Contractual necessity (treatment)",
                    "Legal obligation",
                    "Vital interests (emergency treatment)",
                    "Public task",
                    "Legitimate interests (with safeguards)"
                ],
                "status": "⚠️  TO BE DETERMINED"
            },
            "data_subject_rights": {
                "right_to_access": {
                    "description": "Individuals can request copy of their personal data",
                    "implementation": "Provide API endpoint for individual data requests"
                },
                "right_to_rectification": {
                    "description": "Individuals can correct inaccurate data",
                    "implementation": "Allow data update requests with verification"
                },
                "right_to_erasure": {
                    "description": "Individuals can request data deletion (right to be forgotten)",
                    "implementation": "Remove data from training sets, retrain model if needed"
                },
                "right_to_portability": {
                    "description": "Individuals can transfer data to another service",
                    "implementation": "Export data in standard formats (CSV, JSON)"
                },
                "right_to_object": {
                    "description": "Individuals can object to processing",
                    "implementation": "Maintain exclusion list, honor opt-outs"
                }
            },
            "data_minimization": {
                "description": "Only collect data strictly necessary for stated purpose",
                "checklist": [
                    "Have you identified all required data elements?",
                    "Can you achieve your goal with less data?",
                    "Are you collecting historical data unnecessarily?",
                    "Have you set retention periods?",
                    "Do you delete data when no longer needed?"
                ]
            },
            "security_measures": {
                "encryption": "Data encrypted at rest and in transit (TLS 1.2+, AES-256)",
                "access_control": "Role-based access control (RBAC) for data access",
                "audit_logging": "All data access logged and monitored",
                "data_breach_plan": "Incident response plan in place (notify users within 72 hours)"
            },
            "documentation": {
                "data_processing_agreement": "Required between hospital and AI system vendor",
                "privacy_impact_assessment": "Document privacy risks and mitigation",
                "data_retention_schedule": "Define how long each data type is kept",
                "transparency": "Inform patients how their data is used"
            }
        }
    
    def get_hipaa_compliance_checklist(self) -> Dict:
        """Get HIPAA compliance checklist (US healthcare privacy law)."""
        return {
            "administrative_safeguards": {
                "security_management_process": "Identify and manage security risks",
                "authorization_and_control": "Control who accesses health data",
                "security_awareness_training": "Train staff on privacy/security",
                "security_incident_procedures": "Handle breaches and incidents"
            },
            "physical_safeguards": {
                "facility_access_controls": "Restrict physical access to data storage",
                "workstation_security": "Secure computers and devices",
                "device_and_media_controls": "Track and protect storage media"
            },
            "technical_safeguards": {
                "encryption_and_decryption": "Encrypt ePHI at rest and in transit",
                "access_controls": "Unique user IDs, emergency access procedures",
                "audit_controls": "Log and review all ePHI access",
                "integrity_controls": "Detect and prevent data modification"
            },
            "breach_notification": {
                "discovery": "Discover breach without unreasonable delay",
                "notification_timeline": "Notify affected individuals within 60 calendar days",
                "regulatory_notification": "Notify HHS and media if 500+ individuals affected",
                "documentation": "Document all breaches for 6 years"
            }
        }


def generate_privacy_report(
    privacy_manager: PrivacyManager,
    df: pd.DataFrame,
    quasi_identifiers: List[str]
) -> str:
    """Generate a comprehensive privacy report."""
    
    k_anon = privacy_manager.check_k_anonymity(df, quasi_identifiers)
    data_min = privacy_manager.data_minimization_check(df)
    diff_priv = privacy_manager.differential_privacy_check(df)
    
    report = """
╔═════════════════════════════════════════════════════════════════════╗
║                      PRIVACY REPORT                                 ║
║              Heart Disease Prediction System                         ║
╚═════════════════════════════════════════════════════════════════════╝

1. K-ANONYMITY ASSESSMENT
   {'─' * 60}
   k-anonymity: {k_anonymity}
   Minimum Required: {min_required}
   Status: {'✓ COMPLIANT' if is_compliant else '✗ NON-COMPLIANT'}
   
   Statistics:
   - Total unique groups: {total_groups}
   - Smallest group size: {smallest_group_size}
   - Largest group size: {largest_group_size}
   - Median group size: {median_group_size}
   - Groups with violations: {groups_with_violations}

2. DATA MINIMIZATION
   {'─' * 60}
   Assessment: {assessment}
   Total Features: {total_features}
   
   Critical Features (needed): {critical_count}
   Important Features (useful): {important_count}
   Optional Features (consider removing): {optional_count}
   
   Recommendation: {recommendation}

3. PRIVACY TECHNIQUES
   {'─' * 60}
   Differential Privacy: {dp_status}
   Description: {dp_description}
   
   When to apply:
   - Before publishing aggregate statistics
   - When sharing research results
   - For sensitive subgroup analysis

4. COMPLIANCE FRAMEWORKS
   {'─' * 60}
   ✓ GDPR (EU)
     - Data subject rights (access, rectification, erasure)
     - Data Protection Impact Assessment required
     - Privacy by design principles
   
   ✓ HIPAA (US)
     - Encryption of ePHI required
     - Audit logs for all data access
     - Breach notification within 60 days
   
   ✓ CCPA (California)
     - Consumer right to know, delete, opt-out
     - Transparency about data use

5. RECOMMENDATIONS
   {'─' * 60}
   • Implement end-to-end encryption for data in transit
   • Use role-based access control (RBAC)
   • Audit all access to patient health data
   • Establish data retention and deletion policies
   • Conduct annual privacy impact assessments
   • Train staff on data protection
   • Maintain breach response procedures

""".format(
        **k_anon,
        assessment=data_min["assessment"],
        total_features=data_min["total_features"],
        critical_count=len(data_min["features_by_importance"]["critical"]),
        important_count=len(data_min["features_by_importance"]["important"]),
        optional_count=len(data_min["features_by_importance"]["optional"]),
        recommendation=data_min["recommendation"],
        dp_status=diff_priv["status"],
        dp_description=diff_priv["description"]
    )
    
    return report
