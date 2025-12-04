# Privacy, Ethics & Fairness Documentation

Comprehensive guide to privacy protection, fairness assurance, and ethical AI practices in the Heart Disease Prediction System.

---

## 📋 Table of Contents

1. [Privacy Framework](#privacy-framework)
2. [Fairness Analysis](#fairness-analysis)
3. [Ethics Principles](#ethics-principles)
4. [Compliance Checklists](#compliance-checklists)
5. [Implementation Guide](#implementation-guide)
6. [Risk Assessment](#risk-assessment)

---

## 🔒 Privacy Framework

### K-Anonymity

K-anonymity ensures that individual records cannot be uniquely identified from a dataset. A dataset has k-anonymity when every combination of identifying attributes occurs in at least k records.

**Implementation:**
```python
from src.privacy import PrivacyManager

manager = PrivacyManager(min_k_anonymity=5)
manager.check_k_anonymity(['Age', 'Gender', 'Smoking'], df)
```

**Quasi-Identifiers in Dataset:**
- Age (continuous - generalize to ranges)
- Gender (binary - limited generalization)
- Smoking status (binary)
- Physical Activity level
- Geographic location (if available)

**Default Threshold:** k ≥ 5 (5-anonymity)
- k=2: Weak protection (needs 2 people with same attributes)
- k=5: Moderate protection (standard in healthcare)
- k=10+: Strong protection (recommended for sensitive data)

### Data Minimization

Collect only necessary data for stated purposes. Track which features are:
- **Essential**: Required for model training (age, BP, cholesterol, etc.)
- **Optional**: Nice-to-have but not critical
- **Sensitive**: Protected by law (age if relates to discrimination)

**Feature Classification:**
```python
ESSENTIAL_FEATURES = [
    'Systolic_BP', 'Diastolic_BP', 'Cholesterol_Total',
    'Heart_Rate', 'Blood_Sugar_Fasting'
]

OPTIONAL_FEATURES = [
    'Diet', 'Stress_Level', 'Physical_Activity'
]

SENSITIVE_ATTRIBUTES = ['Age', 'Gender']  # For fairness monitoring
```

**Policy:**
- Only collect essential features by default
- Request explicit consent for optional data
- Minimize retention to necessary period
- Delete data when purpose ends

### Encryption & Data Security

**Data in Transit:**
- TLS 1.2 minimum for all API communications
- HTTPS enforced on all endpoints
- Certificate pinning for sensitive clients

**Data at Rest:**
- Database encryption using AES-256
- PII encrypted at field level
- Encryption keys stored separately in vault

**Access Control:**
```python
# Example: Role-based access
ROLES = {
    'analyst': ['read_reports', 'view_dashboards'],
    'admin': ['read_reports', 'modify_models', 'manage_users'],
    'scientist': ['full_access'],
    'auditor': ['read_logs', 'view_audit_trail']
}
```

### Audit Logging

All data access and model decisions logged for audit trails:

```python
from src.logger import logger, JSONFormatter

logger.info('prediction_made', extra={
    'timestamp': datetime.now().isoformat(),
    'user_id': user_id,
    'feature_hash': hash(patient_data),
    'prediction': prediction,
    'confidence': probability
})
```

**Logged Events:**
- Data access (who, when, what)
- Model training (dataset size, features, performance)
- Predictions (patient identifier, features, result)
- System access (logins, permission changes)
- Data modification (updates, deletions, corrections)

---

## ⚖️ Fairness Analysis

### Demographic Parity

Ensures equal positive prediction rates across demographic groups.

**Definition:** 
$$P(\hat{Y}=1|A=a_1) = P(\hat{Y}=1|A=a_2)$$

Where:
- $\hat{Y}$ = predicted label
- $A$ = protected attribute (e.g., gender, age group)
- $a_1, a_2$ = different values of protected attribute

**Example - Gender Parity:**
```
Male group:   60% positive predictions
Female group: 60% positive predictions
Ratio: 1.0 (perfectly balanced)
```

**Acceptable Range:** 0.8-1.2 (80-120% ratio)
- <0.8 or >1.2: Potential unfairness
- Intervention: Adjust decision threshold or retrain

**Implementation:**
```python
from src.fairness import FairnessAnalyzer

analyzer = FairnessAnalyzer()
parity = analyzer.analyze_demographic_parity(
    y_pred=predictions,
    X_demographics={'Gender': gender_values}
)
```

### Equal Opportunity (TPR Parity)

Ensures equal true positive rates across groups - "equal chance to be correctly identified as positive."

**Definition:**
$$P(\hat{Y}=1|Y=1, A=a_1) = P(\hat{Y}=1|Y=1, A=a_2)$$

This is TPR (True Positive Rate / Sensitivity / Recall) parity.

**Example - Age Group Opportunity:**
```
Young (≤50):   80% of actual disease cases caught
Older (>50):   80% of actual disease cases caught
Ratio: 1.0 (equal opportunity)
```

**Rationale:** Both groups should get equal treatment if they have disease.

**Implementation:**
```python
results = analyzer.analyze_equal_opportunity(
    y_pred=predictions,
    y_true=actual_disease,
    X_demographics={'Age_Group': age_groups}
)
```

### Calibration Analysis

Ensures predicted probabilities match actual outcomes within groups.

**Definition:** For predicted probability p and outcome Y:
$$P(Y=1|\hat{P}=p, A=a) \approx p \text{ for all } a$$

**Example:**
```
Predictions of "50% risk" should have ~50% actual disease rate
Predictions of "90% risk" should have ~90% actual disease rate
... across all demographic groups
```

**Methods:**
- Hosmer-Lemeshow test (statistical test)
- Calibration curves (visual inspection)
- ECE: Expected Calibration Error (metric)

**Implementation:**
```python
calib = analyzer.analyze_calibration(
    y_pred_proba=probabilities,
    y_true=actual_disease,
    X_demographics={'Gender': gender_values}
)
```

### Comprehensive Fairness Audit

Automated audit running all fairness checks:

```python
audit_results = analyzer.comprehensive_fairness_audit(
    y_pred=predictions,
    y_true=actual_disease,
    y_pred_proba=probabilities,
    X_demographics=X_test[['Gender', 'Age_Group']]
)

report = generate_fairness_report(audit_results)
print(report)
```

**Audit Output:**
- Demographic parity ratios per attribute
- Equal opportunity (TPR) disparities
- Calibration metrics per group
- Risk indicators
- Actionable recommendations

---

## 🔧 Ethics Principles

### Transparency

Users and doctors should understand how predictions are made.

**Implementation:**
1. **SHAP Explanations** - Feature importance for each prediction
2. **Model Card** - Document model design, training, limitations
3. **Data Sheet** - Document dataset composition, biases
4. **User Communication** - Clear explanation of risk factors

**Example Model Card:**
```markdown
# RandomForest Model Card

## Model Details
- Type: Random Forest Classifier
- Training Data: 50,000 synthetic records
- Features: 20 medical attributes
- AUC: 1.0 (test set)

## Performance by Subgroup
- Male: AUC=0.99
- Female: AUC=1.0
- Age ≤50: AUC=0.98
- Age >50: AUC=1.0

## Known Limitations
- Synthetic data (not real patients)
- No temporal validation
- Class imbalance not addressed
```

### Accountability

System owners are responsible for decisions and harms.

**Accountability Framework:**
1. **Responsibility Assignment** - Who owns fairness?
2. **Review Process** - Regular fairness audits
3. **Escalation Path** - How to report issues
4. **Remediation Process** - How to fix problems
5. **Affected Parties** - Notification if bias found

**Process:**
```
Fairness audit → Issues found?
  ↓ Yes
Severity assessment (critical/major/minor)
  ↓
Notify affected parties if bias impacted decisions
  ↓
Root cause analysis
  ↓
Remediation plan (retrain, adjust threshold, retire model)
  ↓
Retest for fairness
  ↓
Documentation and audit trail
```

### Beneficence

Use AI to improve patient outcomes and equity.

**Goals:**
- Improve early detection of heart disease
- Reduce false negatives (miss cases) especially in underrepresented groups
- Promote equitable healthcare access
- Reduce clinician bias through objective assessment

### Non-Maleficence

Do no harm - avoid negative impacts.

**Risks to Mitigate:**
1. **Discrimination** - Systematic bias against groups
   - Monitor fairness metrics continuously
   - Regular bias audits

2. **False Security** - Over-reliance on AI
   - Always require doctor review
   - Mark as "decision support only"

3. **Privacy Breach** - Data exposure
   - Encrypt sensitive data
   - Limit access
   - Audit logs

4. **Algorithmic Drift** - Performance degradation
   - Monitor on new data
   - Detect data drift
   - Retrain when needed

---

## 📋 Compliance Checklists

### GDPR (General Data Protection Regulation)

For EU patients or services offered in EU:

**Data Subject Rights:**
- [ ] Right to access - Patients can request their data
- [ ] Right to rectification - Patients can correct data
- [ ] Right to erasure - "Right to be forgotten"
- [ ] Right to restrict processing - Limit how data used
- [ ] Right to portability - Get data in standard format
- [ ] Right to object - Opt-out of processing
- [ ] Right to human review - For decisions affecting them

**Lawful Basis:**
- [ ] Obtain explicit consent OR
- [ ] Medical necessity (healthcare professional judgment) OR
- [ ] Legal obligation

**Technical Requirements:**
- [ ] Data minimization (only necessary data)
- [ ] Encryption (data at rest and in transit)
- [ ] Access controls (role-based)
- [ ] Audit logs (track access)
- [ ] Data retention limits (delete when no longer needed)

**Process Requirements:**
- [ ] Conduct DPIA (Data Protection Impact Assessment)
- [ ] Privacy policy published and easy to understand
- [ ] Breach notification within 72 hours
- [ ] Data Processing Agreement with processors
- [ ] Privacy by design in all systems

**Implementation Checklist:**
```python
GDPR_REQUIREMENTS = {
    'data_subject_rights': {
        'access': True,          # API to get user's data
        'rectification': True,   # API to correct data
        'erasure': True,         # Process to delete data
        'portability': True,     # Export in standard format
    },
    'technical': {
        'encryption': 'AES-256',
        'access_control': 'role_based',
        'audit_logs': True,
        'retention_limit': '3_years'
    }
}
```

### HIPAA (Health Insurance Portability & Accountability Act)

For US healthcare systems:

**Protected Health Information (PHI):**
- [ ] Access controls - Limit to authorized users
- [ ] Audit controls - Track all access
- [ ] Integrity controls - Ensure data not modified
- [ ] Transmission security - Encrypt in transit

**Administrative Safeguards:**
- [ ] Privacy policy (Notice of Privacy Practices)
- [ ] Workforce security program
- [ ] Risk assessment and management
- [ ] Breach notification plan
- [ ] Business associate agreements with vendors

**Physical Safeguards:**
- [ ] Facility access controls (locked servers)
- [ ] Workstation security
- [ ] Device and media controls (secure disposal)

**Technical Safeguards:**
- [ ] Encryption and decryption
- [ ] Unique user identification
- [ ] Emergency access procedures
- [ ] Audit controls (logging)
- [ ] Integrity controls

**Breach Notification:**
- [ ] Detect breach within 30 days
- [ ] Notify affected individuals
- [ ] Notify media (if 500+ affected)
- [ ] Notify HHS Office for Civil Rights
- [ ] Keep written documentation

**Implementation Checklist:**
```python
HIPAA_REQUIREMENTS = {
    'phi_protection': {
        'access_control': 'user_authentication',
        'audit_logs': True,
        'encryption': 'TLS_1.2_minimum'
    },
    'breach_response': {
        'detection_time': '30_days',
        'notification_time': '60_days',
        'documentation': True
    }
}
```

---

## 💻 Implementation Guide

### Running Privacy Analysis

```python
# Setup
from src.privacy import PrivacyManager, generate_privacy_report
import pandas as pd

df = pd.read_csv('data/synthetic_heart_disease_dataset.csv')

# Check K-Anonymity
manager = PrivacyManager(min_k_anonymity=5)
k_results = manager.check_k_anonymity(['Age', 'Gender'], df)

# Check Data Minimization
min_results = manager.data_minimization_check(df)

# Generate Report
privacy_report = generate_privacy_report(manager, df, 
                                         ['Age', 'Gender'])
print(privacy_report)
```

### Running Fairness Audit

```python
# Setup
from src.fairness import FairnessAnalyzer, generate_fairness_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train model and make predictions
model = train(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Run audit
analyzer = FairnessAnalyzer()
audit = analyzer.comprehensive_fairness_audit(
    y_pred=y_pred,
    y_true=y_test,
    y_pred_proba=y_pred_proba,
    X_demographics=X_test[['Gender', 'Age_Group']]
)

# Generate report
fairness_report = generate_fairness_report(audit)
print(fairness_report)
```

### Integration with Training Pipeline

Add privacy and fairness checks to `src/train.py`:

```python
from src.privacy import generate_privacy_report
from src.fairness import generate_fairness_report

def main():
    # ... existing training code ...
    
    # Privacy analysis
    privacy_report = generate_privacy_report(manager, df, 
                                             ['Age', 'Gender'])
    with open('reports/privacy_report.txt', 'w') as f:
        f.write(privacy_report)
    
    # Fairness audit
    fairness_audit = analyzer.comprehensive_fairness_audit(
        y_pred=y_pred, y_true=y_test,
        y_pred_proba=y_pred_proba,
        X_demographics=X_test[['Gender', 'Age_Group']]
    )
    fairness_report = generate_fairness_report(fairness_audit)
    with open('reports/fairness_report.txt', 'w') as f:
        f.write(fairness_report)
    
    print("Privacy and fairness analysis complete!")
```

---

## ⚠️ Risk Assessment

### Privacy Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Data Breach | Medium | High | Encryption, access control |
| Re-identification | Low | High | K-anonymity, data minimization |
| Unauthorized Access | Medium | High | Authentication, audit logs |
| Data Retention | Medium | Medium | Automatic deletion, retention limits |

### Fairness Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Gender Bias | High | High | Demographic parity monitoring |
| Age Bias | High | High | Equal opportunity testing |
| Disparate Impact | Medium | High | Regular fairness audits |
| Calibration Drift | Medium | Medium | Calibration monitoring, retraining |

### Security Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Model Poisoning | Low | High | Input validation, monitoring |
| Adversarial Attack | Low | Medium | Robustness testing |
| DoS Attack | Medium | Medium | Rate limiting, load balancing |
| Configuration Error | Medium | Medium | Infrastructure as code, testing |

---

## 📚 References

### Privacy & Compliance
- [GDPR Official Text](https://gdpr-info.eu/)
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/index.html)
- [CCPA - California Consumer Privacy Act](https://oag.ca.gov/privacy/ccpa)

### Fairness in ML
- [Fairness & Machine Learning](https://fairmlbook.org/)
- [Algorithmic Fairness](https://algorithmicfairness.org/)
- [Google's Fairness Definitions Explained](https://developers.google.com/machine-learning/crash-course/fairness/video-lecture)

### Healthcare AI Ethics
- [FDA Guidance on AI/ML in Medical Devices](https://www.fda.gov/medical-devices/software-medical-device-samd)
- [American Medical Association AI Ethics](https://www.ama-assn.org/publications/ama-journal-ethics/artificial-intelligence)
- [WHO Guidance on AI Ethics](https://www.who.int/publications/i/item/9789240029200)

---

## ✅ Verification & Testing

### Privacy Testing
```bash
# Run privacy analysis
python -c "from src.privacy import PrivacyManager; m = PrivacyManager(); print(m.check_k_anonymity(['Age'], df))"

# Check data minimization
python -c "from src.privacy import PrivacyManager; m = PrivacyManager(); print(m.data_minimization_check(df))"
```

### Fairness Testing
```bash
# Run fairness audit
python -c "from src.fairness import FairnessAnalyzer; a = FairnessAnalyzer(); print(a.comprehensive_fairness_audit(...))"

# Check demographic parity
python -c "from src.fairness import FairnessAnalyzer; a = FairnessAnalyzer(); print(a.analyze_demographic_parity(...))"
```

### Continuous Monitoring
```bash
# View logs
tail -f logs/app_*.log | grep "privacy\|fairness"

# Monitor drift
python -c "from src.monitor import ModelMonitor; m = ModelMonitor(); print(m.detect_drift(X_new))"
```

---

**Document Version:** 1.0.0  
**Last Updated:** November 16, 2025  
**Status:** Active  
**Review Frequency:** Quarterly
