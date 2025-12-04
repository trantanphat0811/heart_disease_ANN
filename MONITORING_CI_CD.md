# Monitoring, Logging & CI/CD Guide

This document describes the comprehensive monitoring, logging, and CI/CD system set up for the Heart Disease ML project.

## 📋 Table of Contents

1. [Logging System](#logging-system)
2. [Model Monitoring](#model-monitoring)
3. [Unit Testing](#unit-testing)
4. [CI/CD Pipeline](#cicd-pipeline)
5. [Docker & Deployment](#docker--deployment)
6. [Makefile Commands](#makefile-commands)

---

## Logging System

### Overview

The logging system captures all application events in a structured JSON format for easy parsing and analysis.

### Configuration

Location: `src/logger.py`

**Features:**
- Dual output: Console (readable) + File (JSON structured)
- Separate error logs for quick debugging
- Automatic timestamp tracking
- JSON formatting for log aggregation tools

### Log Files

Logs are stored in the `logs/` directory with timestamps:

```
logs/
├── app_20231116_140530.log      # Main application log (JSON)
└── errors_20231116_140530.log   # Error log (plain text)
```

### Usage

```python
from src.logger import logger, log_training_start, log_prediction

# Log training start
log_training_start({
    "test_size": 0.2,
    "models": ["logistic_regression", "random_forest"]
})

# Log predictions
log_prediction(
    input_data=patient_data,
    prediction={"risk_score": 0.85, "label": 1},
    model_version="v1.0"
)
```

### Viewing Logs

```bash
# View recent logs
tail -f logs/app_*.log

# View errors only
tail -f logs/errors_*.log

# Parse JSON logs
cat logs/app_*.log | jq '.message' | grep "Training"
```

---

## Model Monitoring

### Overview

Monitors model performance and detects data drift automatically.

### Features

**Data Drift Detection:**
- Tracks feature statistics (mean, std, min, max)
- Compares new data against baseline using KS test
- Alerts on statistical divergence

**Prediction Quality:**
- Monitors prediction distribution
- Tracks confidence scores
- Identifies low-confidence predictions

### Usage

```python
from src.monitor import ModelMonitor, check_prediction_quality
import pandas as pd

# Initialize monitor with baseline
monitor = ModelMonitor("models/baseline_stats.json")

# Save baseline from training data
monitor.save_baseline(X_train)

# Detect drift in new data
drift_report = monitor.detect_drift(X_new, threshold=0.05)
if drift_report["status"] == "drift_detected":
    print(f"⚠️  Drift detected in features: {drift_report['drifts']}")

# Check prediction quality
quality_metrics = check_prediction_quality(predictions, probabilities)
```

### Configuration

- **Drift threshold**: 0.05 (5% significance level)
- **Confidence threshold**: 0.6 (60% minimum confidence)
- **Baseline path**: `models/baseline_stats.json`

---

## Unit Testing

### Overview

Comprehensive test suite covering data loading, model training, and evaluation.

### Test Structure

```
tests/
└── test_pipeline.py
    ├── TestDataLoading
    ├── TestModelTraining
    └── TestModelEvaluation
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run specific test class
pytest tests/test_pipeline.py::TestModelTraining -v

# Run with markers
pytest tests/ -v -m "not slow"
```

### Test Coverage

- **Data Loading**: Shape, columns, missing values
- **Model Training**: Pipeline creation, prediction shapes, probability ranges
- **Model Evaluation**: Metric computation, value ranges, NaN checks

### Expected Coverage

- Minimum coverage threshold: 80%
- Coverage report: `htmlcov/index.html`

---

## CI/CD Pipeline

### Overview

Automated testing and deployment using GitHub Actions.

### Workflow: `.github/workflows/ci-cd.yml`

#### Trigger Events

- Push to `main` or `develop` branches
- Pull requests to `main`
- Daily scheduled run at 2 AM UTC

#### Jobs

**1. Test Job**
- Python 3.9, 3.10, 3.11 compatibility
- Lint with flake8
- Run pytest with coverage
- Train full model
- Upload to Codecov

**2. Code Quality Job**
- Format checking with black
- Import sorting with isort
- Complexity analysis

**3. Security Job**
- Bandit security scanning
- Dependency vulnerability checking (safety)

**4. Build Artifacts Job**
- Generate EDA reports
- Train final model
- Generate HTML report
- Upload artifacts (30-day retention)

**5. Docker Job**
- Build Docker image
- Test image runs correctly

**6. Deploy Job**
- Runs only on main branch push
- Requires all other jobs to pass
- Outputs deployment readiness status

### Viewing CI/CD Results

GitHub Actions UI: `Settings → Actions`

### Branch Protection

Recommended settings in GitHub:
- Require status checks to pass
- Require code reviews before merge
- Dismiss stale PR approvals

---

## Docker & Deployment

### Docker Setup

#### Build Image

```bash
make docker-build
# or
docker build -t heart-disease-api:latest .
```

#### Run Container

```bash
make docker-up
# or
docker-compose up -d
```

#### Available Services

- **API**: `http://localhost:8000`
- **Dashboard**: `http://localhost:8501`
- **Web Report**: `http://localhost:3000`

### Docker Compose Services

```yaml
api:
  - FastAPI prediction service
  - Port 8000
  - Health check enabled

dashboard:
  - Streamlit analytics dashboard
  - Port 8501

web:
  - HTTP server for HTML reports
  - Port 3000
```

### Environment Variables

```bash
LOG_LEVEL=INFO              # Logging level
PYTHONUNBUFFERED=1         # Python output buffering
MODEL_PATH=models/best_model.joblib
DATA_PATH=data/synthetic_heart_disease_dataset.csv
```

### Health Checks

All services have health checks configured:

```bash
# API health
curl http://localhost:8000/health

# Manual container health
docker ps | grep heart-disease-api
```

---

## Makefile Commands

### Quick Reference

```bash
# Setup
make venv           # Create virtual environment
make install        # Install dependencies

# Development
make train          # Train ML model
make eda            # Run EDA
make report         # Generate HTML report
make test           # Run tests
make test-cov       # Tests with coverage
make lint           # Code linting
make format         # Format code

# Services
make api            # Start FastAPI
make dashboard      # Start Streamlit
make web            # Start web server
make all-services   # Start all

# Docker
make docker-build   # Build image
make docker-up      # Start containers
make docker-down    # Stop containers
make docker-logs    # View logs

# Monitoring
make logs           # View app logs
make monitor        # Start monitoring

# Cleanup
make clean          # Remove build artifacts
make clean-docker   # Remove containers
```

---

## Deployment Checklist

- [ ] All tests pass locally (`make test`)
- [ ] Code formatted (`make format`)
- [ ] No linting issues (`make lint`)
- [ ] Docker image builds (`make docker-build`)
- [ ] Push to main branch
- [ ] GitHub Actions CI/CD passes
- [ ] Artifacts generated and uploaded
- [ ] Deploy to production environment

---

## Troubleshooting

### Tests Failing

```bash
# Run with verbose output
pytest tests/ -vv

# Run specific test
pytest tests/test_pipeline.py::TestDataLoading::test_load_data_shape -vv

# Check dependencies
pip list | grep -E "scikit|pandas|joblib"
```

### Docker Issues

```bash
# View logs
make docker-logs

# Rebuild without cache
docker build --no-cache -t heart-disease-api:latest .

# Remove all containers
docker-compose down -v
```

### CI/CD Failures

1. Check GitHub Actions logs
2. Run same commands locally
3. Verify Python version compatibility
4. Check dependency versions in requirements.txt

### Performance Issues

```bash
# Monitor container resources
docker stats

# Check model size
ls -lh models/best_model.joblib

# Profile code
python -m cProfile -s cumtime src/train.py
```

---

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Documentation](https://docs.docker.com/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Python Logging](https://docs.python.org/3/library/logging.html)

---

## Questions?

For issues or improvements, please create a GitHub issue or contact the development team.

Last updated: 16 Nov 2025
