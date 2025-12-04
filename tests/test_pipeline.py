"""Unit tests for the machine learning pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train import load_data, build_and_train, evaluate


@pytest.fixture
def sample_data():
    """Create sample test data."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Age': np.random.randint(30, 80, n_samples),
        'Gender': np.random.choice(['M', 'F'], n_samples),
        'Weight': np.random.randint(50, 120, n_samples),
        'Height': np.random.uniform(1.5, 2.0, n_samples),
        'BMI': np.random.uniform(18, 35, n_samples),
        'Smoking': np.random.choice([0, 1], n_samples),
        'Alcohol_Intake': np.random.choice([0, 1, 2, 3], n_samples),
        'Physical_Activity': np.random.choice([0, 1, 2, 3, 4, 5], n_samples),
        'Diet': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'Stress_Level': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'Hypertension': np.random.choice([0, 1], n_samples),
        'Diabetes': np.random.choice([0, 1], n_samples),
        'Hyperlipidemia': np.random.choice([0, 1], n_samples),
        'Family_History': np.random.choice([0, 1], n_samples),
        'Previous_Heart_Attack': np.random.choice([0, 1], n_samples),
        'Systolic_BP': np.random.randint(100, 180, n_samples),
        'Diastolic_BP': np.random.randint(60, 120, n_samples),
        'Heart_Rate': np.random.randint(50, 100, n_samples),
        'Blood_Sugar_Fasting': np.random.randint(70, 150, n_samples),
        'Cholesterol_Total': np.random.randint(100, 300, n_samples),
        'Heart_Disease': np.random.choice([0, 1], n_samples)
    }
    
    return pd.DataFrame(data)


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_data_shape(self, sample_data, tmp_path):
        """Test that loaded data has correct shape."""
        # Save sample data
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        
        # Load data
        df = load_data(str(csv_path))
        
        assert df.shape == sample_data.shape
        assert list(df.columns) == list(sample_data.columns)
    
    def test_load_data_no_missing_values(self, sample_data, tmp_path):
        """Test that loaded data handles missing values."""
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        
        df = load_data(str(csv_path))
        
        # After preprocessing, shouldn't have missing values
        assert df.isnull().sum().sum() == 0


class TestModelTraining:
    """Test model training pipeline."""
    
    def test_build_and_train_creates_pipeline(self, sample_data):
        """Test that training creates a valid pipeline."""
        X = sample_data.drop('Heart_Disease', axis=1)
        y = sample_data['Heart_Disease']
        
        model = build_and_train(X, y, model_name="logreg")
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_model_prediction_shape(self, sample_data):
        """Test that model predictions have correct shape."""
        X = sample_data.drop('Heart_Disease', axis=1)
        y = sample_data['Heart_Disease']
        
        model = build_and_train(X, y, model_name="logreg")
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert predictions.shape == (len(X),)
        assert probabilities.shape == (len(X), 2)
        assert np.all((predictions == 0) | (predictions == 1))
        assert np.all((probabilities >= 0) & (probabilities <= 1))


class TestModelEvaluation:
    """Test model evaluation metrics."""
    
    def test_evaluate_returns_metrics(self, sample_data):
        """Test that evaluation returns expected metrics."""
        X = sample_data.drop('Heart_Disease', axis=1)
        y = sample_data['Heart_Disease']
        
        model = build_and_train(X, y, model_name="logreg")
        metrics = evaluate(model, X, y)
        
        assert 'auc' in metrics
        assert 'accuracy' in metrics
        assert isinstance(metrics['auc'], float)
        assert isinstance(metrics['accuracy'], float)
        assert 0 <= metrics['auc'] <= 1
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_evaluate_metrics_reasonable(self, sample_data):
        """Test that metrics are in reasonable ranges."""
        X = sample_data.drop('Heart_Disease', axis=1)
        y = sample_data['Heart_Disease']
        
        model = build_and_train(X, y, model_name="logreg")
        metrics = evaluate(model, X, y)
        
        # Metrics should be reasonable (not NaN, not negative)
        assert not np.isnan(metrics['auc'])
        assert not np.isnan(metrics['accuracy'])
        assert metrics['auc'] >= 0
        assert metrics['accuracy'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
