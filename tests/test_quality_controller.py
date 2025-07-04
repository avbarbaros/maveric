"""Tests for quality controller."""

import pytest
import pandas as pd
import numpy as np

from maveric.quality import QualityController
from maveric.quality.filters import ThresholdFilter, BalancedFilter


class TestQualityController:
    """Test QualityController class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'id': range(n_samples),
            'url': [f'http://example.com/{i}.jpg' for i in range(n_samples)],
            'text': [f'Sample text {i}' for i in range(n_samples)],
            'label': np.random.choice(['cat', 'dog', 'bird'], n_samples),
            'weighted_class_score': np.random.uniform(0.3, 0.9, n_samples),
            'consistency': np.random.uniform(0.5, 1.0, n_samples),
            'sharpness_score': np.random.uniform(0.6, 1.0, n_samples),
            'resolution_score': np.random.uniform(0.2, 1.5, n_samples),
            'color_score': np.random.uniform(0.4, 1.0, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def test_load_data(self, sample_data):
        """Test loading data into quality controller."""
        qc = QualityController()
        qc.load_data(sample_data)
        
        assert len(qc.data) == len(sample_data)
        assert 'label' in qc.data.columns
    
    def test_set_threshold(self, sample_data):
        """Test setting thresholds."""
        qc = QualityController(sample_data)
        
        qc.set_threshold('sharpness_score', 0.8)
        assert qc.thresholds['sharpness_score'] == 0.8
    
    def test_apply_thresholds(self, sample_data):
        """Test applying thresholds."""
        qc = QualityController(sample_data)
        
        # Set strict thresholds
        qc.set_threshold('sharpness_score', 0.9)
        qc.set_threshold('consistency', 0.8)
        
        initial_count = len(qc.data)
        filtered_count = qc.apply_thresholds()
        
        assert filtered_count < initial_count
        assert len(qc.filtered_data) == filtered_count
    
    def test_balance_dataset(self, sample_data):
        """Test dataset balancing."""
        qc = QualityController(sample_data)
        
        # Apply some filtering first
        qc.set_threshold('weighted_class_score', 0.5)
        qc.apply_thresholds()
        
        # Balance dataset
        balanced_df = qc.balance_dataset(
            strategy='min',
            min_samples=10
        )
        
        # Check that all classes have same number of samples
        class_counts = balanced_df['label'].value_counts()
        assert len(set(class_counts.values)) == 1
    
    def test_get_statistics(self, sample_data):
        """Test getting statistics."""
        qc = QualityController(sample_data)
        qc.apply_thresholds()
        
        stats = qc.get_statistics()
        
        assert 'total_samples' in stats
        assert 'filtered_samples' in stats
        assert 'retention_rate' in stats
        assert stats['total_samples'] == len(sample_data)


class TestFilters:
    """Test filter classes."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'score1': [0.1, 0.5, 0.8, 0.9, 0.3],
            'score2': [0.7, 0.8, 0.9, 0.6, 0.5],
            'label': ['A', 'B', 'A', 'B', 'A']
        })
    
    def test_threshold_filter(self, sample_df):
        """Test threshold filter."""
        filter = ThresholdFilter({'score1': 0.5, 'score2': 0.7})
        filtered = filter.apply(sample_df)
        
        assert len(filtered) < len(sample_df)
        assert all(filtered['score1'] >= 0.5)
        assert all(filtered['score2'] >= 0.7)
    
    def test_balanced_filter(self, sample_df):
        """Test balanced filter."""
        filter = BalancedFilter(
            strategy='min',
            min_threshold=1,
            sort_by='score1'
        )
        
        balanced = filter.apply(sample_df)
        
        # Check balance
        class_counts = balanced['label'].value_counts()
        assert len(set(class_counts.values)) == 1
