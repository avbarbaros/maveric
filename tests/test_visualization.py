"""Tests for visualization module."""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, Mock
from PIL import Image

from maveric.visualization import MetricsVisualizer, SampleVisualizer


class TestMetricsVisualizer:
    """Test MetricsVisualizer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for visualization."""
        np.random.seed(42)
        return pd.DataFrame({
            'sharpness_score': np.random.normal(0.8, 0.1, 1000),
            'consistency': np.random.normal(0.7, 0.15, 1000),
            'resolution_score': np.random.normal(0.6, 0.2, 1000)
        })
    
    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance."""
        return MetricsVisualizer()
    
    def test_plot_metric_distribution(self, visualizer, sample_data):
        """Test plotting single metric distribution."""
        fig = visualizer.plot_metric_distribution(
            sample_data,
            'sharpness_score',
            threshold=0.8
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        # Check that threshold line exists
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert any(line.get_color() == 'red' for line in lines)
        
        plt.close(fig)
    
    def test_plot_multi_metric_distributions(self, visualizer, sample_data):
        """Test plotting multiple metric distributions."""
        metrics = ['sharpness_score', 'consistency', 'resolution_score']
        thresholds = {
            'sharpness_score': 0.8,
            'consistency': 0.7
        }
        
        fig = visualizer.plot_multi_metric_distributions(
            sample_data,
            metrics,
            thresholds,
            ncols=2
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= len(metrics)
        
        plt.close(fig)
    
    def test_plot_metric_comparison(self, visualizer, sample_data):
        """Test plotting metric comparison."""
        # Create filtered data
        filtered_data = sample_data[sample_data['sharpness_score'] > 0.8]
        
        fig = visualizer.plot_metric_comparison(
            sample_data,
            filtered_data,
            ['sharpness_score', 'consistency']
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestSampleVisualizer:
    """Test SampleVisualizer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with image URLs."""
        return pd.DataFrame({
            'id': range(10),
            'url': [f'http://example.com/img{i}.jpg' for i in range(10)],
            'label': ['cat'] * 5 + ['dog'] * 5,
            'weighted_class_score': np.random.uniform(0.5, 0.9, 10),
            'sharpness_score': np.random.uniform(0.6, 1.0, 10)
        })
    
    @pytest.fixture
    def visualizer(self):
        """Create sample visualizer."""
        return SampleVisualizer()
    
    @patch('maveric.visualization.samples.requests.get')
    def test_visualize_samples(self, mock_get, visualizer, sample_data):
        """Test sample visualization."""
        # Mock image loading
        mock_response = Mock()
        mock_response.content = b'fake_image_data'
        mock_get.return_value = mock_response
        
        with patch('PIL.Image.open') as mock_open:
            mock_open.return_value = Image.new('RGB', (100, 100))
            
            fig = visualizer.visualize_samples(
                sample_data,
                n_samples=3,
                sample_type='random'
            )
            
            assert isinstance(fig, plt.Figure)
            assert len(fig.axes) == 3
            
            plt.close(fig)
