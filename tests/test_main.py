"""Tests for main MAVERIC class."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from maveric import MAVERIC, MAVERICConfig
from maveric.core.interfaces import RetrievalResult, QualityResult


class TestMAVERIC:
    """Test main MAVERIC class."""
    
    @pytest.fixture
    def maveric(self, tmp_path):
        """Create MAVERIC instance with temp cache."""
        config = MAVERICConfig(
            cache_base_dir=str(tmp_path),
            device="cpu"
        )
        
        with patch('maveric.main.Retriever'), \
             patch('maveric.main.CacheManager'):
            return MAVERIC(config)
    
    def test_initialization(self, maveric):
        """Test MAVERIC initialization."""
        assert maveric.config is not None
        assert maveric.cache_manager is not None
        assert maveric.retriever is not None
    
    @patch('maveric.main.REACTDatasetHandler')
    def test_retrieve(self, mock_handler, maveric):
        """Test retrieval functionality."""
        # Mock cache manager to return None (no cached results)
        maveric.cache_manager.load_results = Mock(return_value=None)
        
        # Mock dataset handler
        mock_handler.return_value = Mock()
        
        # Mock retriever
        maveric.retriever.retrieve = Mock(
            return_value=RetrievalResult(
                samples=[{'id': 1, 'score': 0.9}],
                source_dataset='react-test-dataset',
                target_dataset='cifar10'
            )
        )
        
        result = maveric.retrieve(
            dataset_name='react-test-dataset',
            target_dataset='cifar10',
            num_samples=100
        )
        
        assert isinstance(result, RetrievalResult)
        assert result.total_samples == 1
    
    def test_quality_control(self, maveric):
        """Test quality control functionality."""
        # Create test data
        test_data = pd.DataFrame({
            'id': range(100),
            'label': ['cat'] * 50 + ['dog'] * 50,
            'weighted_class_score': np.random.uniform(0.4, 0.9, 100),
            'consistency': np.random.uniform(0.5, 1.0, 100)
        })
        
        # Mock quality controller
        with patch('maveric.main.QualityController') as mock_qc:
            mock_instance = Mock()
            mock_instance.apply_thresholds.return_value = 80
            mock_instance.create_quality_result.return_value = QualityResult(
                filtered_samples=test_data.iloc[:80].to_dict('records'),
                original_samples=test_data.to_dict('records'),
                thresholds={'weighted_class_score': 0.5},
                balance_strategy='none'
            )
            mock_qc.return_value = mock_instance
            
            result = maveric.quality_control(
                test_data,
                thresholds={'weighted_class_score': 0.5}
            )
            
            assert isinstance(result, QualityResult)
            assert result.filtered_count == 80
    
    def test_save_load_config(self, maveric, tmp_path):
        """Test saving and loading configuration."""
        config_path = tmp_path / 'test_config.yaml'
        
        maveric.save_config(str(config_path))
        assert config_path.exists()
        
        # Load from config
        loaded_maveric = MAVERIC.from_config_file(str(config_path))
        assert loaded_maveric.config.clip_model == maveric.config.clip_model
