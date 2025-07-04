"""Tests for configuration module."""

import pytest
import tempfile
from pathlib import Path

from maveric.config import MAVERICConfig, TrainingConfig


class TestMAVERICConfig:
    """Test MAVERICConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MAVERICConfig()
        
        assert config.clip_model == "ViT-B/32"
        assert config.batch_size == 32
        assert config.enable_image_cache is True
        
    def test_config_validation(self):
        """Test configuration validation."""
        config = MAVERICConfig()
        warnings = config.validate()
        
        # Should have minimal warnings with default config
        assert isinstance(warnings, list)
        
    def test_save_load_yaml(self):
        """Test saving and loading YAML config."""
        config = MAVERICConfig(
            clip_model="ViT-B/16",
            batch_size=64
        )
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            loaded_config = MAVERICConfig.from_yaml(f.name)
            
        assert loaded_config.clip_model == "ViT-B/16"
        assert loaded_config.batch_size == 64
        
        Path(f.name).unlink()
    
    def test_save_load_json(self):
        """Test saving and loading JSON config."""
        config = MAVERICConfig(
            cache_base_dir="/test/cache",
            device="cpu"
        )
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config.to_json(f.name)
            loaded_config = MAVERICConfig.from_json(f.name)
            
        assert loaded_config.cache_base_dir == "/test/cache"
        assert loaded_config.device == "cpu"
        
        Path(f.name).unlink()
    
    def test_config_update(self):
        """Test updating configuration values."""
        config = MAVERICConfig()
        
        config.update(batch_size=128, device="cuda")
        assert config.batch_size == 128
        assert config.device == "cuda"
        
        # Test invalid key
        with pytest.raises(ValueError):
            config.update(invalid_key="value")


class TestTrainingConfig:
    """Test TrainingConfig class."""
    
    def test_default_training_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        
        assert config.epochs == 10
        assert config.learning_rate == 1e-5
        assert config.optimizer == "adamw"
        
    def test_training_config_validation(self):
        """Test training configuration validation."""
        # Valid config
        config = TrainingConfig(epochs=5, learning_rate=1e-4)
        warnings = config.validate()
        assert len(warnings) == 0
        
        # Invalid config
        invalid_config = TrainingConfig(epochs=0, learning_rate=-1)
        warnings = invalid_config.validate()
        assert len(warnings) > 0
