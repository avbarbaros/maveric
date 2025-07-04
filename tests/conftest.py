"""Shared test fixtures and configuration."""

import pytest
import torch
import numpy as np


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    
@pytest.fixture
def device():
    """Get test device (CPU for tests)."""
    return 'cpu'