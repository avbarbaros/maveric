"""Interface definitions and data classes for MAVERIC."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class RetrievalResult:
    """
    Result container for retrieval operations.
    Stores retrieved samples with their computed scores and metadata.
    """
    
    # Core data
    samples: List[Dict[str, Any]]
    source_dataset: str
    target_dataset: str
    
    # Metadata
    retrieval_timestamp: datetime = field(default_factory=datetime.now)
    total_samples: int = field(init=False)
    
    # Statistics
    class_distribution: Dict[str, int] = field(default_factory=dict)
    score_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.total_samples = len(self.samples)
        self._calculate_statistics()
    
    def _calculate_statistics(self):
        """Calculate class distribution and score statistics."""
        if not self.samples:
            return
            
        # Calculate class distribution
        for sample in self.samples:
            label = sample.get('label', 'unknown')
            self.class_distribution[label] = self.class_distribution.get(label, 0) + 1
        
        # Calculate score statistics
        df = pd.DataFrame(self.samples)
        score_columns = [col for col in df.columns if 'score' in col or 'consistency' in col]
        
        for col in score_columns:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    self.score_statistics[col] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median())
                    }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert samples to pandas DataFrame."""
        return pd.DataFrame(self.samples)
    
    def save(self, path: str):
        """Save retrieval results to file."""
        import json
        data = {
            'samples': self.samples,
            'metadata': {
                'source_dataset': self.source_dataset,
                'target_dataset': self.target_dataset,
                'timestamp': self.retrieval_timestamp.isoformat(),
                'total_samples': self.total_samples,
                'class_distribution': self.class_distribution,
                'score_statistics': self.score_statistics,
                'config': self.config
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'RetrievalResult':
        """Load retrieval results from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        result = cls(
            samples=data['samples'],
            source_dataset=data['metadata']['source_dataset'],
            target_dataset=data['metadata']['target_dataset']
        )
        
        # Restore metadata
        result.retrieval_timestamp = datetime.fromisoformat(data['metadata']['timestamp'])
        result.class_distribution = data['metadata']['class_distribution']
        result.score_statistics = data['metadata']['score_statistics']
        result.config = data['metadata']['config']
        
        return result
    
    @property
    def available_metrics(self) -> List[str]:
        """Get list of available quality metrics in the results."""
        if not self.samples:
            return []
        return list(self.score_statistics.keys())


@dataclass
class QualityResult:
    """
    Result container for quality control operations.
    Stores filtered samples and filtering statistics.
    """
    
    # Filtered data
    filtered_samples: List[Dict[str, Any]]
    original_samples: List[Dict[str, Any]]
    
    # Filtering configuration
    thresholds: Dict[str, float]
    weights: Dict[str, float] = field(default_factory=dict)
    balance_strategy: str = "none"
    
    # Statistics
    filtered_count: int = field(init=False)
    original_count: int = field(init=False)
    retention_rate: float = field(init=False)
    
    # Per-class statistics
    class_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate statistics after initialization."""
        self.filtered_count = len(self.filtered_samples)
        self.original_count = len(self.original_samples)
        self.retention_rate = self.filtered_count / self.original_count if self.original_count > 0 else 0.0
        self._calculate_class_statistics()
    
    def _calculate_class_statistics(self):
        """Calculate per-class filtering statistics."""
        original_df = pd.DataFrame(self.original_samples)
        filtered_df = pd.DataFrame(self.filtered_samples)
        
        if 'label' in original_df.columns:
            for class_name in original_df['label'].unique():
                original_class = original_df[original_df['label'] == class_name]
                filtered_class = filtered_df[filtered_df['label'] == class_name] if len(filtered_df) > 0 else pd.DataFrame()
                
                self.class_statistics[class_name] = {
                    'original_count': len(original_class),
                    'filtered_count': len(filtered_class),
                    'retention_rate': len(filtered_class) / len(original_class) if len(original_class) > 0 else 0.0
                }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert filtered samples to pandas DataFrame."""
        return pd.DataFrame(self.filtered_samples)
    
    def get_summary(self) -> str:
        """Get a text summary of the quality control results."""
        summary = f"Quality Control Results:\n"
        summary += f"  Original samples: {self.original_count:,}\n"
        summary += f"  Filtered samples: {self.filtered_count:,}\n"
        summary += f"  Retention rate: {self.retention_rate:.1%}\n"
        summary += f"  Balance strategy: {self.balance_strategy}\n"
        
        if self.class_statistics:
            summary += f"\nPer-class statistics:\n"
            for class_name, stats in sorted(self.class_statistics.items()):
                summary += f"  {class_name}: {stats['filtered_count']}/{stats['original_count']} "
                summary += f"({stats['retention_rate']:.1%})\n"
        
        return summary


@dataclass
class CustomizationResult:
    """
    Result container for model customization.
    Stores trained model information and evaluation metrics.
    """
    
    # Model information
    model_name: str
    base_model_name: str
    
    # Training configuration
    training_config: Dict[str, Any]
    training_samples: int
    
    # Evaluation metrics
    test_accuracy: float
    zero_shot_baseline: float
    improvement: float = field(init=False)
    
    # Per-class metrics
    class_accuracies: Dict[str, float] = field(default_factory=dict)
    
    # Training history
    training_history: Dict[str, List[float]] = field(default_factory=dict)
    
    # Model checkpoint path
    checkpoint_path: Optional[str] = None
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.improvement = self.test_accuracy - self.zero_shot_baseline
    
    def get_summary(self) -> str:
        """Get a text summary of customization results."""
        summary = f"Model Customization Results:\n"
        summary += f"  Base model: {self.base_model_name}\n"
        summary += f"  Training samples: {self.training_samples:,}\n"
        summary += f"  Zero-shot baseline: {self.zero_shot_baseline:.2f}%\n"
        summary += f"  Test accuracy: {self.test_accuracy:.2f}%\n"
        summary += f"  Improvement: {self.improvement:+.2f}%\n"
        
        if self.class_accuracies:
            summary += f"\nTop 5 classes:\n"
            sorted_classes = sorted(self.class_accuracies.items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
            for class_name, acc in sorted_classes:
                summary += f"  {class_name}: {acc:.1f}%\n"
        
        return summary


class ProgressCallback:
    """
    Interface for progress tracking callbacks.
    Allows external code to monitor long-running operations.
    """
    
    def __init__(self, 
                 on_update: Callable[[int, int, str], None] = None,
                 on_complete: Callable[[], None] = None):
        """
        Initialize progress callback.
        
        Args:
            on_update: Function called on progress updates (current, total, message)
            on_complete: Function called when operation completes
        """
        self.on_update = on_update or (lambda c, t, m: None)
        self.on_complete = on_complete or (lambda: None)
        
    def update(self, current: int, total: int, message: str = ""):
        """Update progress."""
        self.on_update(current, total, message)
        
    def complete(self):
        """Mark operation as complete."""
        self.on_complete()