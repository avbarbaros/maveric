"""Interface definitions and data classes for MAVERIC."""

# Import dataclasses for creating structured data containers
from dataclasses import dataclass, field
# Import type hints for better code documentation and IDE support
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
# Import data processing libraries
import pandas as pd
import numpy as np
# Import datetime for timestamp handling 
from datetime import datetime


@dataclass
class RetrievalResult:
    """
    Data container for storing retrieval operation results.
    Stores retrieved samples with their computed scores and metadata.
    """
    
    # Core data - required fields for initialization
    samples: List[Dict[str, Any]]  # List of sample dictionaries with metadata and scores
    source_dataset: str  # Name of the source dataset used for retrieval
    target_dataset: str  # Name of the target dataset being matched against
    
    # Metadata - automatically generated fields
    retrieval_timestamp: datetime = field(default_factory=datetime.now)  # When retrieval was performed
    total_samples: int = field(init=False)  # Calculated automatically in __post_init__
    
    # Statistics - computed automatically
    class_distribution: Dict[str, int] = field(default_factory=dict)  # Count of samples per class
    score_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)  # Mean, std, min, max for each metric
    
    # Configuration used during retrieval
    config: Dict[str, Any] = field(default_factory=dict)  # Store retrieval configuration for reproducibility
    
    def __post_init__(self):
        """Calculate derived fields after dataclass initialization."""
        # Set total samples count based on actual samples list length
        self.total_samples = len(self.samples)
        # Calculate class distribution and score statistics
        self._calculate_statistics()
    
    def _calculate_statistics(self):
        """Calculate class distribution and score statistics from samples."""
        # Early return if no samples to process
        if not self.samples:
            return
            
        # Count samples per class label
        for sample in self.samples:
            # Get label from sample, default to 'unknown' if missing
            label = sample.get('label', 'unknown')
            # Increment count for this class, initialize to 0 if first occurrence
            self.class_distribution[label] = self.class_distribution.get(label, 0) + 1
        
        # Calculate statistical summaries for quality metrics
        df = pd.DataFrame(self.samples)  # Convert samples to DataFrame for easier analysis
        # Find columns containing quality scores or consistency metrics
        score_columns = [col for col in df.columns if 'score' in col or 'consistency' in col]
        
        # Calculate statistics for each score column
        for col in score_columns:
            if col in df.columns:
                # Remove NaN values before calculating statistics
                values = df[col].dropna()
                if len(values) > 0:
                    # Store comprehensive statistics for this metric
                    self.score_statistics[col] = {
                        'mean': float(values.mean()),    # Average score
                        'std': float(values.std()),      # Standard deviation
                        'min': float(values.min()),      # Minimum score
                        'max': float(values.max()),      # Maximum score
                        'median': float(values.median()) # Median score
                    }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert samples list to pandas DataFrame for analysis."""
        # Transform list of dictionaries into structured DataFrame
        return pd.DataFrame(self.samples)
    
    def save(self, path: str):
        """Serialize and save retrieval results to JSON file."""
        import json
        # Structure data for JSON serialization
        data = {
            'samples': self.samples,  # Raw sample data
            'metadata': {  # All metadata in nested structure
                'source_dataset': self.source_dataset,
                'target_dataset': self.target_dataset,
                'timestamp': self.retrieval_timestamp.isoformat(),  # Convert datetime to ISO string
                'total_samples': self.total_samples,
                'class_distribution': self.class_distribution,
                'score_statistics': self.score_statistics,
                'config': self.config
            }
        }
        # Write JSON with proper indentation for readability
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'RetrievalResult':
        """Class method to deserialize and load retrieval results from JSON file."""
        import json
        # Read JSON data from file
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Create new instance with core data (triggers __post_init__)
        result = cls(
            samples=data['samples'],
            source_dataset=data['metadata']['source_dataset'],
            target_dataset=data['metadata']['target_dataset']
        )
        
        # Manually restore metadata that wasn't set in constructor
        result.retrieval_timestamp = datetime.fromisoformat(data['metadata']['timestamp'])  # Parse ISO string back to datetime
        result.class_distribution = data['metadata']['class_distribution']
        result.score_statistics = data['metadata']['score_statistics']
        result.config = data['metadata']['config']
        
        return result
    
    @property
    def available_metrics(self) -> List[str]:
        """Property to get list of available quality metrics in the results."""
        # Return empty list if no samples
        if not self.samples:
            return []
        # Return all metrics that have computed statistics
        return list(self.score_statistics.keys())


@dataclass
class QualityResult:
    """
    Data container for quality control operation results.
    Stores filtered samples and comprehensive filtering statistics.
    """
    
    # Sample data - before and after filtering
    filtered_samples: List[Dict[str, Any]]  # Samples that passed quality thresholds
    original_samples: List[Dict[str, Any]]  # Original unfiltered samples for comparison
    
    # Configuration used for filtering
    thresholds: Dict[str, float]  # Quality thresholds applied during filtering
    weights: Dict[str, float] = field(default_factory=dict)  # Metric weights if used
    balance_strategy: str = "none"  # Dataset balancing strategy applied
    
    # Overall statistics - calculated automatically
    filtered_count: int = field(init=False)    # Number of samples after filtering
    original_count: int = field(init=False)    # Number of samples before filtering
    retention_rate: float = field(init=False)  # Percentage of samples retained
    
    # Detailed per-class statistics
    class_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Class-wise filtering stats
    
    def __post_init__(self):
        """Calculate filtering statistics after dataclass initialization."""
        # Count samples in each dataset
        self.filtered_count = len(self.filtered_samples)
        self.original_count = len(self.original_samples)
        # Calculate retention rate (avoid division by zero)
        self.retention_rate = self.filtered_count / self.original_count if self.original_count > 0 else 0.0
        # Calculate detailed per-class statistics
        self._calculate_class_statistics()
    
    def _calculate_class_statistics(self):
        """Calculate detailed per-class filtering statistics."""
        # Convert samples to DataFrames for easier analysis
        original_df = pd.DataFrame(self.original_samples)
        filtered_df = pd.DataFrame(self.filtered_samples)
        
        # Only calculate if we have class labels
        if 'label' in original_df.columns:
            # Iterate through each unique class in original data
            for class_name in original_df['label'].unique():
                # Get samples for this specific class
                original_class = original_df[original_df['label'] == class_name]
                # Handle case where no filtered samples exist
                filtered_class = filtered_df[filtered_df['label'] == class_name] if len(filtered_df) > 0 else pd.DataFrame()
                
                # Store comprehensive statistics for this class
                self.class_statistics[class_name] = {
                    'original_count': len(original_class),  # How many samples this class had originally
                    'filtered_count': len(filtered_class),  # How many samples passed filtering
                    'retention_rate': len(filtered_class) / len(original_class) if len(original_class) > 0 else 0.0  # Class-specific retention rate
                }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert filtered samples to pandas DataFrame for analysis."""
        # Transform filtered samples list into structured DataFrame
        return pd.DataFrame(self.filtered_samples)
    
    def get_summary(self) -> str:
        """Generate human-readable text summary of quality control results."""
        # Build summary with overall statistics
        summary = f"Quality Control Results:\n"
        summary += f"  Original samples: {self.original_count:,}\n"   # Format with thousands separator
        summary += f"  Filtered samples: {self.filtered_count:,}\n"   # Format with thousands separator
        summary += f"  Retention rate: {self.retention_rate:.1%}\n"   # Format as percentage
        summary += f"  Balance strategy: {self.balance_strategy}\n"
        
        # Add per-class breakdown if available
        if self.class_statistics:
            summary += f"\nPer-class statistics:\n"
            # Sort classes alphabetically for consistent output
            for class_name, stats in sorted(self.class_statistics.items()):
                summary += f"  {class_name}: {stats['filtered_count']}/{stats['original_count']} "
                summary += f"({stats['retention_rate']:.1%})\n"  # Show filtered/original (retention%)
        
        return summary


@dataclass
class CustomizationResult:
    """
    Data container for model customization/fine-tuning results.
    Stores trained model information and comprehensive evaluation metrics.
    """
    
    # Model identification
    model_name: str       # Name of the customized model
    base_model_name: str  # Name of the original base model
    
    # Training configuration and data
    training_config: Dict[str, Any]  # Training hyperparameters and settings
    training_samples: int            # Number of samples used for training
    
    # Performance metrics
    test_accuracy: float        # Accuracy on test set after training
    zero_shot_baseline: float   # Baseline accuracy without fine-tuning
    improvement: float = field(init=False)  # Calculated automatically as difference
    
    # Detailed evaluation metrics
    class_accuracies: Dict[str, float] = field(default_factory=dict)  # Per-class accuracy scores
    
    # Training progress tracking
    training_history: Dict[str, List[float]] = field(default_factory=dict)  # Loss/accuracy curves over epochs
    
    # Model persistence
    checkpoint_path: Optional[str] = None  # Path to saved model checkpoint
    
    def __post_init__(self):
        """Calculate derived performance metrics after initialization."""
        # Calculate improvement over baseline (can be positive or negative)
        self.improvement = self.test_accuracy - self.zero_shot_baseline
    
    def get_summary(self) -> str:
        """Generate human-readable summary of model customization results."""
        # Build summary with key performance metrics
        summary = f"Model Customization Results:\n"
        summary += f"  Base model: {self.base_model_name}\n"
        summary += f"  Training samples: {self.training_samples:,}\n"           # Format with thousands separator
        summary += f"  Zero-shot baseline: {self.zero_shot_baseline:.2f}%\n"    # 2 decimal places
        summary += f"  Test accuracy: {self.test_accuracy:.2f}%\n"            # 2 decimal places
        summary += f"  Improvement: {self.improvement:+.2f}%\n"              # Show + or - sign
        
        # Show top-performing classes if per-class data available
        if self.class_accuracies:
            summary += f"\nTop 5 classes:\n"
            # Sort classes by accuracy in descending order, take top 5
            sorted_classes = sorted(self.class_accuracies.items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
            for class_name, acc in sorted_classes:
                summary += f"  {class_name}: {acc:.1f}%\n"  # 1 decimal place for class accuracies
        
        return summary


class ProgressCallback:
    """
    Callback interface for tracking progress of long-running operations.
    Allows external code to monitor and display progress of retrieval, quality control, etc.
    """
    
    def __init__(self, 
                 on_update: Callable[[int, int, str], None] = None,
                 on_complete: Callable[[], None] = None):
        """
        Initialize progress callback with optional callback functions.
        
        Args:
            on_update: Function called on progress updates (current, total, message)
            on_complete: Function called when operation completes
        """
        # Use provided callback or default to no-op lambda functions
        self.on_update = on_update or (lambda c, t, m: None)      # Do nothing if no callback provided
        self.on_complete = on_complete or (lambda: None)          # Do nothing if no callback provided
        
    def update(self, current: int, total: int, message: str = ""):
        """Report progress update to registered callback function."""
        # Call the registered update callback with current progress
        self.on_update(current, total, message)
        
    def complete(self):
        """Signal that the operation has completed successfully."""
        # Call the registered completion callback
        self.on_complete()