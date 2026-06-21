"""Configuration management for MAVERIC."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union
import yaml
import json
from pathlib import Path


@dataclass
class MAVERICConfig:
    """
    Main configuration for MAVERIC system.
    
    This configuration controls all aspects of MAVERIC's behavior, from model selection
    to quality thresholds. It can be saved and loaded from YAML files for reproducibility.
    """
    
    # Model configuration
    clip_model: str = "ViT-B/32"
    device: str = "auto"  # auto, cuda, cpu
    
    # Cache configuration
    cache_base_dir: str = "./cache"
    enable_image_cache: bool = True
    cache_format: str = "jpg"
    cache_quality: int = 95  # JPEG quality for cached images
    
    # Retrieval configuration
    batch_size: int = 32
    num_workers: int = 0
    n_reference_images: int = 10
    retrieval_rotation_size: int = 1000
    max_retries: int = 3
    request_timeout: int = 5

    # Scoring mode configuration
    scoring_mode: str = "clip"  # "clip" (default, CLIP-based multi-modal) or "hu_moments" (shape-based) 

    # Quality metrics configuration - organized by category
    quality_metrics: List[str] = field(default_factory=lambda: [
        # Visual metrics (image-only)
        'resolution', 'sharpness', 'color_diversity',
        # Semantic metrics (text-only) 
        'text_quality', 'caption_length',
        # Multimodal metrics (cross-modal)
        'target_class_quality', 'multimodal_consistency'
    ])
    
    # Metric weights for composite scoring  
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        'img2img': 0.25,
        'txt2txt': 0.25,
        'img2txt': 0.25,
        'txt2img': 0.25
    })
    
    # Default quality thresholds - organized by metric category
    default_thresholds: Dict[str, float] = field(default_factory=lambda: {
        # Visual metrics
        'resolution_score': 0.370,
        'sharpness_score': 0.880,
        'color_score': 0.768,
        # Semantic metrics
        'text_quality_score': 0.600,      # Text quality threshold
        'caption_length_score': 0.700,    # Caption length threshold
        # Multimodal metrics
        'target_class_quality': 0.493,    # Target class quality using CLIP mappings
        'consistency': 0.796,             # Multimodal consistency
        'weighted_class_score': 0.493     # Weighted similarity score across modalities
    })
    
    # Dataset balancing configuration
    balance_strategy: str = "min"  # Balancing strategy: median, mean, min, max, none
    balance_min_samples: int = 15
    balance_enable_oversampling: bool = False
    
    # Logging configuration
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: Optional[str] = None
    
    # Progress display configuration
    enable_real_time_stats: bool = True

    # Reproducibility configuration
    seed: int = 42  # Random seed for reproducible sampling

    # Performance optimization configuration
    enable_target_class_quality: bool = False  # Enable/disable EfficientNet-based TargetClassQualityMetric (time-consuming, ~50-70% overhead)

    # Cross-dataset sample caching configuration
    enable_sample_cache: bool = True  # Enable caching of sample metadata + CLIP embeddings (80-95% speedup for subsequent datasets)
    sample_cache_version: int = 3  # Sample cache format version (v3: includes CLIP embeddings, increment to invalidate old cache)

    # Visualization configuration
    viz_style: str = "default"
    viz_dpi: int = 100
    viz_save_figures: bool = False
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Auto-detect device if set to "auto"
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create directories if they don't exist
        Path(self.cache_base_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'MAVERICConfig':
        """
        Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            MAVERICConfig instance
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Filter config_dict to only include fields that exist in MAVERICConfig
        if hasattr(cls, '__dataclass_fields__'):
            valid_fields = set(cls.__dataclass_fields__.keys())
            filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}

            # Fix problematic cache paths
            if 'cache_base_dir' in filtered_config:
                cache_path = filtered_config['cache_base_dir']
                if cache_path.startswith('/content/') and not Path('/content').exists():
                    # Replace with local cache for non-Colab environments
                    filtered_config['cache_base_dir'] = './maveric_cache'
                    print(f"⚠️ Replaced inaccessible cache path {cache_path} with ./maveric_cache")

            # Log which fields were ignored for debugging
            ignored_fields = set(config_dict.keys()) - valid_fields
            if ignored_fields:
                print(f"⚠️ Ignoring unknown config fields: {sorted(ignored_fields)}")

            return cls(**filtered_config)
        else:
            return cls(**config_dict)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'MAVERICConfig':
        """
        Load configuration from a JSON file.
        
        Args:
            path: Path to JSON configuration file
            
        Returns:
            MAVERICConfig instance
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: Union[str, Path]):
        """
        Save configuration to a YAML file.
        
        Args:
            path: Output path for YAML file
        """
        config_dict = asdict(self)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, path: Union[str, Path]):
        """
        Save configuration to a JSON file.
        
        Args:
            path: Output path for JSON file
        """
        config_dict = asdict(self)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def update(self, **kwargs):
        """
        Update configuration values.
        
        Args:
            **kwargs: Key-value pairs to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
    
    def get_cache_paths(self) -> Dict[str, Path]:
        """
        Get all cache directory paths.
        
        Returns:
            Dictionary mapping cache types to Path objects
        """
        base_path = Path(self.cache_base_dir)
        return {
            'base': base_path,
            'images': base_path / 'image_cache',
            'results': base_path / 'results',
            'models': base_path / 'models',
            'embeddings': base_path / 'embeddings'
        }
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of warnings.
        
        Returns:
            List of warning messages (empty if configuration is valid)
        """
        warnings = []
        
        # Check if cache directory is accessible
        try:
            Path(self.cache_base_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            warnings.append(f"Cannot create cache directory: {e}")
        
        # Validate batch size
        if self.batch_size < 1:
            warnings.append("Batch size must be at least 1")

        # Validate scoring mode
        if self.scoring_mode not in ["clip", "hu_moments"]:
            warnings.append(f"Invalid scoring_mode '{self.scoring_mode}'. Must be 'clip' or 'hu_moments'")

        # Validate weights sum to 1.0
        weight_sum = sum(self.metric_weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            warnings.append(f"Metric weights sum to {weight_sum}, not 1.0")
        
        # Validate thresholds are in reasonable ranges
        for metric, threshold in self.default_thresholds.items():
            if 'resolution' in metric:
                if threshold < 0 or threshold > 10:
                    warnings.append(f"{metric} threshold {threshold} outside range [0, 10]")
            else:
                if threshold < 0 or threshold > 1:
                    warnings.append(f"{metric} threshold {threshold} outside range [0, 1]")
        
        return warnings


@dataclass
class TrainingConfig:
    """
    Configuration for model training/customization.
    
    This configuration controls the training process when customizing
    vision-language models with filtered data.
    """
    
    # Basic training parameters
    epochs: int = 10
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_steps: int = 0
    
    # Regularization
    use_regularization: bool = True
    regularization_weight: float = 0.5
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_strength: int = 2  # For RandAugment
    augmentation_magnitude: int = 9  # For RandAugment

    # Domain adaptation (simulates test data characteristics)
    use_domain_adaptation: bool = False
    domain_blur_probability: float = 0.3
    domain_blur_sigma_range: Tuple[float, float] = (0.1, 2.0)
    domain_jpeg_probability: float = 0.3
    domain_jpeg_quality_range: Tuple[int, int] = (30, 95)
    domain_downsample_probability: float = 0.3
    # Fixed target size for datasets like CIFAR-10 (32), MNIST (28), etc.
    domain_target_size: Optional[int] = None  # None = use scale_range
    # Scale range fallback for generic datasets
    domain_downsample_scale_range: Tuple[float, float] = (0.5, 0.9)

    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "cosine"  # cosine, linear, constant
    gradient_clip_value: float = 1.0
    
    # Evaluation
    eval_frequency: int = 1  # Evaluate every N epochs
    save_best_model: bool = True
    skip_epoch_evaluation: bool = False  # Skip per-epoch evaluation (useful for unified training)
    # Checkpoint selection: choose the best epoch on the validation fold, NEVER on test.
    # "val_acc" (default) = clean held-out selection. "test_acc" = legacy/ablation only.
    checkpoint_selection_metric: str = "val_acc"
    evaluate_test_each_epoch: bool = False  # monitoring only; never used for selection


    # Validation strategy
    use_validation: bool = True  # Whether to use validation during training
    validation_method: str = "stratified_kfold"  # "stratified_kfold" or "simple_split"
    validation_k_folds: int = 5  # Number of folds for k-fold validation
    validation_split: float = 0.2  # Fraction for simple split validation
    
    # Checkpointing
    checkpoint_dir: Optional[str] = None
    save_frequency: int = 1  # Save checkpoint every N epochs
    keep_last_n_checkpoints: int = 3
    
    # Advanced options
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1

    # Text source for training
    text_source: str = "labels"  # "labels" or "captions"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def validate(self) -> List[str]:
        """
        Validate training configuration.
        
        Returns:
            List of warning messages
        """
        warnings = []
        
        if self.epochs < 1:
            warnings.append("Epochs must be at least 1")
        
        if self.learning_rate <= 0:
            warnings.append("Learning rate must be positive")
        
        if self.optimizer not in ["adamw", "adam", "sgd"]:
            warnings.append(f"Unknown optimizer: {self.optimizer}")
        
        if self.scheduler not in ["cosine", "linear", "constant"]:
            warnings.append(f"Unknown scheduler: {self.scheduler}")
        
        if self.checkpoint_selection_metric not in ("val_acc", "test_acc"):
            warnings.append(f"Unknown checkpoint_selection_metric: {self.checkpoint_selection_metric}")

        if self.checkpoint_selection_metric == "val_acc" and not self.use_validation:
            warnings.append("checkpoint_selection_metric='val_acc' requires use_validation=True")

        if self.text_source not in ["labels", "captions"]:
            warnings.append(f"Unknown text_source: {self.text_source}. Must be 'labels' or 'captions'")

        return warnings


@dataclass
class ExperimentConfig:
    """
    Configuration for running experiments.
    
    This configuration helps manage experimental runs with different
    settings and track results systematically.
    """
    
    # Experiment identification
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Component configurations
    maveric_config: MAVERICConfig = field(default_factory=MAVERICConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment settings
    seed: int = 42
    num_runs: int = 1
    
    # Output settings
    output_dir: str = "./experiments"
    save_artifacts: bool = True
    
    # Tracking
    use_wandb: bool = False
    wandb_project: str = "maveric"
    wandb_entity: Optional[str] = None
    
    def get_run_name(self, run_idx: int = 0) -> str:
        """Generate a unique run name."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.name}_run{run_idx}_{timestamp}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary including nested configs."""
        return {
            'name': self.name,
            'description': self.description,
            'tags': self.tags,
            'maveric_config': self.maveric_config.to_dict(),
            'training_config': self.training_config.to_dict(),
            'seed': self.seed,
            'num_runs': self.num_runs,
            'output_dir': self.output_dir,
            'save_artifacts': self.save_artifacts,
            'use_wandb': self.use_wandb,
            'wandb_project': self.wandb_project,
            'wandb_entity': self.wandb_entity
        }