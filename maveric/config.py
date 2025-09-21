"""Configuration management for MAVERIC."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union
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
        'img2img': 0.40,
        'txt2txt': 0.20,
        'img2txt': 0.20,
        'txt2img': 0.20
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
        'imagenet_probability': 0.5,      # ImageNet probability threshold (global quality)
        'target_class_quality': 0.493,    # Target class quality using CLIP mappings
        'consistency': 0.796,             # Multimodal consistency
        # Legacy
        'weighted_class_score': 0.493     # Kept for backwards compatibility
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

            # Map legacy field names
            if 'quality_thresholds' in config_dict and 'quality_thresholds' not in valid_fields:
                if 'default_thresholds' in valid_fields:
                    filtered_config['default_thresholds'] = config_dict['quality_thresholds']

            # Fix problematic cache paths
            if 'cache_base_dir' in filtered_config:
                cache_path = filtered_config['cache_base_dir']
                if cache_path.startswith('/content/') and not Path('/content').exists():
                    # Replace with local cache for non-Colab environments
                    filtered_config['cache_base_dir'] = './maveric_cache'
                    print(f"⚠️ Replaced inaccessible cache path {cache_path} with ./maveric_cache")

            # Log which fields were ignored for debugging
            ignored_fields = set(config_dict.keys()) - valid_fields - {'quality_thresholds'}
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
    
    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "cosine"  # cosine, linear, constant
    gradient_clip_value: float = 1.0
    
    # Evaluation
    eval_frequency: int = 1  # Evaluate every N epochs
    save_best_model: bool = True
    
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