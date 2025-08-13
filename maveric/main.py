"""Main MAVERIC API that ties everything together."""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging

# Convenience imports
import pandas as pd

from .config import MAVERICConfig, TrainingConfig
from .core.base import BaseComponent
from .core.interfaces import RetrievalResult, QualityResult, CustomizationResult
from .core.exceptions import MAVERICError, ConfigurationError
from .core.progress import RealTimeStats
from .retrieval import Retriever, CacheManager
from .retrieval.dataset_handlers import REACTDatasetHandler
from .quality import QualityController
from .customization import ModelCustomizer
from .visualization import MetricsVisualizer, SampleVisualizer
from .interactive import QualityDashboard
from .utils import setup_logging


class MAVERIC(BaseComponent):
    """
    Main entry point for MAVERIC library.
    
    Provides high-level API for retrieval, quality control, and customization
    of vision-language models using curated datasets.
    """
    
    def __init__(self, config: Optional[MAVERICConfig] = None):
        """
        Initialize MAVERIC with configuration.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        super().__init__("MAVERIC")
        
        # Set configuration
        self.config = config or MAVERICConfig()
        
        # Initialize real-time stats based on config
        self.real_time_stats = RealTimeStats(enable_display=self.config.enable_real_time_stats) if self.config.enable_real_time_stats else None
        
        # Validate configuration
        warnings = self.config.validate()
        if warnings:
            for warning in warnings:
                self.log_warning(warning)
        
        # Setup logging
        setup_logging(
            level=self.config.log_level,
            log_to_file=self.config.log_to_file,
            log_file=self.config.log_file_path
        )
        
        # Initialize components
        self._init_components()
        
        self.log_info("MAVERIC initialized successfully")
    
    def _init_components(self):
        """Initialize core components."""
        # Cache manager
        self.cache_manager = CacheManager(
            base_dir=self.config.cache_base_dir,
            enable_image_cache=self.config.enable_image_cache,
            cache_format=self.config.cache_format,
            cache_quality=self.config.cache_quality,
            stats_callback=self.real_time_stats.update_stats if self.real_time_stats else None
        )
        
        # Retriever
        self.retriever = Retriever(
            clip_model=self.config.clip_model,
            device=self.config.device,
            cache_manager=self.cache_manager,
            n_reference_images=self.config.n_reference_images
        )
        
        # Quality controller (initialized on demand)
        self.quality_controller = None
        
        # Model customizer (initialized on demand)
        self.customizer = None
        
        # Visualizers
        self.visualizers = {
            'metrics': MetricsVisualizer(
                style=self.config.viz_style,
                dpi=self.config.viz_dpi
            ),
            'samples': SampleVisualizer()
        }
    
    def retrieve(self,
                 dataset_name: str,
                 target_dataset: str,
                 num_samples: Optional[int] = None,
                 start_index: int = 0,
                 cache_results: bool = True) -> RetrievalResult:
        """
        Retrieve and score samples from source dataset for target dataset.
        
        Args:
            dataset_name: Source dataset name (e.g., 'react-vl/react-retrieval-datasets')
            target_dataset: Target dataset name (e.g., 'cifar10', 'cifar100', 'imagenet')
            num_samples: Number of samples to retrieve (None for all)
            start_index: Index to start retrieval from
            cache_results: Whether to cache retrieval results
            
        Returns:
            RetrievalResult object containing retrieved samples and metadata
        """
        self.log_info(f"Starting retrieval from {dataset_name} for {target_dataset}")
        
        # Check if we have cached results
        if cache_results:
            cached_results = self.cache_manager.load_results(target_dataset)
            if cached_results:
                self.log_info(f"Loaded {len(cached_results)} cached results")
                return RetrievalResult(
                    samples=cached_results,
                    source_dataset=dataset_name,
                    target_dataset=target_dataset
                )
        
        # Create dataset handler
        if "react" in dataset_name.lower():
            dataset_handler = REACTDatasetHandler(dataset_name)
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not yet supported")
        
        # Skip to start index if needed
        if start_index > 0:
            dataset_handler = dataset_handler.skip(start_index)
        
        # Perform retrieval
        result = self.retriever.retrieve(
            dataset_handler=dataset_handler,
            target_dataset=target_dataset,
            rotation_size=self.config.retrieval_rotation_size,
            num_samples=num_samples,
            start_index=start_index
        )
        
        self.log_info(f"Retrieved {result.total_samples} samples")
        
        # Finalize real-time stats display
        if self.real_time_stats:
            self.real_time_stats.final_display()
        
        return result
    
    def quality_control(self,
                       data: Union[RetrievalResult, pd.DataFrame, str],
                       thresholds: Optional[Dict[str, float]] = None,
                       weights: Optional[Dict[str, float]] = None,
                       balance_strategy: str = 'median',
                       balance_config: Optional[Dict[str, Any]] = None) -> QualityResult:
        """
        Apply quality control filtering to retrieved samples.
        
        Args:
            data: RetrievalResult, DataFrame, or path to saved results
            thresholds: Quality thresholds (uses config defaults if None)
            weights: Metric weights (uses config defaults if None)
            balance_strategy: Dataset balancing strategy
            balance_config: Additional balance configuration
            
        Returns:
            QualityResult object with filtered samples
        """
        # Initialize quality controller if needed
        if self.quality_controller is None:
            self.quality_controller = QualityController()
        
        # Load data
        if isinstance(data, RetrievalResult):
            self.quality_controller.load_data(data.to_dataframe())
        else:
            self.quality_controller.load_data(data)
        
        # Set thresholds
        if thresholds is None:
            thresholds = self.config.default_thresholds
        
        for metric, value in thresholds.items():
            self.quality_controller.set_threshold(metric, value)
        
        # Set weights
        if weights is None:
            weights = self.config.metric_weights
        
        for metric, value in weights.items():
            self.quality_controller.set_class_weight(metric, value)
        
        # Apply thresholds
        filtered_count = self.quality_controller.apply_thresholds()
        self.log_info(f"Filtered to {filtered_count} samples")
        
        # Apply balancing if requested
        if balance_strategy != 'none':
            balance_cfg = balance_config or {
                'min_samples': self.config.balance_min_samples,
                'max_samples': self.config.balance_max_samples,
                'enable_oversampling': self.config.balance_enable_oversampling
            }
            
            self.quality_controller.balance_dataset(
                strategy=balance_strategy,
                **balance_cfg
            )
        
        # Create result
        result = self.quality_controller.create_quality_result(
            thresholds=thresholds,
            balance_strategy=balance_strategy
        )
        
        return result
    
    def customize_model(self,
                       quality_result: QualityResult,
                       model_name: str = None,
                       training_config: Optional[TrainingConfig] = None,
                       target_dataset: str = None) -> CustomizationResult:
        """
        Customize pre-trained model using filtered samples.
        
        Args:
            quality_result: Result from quality_control() method
            model_name: Pre-trained model to customize (uses config default if None)
            training_config: Training configuration (uses defaults if None)
            target_dataset: Target dataset name for the customization
            
        Returns:
            CustomizationResult with trained model and metrics
        """
        # Use defaults if not provided
        if model_name is None:
            model_name = self.config.clip_model
        
        if training_config is None:
            training_config = TrainingConfig()
        
        # Validate training config
        warnings = training_config.validate()
        if warnings:
            for warning in warnings:
                self.log_warning(warning)
        
        # Initialize customizer if needed
        if self.customizer is None:
            checkpoint_dir = None
            if training_config.checkpoint_dir:
                checkpoint_dir = training_config.checkpoint_dir
            elif self.config.cache_base_dir:
                checkpoint_dir = Path(self.config.cache_base_dir) / 'checkpoints'
            
            self.customizer = ModelCustomizer(
                base_model_name=model_name,
                device=self.config.device,
                checkpoint_dir=checkpoint_dir
            )
        
        # Get class names from the data
        import pandas as pd
        df = pd.DataFrame(quality_result.filtered_samples)
        if 'label' not in df.columns:
            raise ValueError("No 'label' column found in filtered data")
        
        class_names = sorted(df['label'].unique())
        
        # Perform customization
        result = self.customizer.customize(
            quality_result=quality_result,
            training_config=training_config,
            target_dataset_name=target_dataset or "custom",
            class_names=class_names
        )
        
        return result
    
    def visualize_retrieval(self,
                           retrieval_result: RetrievalResult,
                           metrics: Optional[List[str]] = None,
                           save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Visualize retrieval results with distribution plots.
        
        Args:
            retrieval_result: Result from retrieve() method
            metrics: Metrics to visualize (uses all available if None)
            save_dir: Directory to save figures (displays if None)
            
        Returns:
            Dictionary mapping metric names to figure objects
        """
        df = retrieval_result.to_dataframe()
        
        if metrics is None:
            metrics = retrieval_result.available_metrics
        
        figures = {}
        
        for metric in metrics:
            if metric in df.columns:
                fig = self.visualizers['metrics'].plot_metric_distribution(
                    df, 
                    metric,
                    save_path=f"{save_dir}/{metric}_dist.png" if save_dir else None
                )
                figures[metric] = fig
        
        # Class distribution
        if 'label' in df.columns:
            from .visualization.plots import plot_class_distribution
            fig = plot_class_distribution(
                df,
                save_path=f"{save_dir}/class_distribution.png" if save_dir else None
            )
            figures['class_distribution'] = fig
        
        return figures
    
    def launch_dashboard(self,
                        data: Union[str, RetrievalResult, QualityResult, pd.DataFrame]) -> Any:
        """
        Launch interactive quality control dashboard.
        
        Args:
            data: Data source (file path, result object, or DataFrame)
            
        Returns:
            Interactive dashboard widget (for Jupyter environments)
        """
        dashboard = QualityDashboard(cache_dir=self.config.cache_base_dir)
        return dashboard.launch(data)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.cache_manager.get_cache_stats()
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """
        Get current real-time statistics snapshot.
        
        Returns:
            Dictionary containing current download/cache statistics
        """
        if self.real_time_stats:
            return self.real_time_stats.get_current_stats()
        return {}
    
    def clear_cache(self, cache_type: str = 'all'):
        """
        Clear cache files.
        
        Args:
            cache_type: Type of cache to clear ('images', 'results', 'embeddings', 'all')
        """
        self.cache_manager.clear_cache(cache_type)
        self.log_info(f"Cleared {cache_type} cache")
    
    def save_config(self, path: str):
        """
        Save current configuration to file.
        
        Args:
            path: Output path for configuration file
        """
        path = Path(path)
        if path.suffix == '.yaml':
            self.config.to_yaml(path)
        elif path.suffix == '.json':
            self.config.to_json(path)
        else:
            raise ValueError("Configuration file must be .yaml or .json")
        
        self.log_info(f"Saved configuration to {path}")
    
    @classmethod
    def from_config_file(cls, path: str) -> 'MAVERIC':
        """
        Create MAVERIC instance from configuration file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            MAVERIC instance
        """
        path = Path(path)
        
        if path.suffix == '.yaml':
            config = MAVERICConfig.from_yaml(path)
        elif path.suffix == '.json':
            config = MAVERICConfig.from_json(path)
        else:
            raise ValueError("Configuration file must be .yaml or .json")
        
        return cls(config)
    
    @staticmethod
    def get_available_elevater_datasets() -> Dict[str, Dict[str, Any]]:
        """
        Get list of available ELEVATER datasets.
        
        Returns:
            Dictionary mapping dataset names to their properties
        """
        from .datasets.elevater_datasets import ELEVATERDataset
        return ELEVATERDataset.ELEVATER_DATASETS
    
    @staticmethod
    def display_elevater_datasets() -> List[str]:
        """
        Display all available ELEVATER datasets and return the sorted list.
        
        Returns:
            Sorted list of dataset names
        """
        from .datasets.elevater_datasets import ELEVATERDataset
        
        datasets = list(ELEVATERDataset.ELEVATER_DATASETS.keys())
        
        print("\n📊 Available ELEVATER Datasets:")
        print("=" * 60)
        for i, dataset in enumerate(sorted(datasets), 1):
            dataset_info = ELEVATERDataset.ELEVATER_DATASETS[dataset]
            print(f"  {i:2d}. {dataset:<20} ({dataset_info['num_classes']} classes)")
        print("=" * 60)
        print(f"Total: {len(datasets)} datasets available")
        
        return sorted(datasets)

