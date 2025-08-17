"""Factory for creating dataset handlers."""

from typing import Dict, Type, Optional, List
from ..core.base import BaseDataset
from ..core.exceptions import DatasetError


class DatasetFactory:
    """
    Factory class for creating dataset handlers.
    
    This factory manages the creation of dataset handlers for different datasets,
    making it easy to add support for new datasets by registering them.
    """
    
    # Registry of available datasets
    _datasets: Dict[str, Type[BaseDataset]] = {}
    
    @classmethod
    def register(cls, name: str, dataset_class: Type[BaseDataset]):
        """
        Register a new dataset handler.
        
        Args:
            name: Dataset name (e.g., 'cifar10', 'cifar100')
            dataset_class: Dataset handler class
        """
        cls._datasets[name.lower()] = dataset_class
    
    @classmethod
    def create_dataset(cls, dataset_name: str, **kwargs) -> BaseDataset:
        """
        Create appropriate dataset handler.
        
        Args:
            dataset_name: Name of dataset (cifar10, cifar100, imagenet, etc.)
            **kwargs: Dataset-specific arguments
            
        Returns:
            Dataset handler instance
            
        Raises:
            DatasetError: If dataset is not supported
        """
        name_lower = dataset_name.lower()
        
        if name_lower not in cls._datasets:
            available = ', '.join(sorted(cls._datasets.keys()))
            raise DatasetError(
                f"Dataset '{dataset_name}' not supported. "
                f"Available datasets: {available}"
            )
        
        dataset_class = cls._datasets[name_lower]
        return dataset_class(**kwargs)
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """Get list of available dataset names."""
        return sorted(cls._datasets.keys())


def get_dataset(dataset_name: str, **kwargs) -> BaseDataset:
    """
    Convenience function to get a dataset handler.
    
    Args:
        dataset_name: Name of the dataset
        **kwargs: Dataset-specific arguments
        
    Returns:
        Dataset handler instance
    """
    return DatasetFactory.create_dataset(dataset_name, **kwargs)


# Auto-register built-in datasets
def _register_builtin_datasets():
    """Register built-in dataset handlers."""
    from .elevater_datasets import ELEVATERDataset
    
    # Register all ELEVATER datasets (official 20 datasets)
    elevater_datasets = [
        'caltech101', 'cifar10', 'cifar100', 'country211', 'dtd', 'eurosat',
        'fer2013', 'fgvc_aircraft', 'food101', 'gtsrb', 'hateful_memes',
        'kitti_distance', 'mnist', 'oxford_flowers102', 'oxford_pets',
        'patchcamelyon', 'rendered_sst2', 'resisc45', 'stanford_cars', 'voc2007'
    ]
    
    for dataset_name in elevater_datasets:
        # Create a factory function for each dataset
        def create_elevater_dataset(name=dataset_name):
            def factory(**kwargs):
                return ELEVATERDataset(name, **kwargs)
            return factory
        
        DatasetFactory.register(dataset_name, create_elevater_dataset())
    
    # Register common aliases
    DatasetFactory.register('cifar-10', create_elevater_dataset('cifar10'))
    DatasetFactory.register('cifar-100', create_elevater_dataset('cifar100'))


# Register datasets on import
_register_builtin_datasets()
