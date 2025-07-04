# maveric/datasets/__init__.py
"""Dataset handling for MAVERIC."""

from .dataset_factory import DatasetFactory, get_dataset
from .cifar_datasets import CIFAR10Dataset, CIFAR100Dataset
from .elevater_datasets import ELEVATERDataset

__all__ = [
    "DatasetFactory",
    "get_dataset",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "ELEVATERDataset"
]