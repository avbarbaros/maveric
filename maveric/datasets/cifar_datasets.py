"""CIFAR-10 and CIFAR-100 dataset handlers."""

import random
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import torchvision
import torch
from torch.utils.data import DataLoader

from ..core.base import BaseDataset
from ..core.exceptions import DatasetError


class CIFARBaseDataset(BaseDataset):
    """Base class for CIFAR datasets with common functionality."""
    
    def __init__(self, root: str = './data', train: bool = True, download: bool = True):
        """
        Initialize CIFAR dataset.
        
        Args:
            root: Root directory for dataset storage
            train: Whether to load training set
            download: Whether to download dataset if not found
        """
        super().__init__()
        self.root = root
        self.train = train
        self.download = download
        self._dataset = None
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the actual dataset - to be implemented by subclasses."""
        raise NotImplementedError
    
    def get_reference_samples(self, n_per_class: int, seed: int = 42) -> Dict[str, List[Image.Image]]:
        """
        Get reference samples for each class.
        
        This method selects representative samples from each class to serve
        as reference points for quality assessment.
        
        Args:
            n_per_class: Number of reference samples per class
            seed: Random seed for reproducible sampling
            
        Returns:
            Dictionary mapping class names to lists of PIL images
        """
        if self._dataset is None:
            raise DatasetError("Dataset not loaded")
        
        # Set random seed for reproducible sampling
        random.seed(seed)
        
        reference_samples = {}
        
        # Group indices by class
        class_indices = {class_name: [] for class_name in self.class_names}
        
        for idx in range(len(self._dataset)):
            _, label = self._dataset[idx]
            class_name = self.class_names[label]
            class_indices[class_name].append(idx)
        
        # Sample from each class
        for class_name, indices in class_indices.items():
            # Randomly sample indices with fixed seed for reproducibility
            sampled_indices = random.sample(
                indices, 
                min(n_per_class, len(indices))
            )
            
            # Get images
            images = []
            for idx in sampled_indices:
                img, _ = self._dataset[idx]
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                images.append(img)
            
            reference_samples[class_name] = images
        
        return reference_samples
    
    def get_text_templates(self) -> List[str]:
        """
        Get text templates for creating prompts.
        
        These templates are used to create text descriptions for each class,
        which are then used for text-image alignment scoring.
        """
        return [
            "a photo of a {}",
            "a clear image of a {}",
            "a picture showing a {}",
            "a photograph of a {}",
            "an image of a {}",
            "a high-quality photo of a {}",
            "a detailed view of a {}"
        ]
    
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True, 
                      num_workers: int = 0) -> DataLoader:
        """
        Get a PyTorch DataLoader for the dataset.
        
        Args:
            batch_size: Batch size for loading
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            
        Returns:
            PyTorch DataLoader
        """
        if self._dataset is None:
            raise DatasetError("Dataset not loaded")
        
        # Custom transform to ensure PIL Image output
        def pil_transform(img):
            if not isinstance(img, Image.Image):
                return Image.fromarray(img)
            return img
        
        # Apply transform
        original_transform = self._dataset.transform
        self._dataset.transform = pil_transform
        
        dataloader = DataLoader(
            self._dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._custom_collate
        )
        
        # Restore original transform
        self._dataset.transform = original_transform
        
        return dataloader
    
    def _custom_collate(self, batch):
        """Custom collate function to handle PIL images."""
        images = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])
        return images, labels


class CIFAR10Dataset(CIFARBaseDataset):
    """Handler for CIFAR-10 dataset."""
    
    @property
    def name(self) -> str:
        """Return the dataset name."""
        return "cifar10"
    
    @property
    def class_names(self) -> List[str]:
        """Return list of class names in the dataset."""
        return [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    def _load_dataset(self):
        """Load CIFAR-10 dataset."""
        try:
            self._dataset = torchvision.datasets.CIFAR10(
                root=self.root,
                train=self.train,
                download=self.download,
                transform=None  # We'll handle transforms separately
            )
            self.log_info(f"Loaded CIFAR-10 dataset with {len(self._dataset)} samples")
        except Exception as e:
            raise DatasetError(f"Failed to load CIFAR-10 dataset: {e}")


class CIFAR100Dataset(CIFARBaseDataset):
    """Handler for CIFAR-100 dataset."""
    
    @property
    def name(self) -> str:
        """Return the dataset name."""
        return "cifar100"
    
    @property
    def class_names(self) -> List[str]:
        """Return list of class names in the dataset."""
        return [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]
    
    def _load_dataset(self):
        """Load CIFAR-100 dataset."""
        try:
            self._dataset = torchvision.datasets.CIFAR100(
                root=self.root,
                train=self.train,
                download=self.download,
                transform=None  # We'll handle transforms separately
            )
            self.log_info(f"Loaded CIFAR-100 dataset with {len(self._dataset)} samples")
        except Exception as e:
            raise DatasetError(f"Failed to load CIFAR-100 dataset: {e}")
