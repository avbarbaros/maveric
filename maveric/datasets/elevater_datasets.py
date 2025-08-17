"""Support for ELEVATER benchmark datasets."""

from typing import Dict, List, Optional, Any
import json
import random
from pathlib import Path
from PIL import Image

from ..core.base import BaseDataset
from ..core.exceptions import DatasetError


class ELEVATERDataset(BaseDataset):
    """
    Handler for ELEVATER benchmark datasets.
    
    ELEVATER (Evaluation of Language-augmented Visual Task Adaptation with Eye-Tracking)
    includes multiple vision datasets for comprehensive evaluation.
    """
    
    # Mapping of ELEVATER dataset names to their properties
    ELEVATER_DATASETS = {
        'caltech101': {
            'num_classes': 102,
            'task': 'classification'
        },
        'cifar10': {
            'num_classes': 10,
            'task': 'classification'
        },
        'cifar100': {
            'num_classes': 100,
            'task': 'classification'
        },
        'country211': {
            'num_classes': 211,
            'task': 'classification'
        },
        'dtd': {
            'num_classes': 47,
            'task': 'classification'
        },
        'eurosat': {
            'num_classes': 10,
            'task': 'classification'
        },
        'fer2013': {
            'num_classes': 7,
            'task': 'classification'
        },
        'fgvc_aircraft': {
            'num_classes': 100,
            'task': 'classification'
        },
        'food101': {
            'num_classes': 101,
            'task': 'classification'
        },
        'gtsrb': {
            'num_classes': 43,
            'task': 'classification'
        },
        'hateful_memes': {
            'num_classes': 2,
            'task': 'classification'
        },
        'kitti_distance': {
            'num_classes': 4,
            'task': 'classification'
        },
        'mnist': {
            'num_classes': 10,
            'task': 'classification'
        },
        'oxford_flowers102': {
            'num_classes': 102,
            'task': 'classification'
        },
        'oxford_pets': {
            'num_classes': 37,
            'task': 'classification'
        },
        'patchcamelyon': {
            'num_classes': 2,
            'task': 'classification'
        },
        'rendered_sst2': {
            'num_classes': 2,
            'task': 'classification'
        },
        'resisc45': {
            'num_classes': 45,
            'task': 'classification'
        },
        'stanford_cars': {
            'num_classes': 196,
            'task': 'classification'
        },
        'voc2007': {
            'num_classes': 20,
            'task': 'classification'
        }
    }
    
    def __init__(self, dataset_name: str, data_dir: Optional[str] = None,
                 metadata_path: Optional[str] = None):
        """
        Initialize ELEVATER dataset handler.
        
        Args:
            dataset_name: Name of the ELEVATER dataset
            data_dir: Directory containing dataset files
            metadata_path: Path to metadata JSON file
        """
        super().__init__()
        
        if dataset_name not in self.ELEVATER_DATASETS:
            raise DatasetError(
                f"Dataset '{dataset_name}' not in ELEVATER. "
                f"Available: {', '.join(sorted(self.ELEVATER_DATASETS.keys()))}"
            )
        
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir) if data_dir else Path('./data/elevater')
        self.metadata_path = metadata_path
        self._metadata = None
        self._class_names = None
        
        # Load metadata if available
        if metadata_path and Path(metadata_path).exists():
            self._load_metadata()
    
    def _load_metadata(self):
        """Load dataset metadata from JSON file."""
        try:
            with open(self.metadata_path, 'r') as f:
                self._metadata = json.load(f)
            
            # Extract class names from metadata
            if 'classes' in self._metadata:
                self._class_names = self._metadata['classes']
            elif 'classnames' in self._metadata:
                self._class_names = self._metadata['classnames']
                
            self.log_info(f"Loaded metadata for {self.dataset_name}")
        except Exception as e:
            raise DatasetError(f"Failed to load metadata: {e}")
    
    @property
    def name(self) -> str:
        """Return the dataset name."""
        return self.dataset_name
    
    @property
    def class_names(self) -> List[str]:
        """Return list of class names in the dataset."""
        if self._class_names is not None:
            return self._class_names
        
        # Fallback: try to load from standard locations
        class_file = self.data_dir / self.dataset_name / 'classes.txt'
        if class_file.exists():
            with open(class_file, 'r') as f:
                self._class_names = [line.strip() for line in f]
            return self._class_names
        
        # If no class names found, generate generic ones
        num_classes = self.ELEVATER_DATASETS[self.dataset_name]['num_classes']
        self._class_names = [f"class_{i}" for i in range(num_classes)]
        return self._class_names
    
    def get_reference_samples(self, n_per_class: int, seed: int = 42) -> Dict[str, List[Image.Image]]:
        """
        Get reference samples for each class.
        
        For ELEVATER datasets, this requires the dataset to be properly set up
        with image files organized by class.
        
        Args:
            n_per_class: Number of reference samples per class
            seed: Random seed for reproducible sampling
        """
        # Set random seed for reproducible sampling
        random.seed(seed)
        reference_samples = {}
        
        # Check if dataset directory exists
        dataset_dir = self.data_dir / self.dataset_name
        if not dataset_dir.exists():
            self.log_warning(
                f"Dataset directory {dataset_dir} not found. "
                "Please download and set up the dataset first."
            )
            return reference_samples
        
        # Try different common directory structures
        for split in ['train', 'training', 'val', 'validation']:
            split_dir = dataset_dir / split
            if split_dir.exists():
                # Look for class subdirectories
                for class_name in self.class_names:
                    class_dir = split_dir / class_name
                    if class_dir.exists() and class_dir.is_dir():
                        # Get image files
                        image_files = list(class_dir.glob('*.jpg')) + \
                                    list(class_dir.glob('*.png')) + \
                                    list(class_dir.glob('*.jpeg'))
                        
                        # Sample images
                        sampled_files = random.sample(
                            image_files,
                            min(n_per_class, len(image_files))
                        )
                        
                        # Load images
                        images = []
                        for img_file in sampled_files:
                            try:
                                img = Image.open(img_file).convert('RGB')
                                images.append(img)
                            except Exception as e:
                                self.log_warning(f"Failed to load {img_file}: {e}")
                        
                        reference_samples[class_name] = images
                
                if reference_samples:
                    break
        
        return reference_samples
    
    def get_text_templates(self) -> List[str]:
        """
        Get text templates for creating prompts.
        
        Templates can be customized based on the specific ELEVATER dataset.
        """
        # Dataset-specific templates
        dataset_templates = {
            'dtd': [
                "a {} texture",
                "a photo of a {} texture",
                "the {} pattern"
            ],
            'eurosat': [
                "a satellite photo of {}",
                "aerial view of {}",
                "satellite imagery showing {}"
            ],
            'fer2013': [
                "a face showing {}",
                "a person feeling {}",
                "facial expression of {}"
            ],
            'food101': [
                "a photo of {}, a type of food",
                "a dish called {}",
                "food image showing {}"
            ],
            'gtsrb': [
                "a {} traffic sign",
                "road sign showing {}",
                "traffic sign for {}"
            ],
            'oxford_flowers102': [
                "a photo of a {}, a type of flower",
                "a flower called {}",
                "blooming {} flower"
            ],
            'oxford_pets': [
                "a photo of a {}, a type of pet",
                "a pet {}",
                "an animal called {}"
            ]
        }
        
        # Return dataset-specific templates if available
        if self.dataset_name in dataset_templates:
            return dataset_templates[self.dataset_name]
        
        # Default templates
        return [
            "a photo of a {}",
            "a clear image of a {}",
            "a picture showing a {}",
            "a photograph of a {}",
            "an image of a {}",
            "a high-quality photo of a {}",
            "a detailed view of a {}"
        ]