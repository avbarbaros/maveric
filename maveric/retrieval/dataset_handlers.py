"""Dataset handlers for different data sources."""

from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional
from datasets import load_dataset
import requests

from ..core.base import BaseComponent
from ..core.exceptions import DatasetError


class DatasetHandler(BaseComponent, ABC):
    """Abstract base class for dataset handlers."""
    
    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset samples."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Get total number of samples."""
        pass
    
    def skip(self, n: int) -> 'DatasetHandler':
        """Skip first n samples."""
        raise NotImplementedError("Subclass must implement skip() method")


class REACTDatasetHandler(DatasetHandler):
    """Handler for REACT retrieval dataset."""
    
    def __init__(self, dataset_name: str = "react-vl/react-retrieval-datasets"):
        """
        Initialize REACT dataset handler.
        
        Args:
            dataset_name: HuggingFace dataset name
        """
        super().__init__("REACTDatasetHandler")
        self.dataset_name = dataset_name
        self._dataset = None
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the dataset from HuggingFace."""
        try:
            self.log_info(f"Loading dataset: {self.dataset_name}")
            self._dataset = load_dataset(self.dataset_name, split='train')
            self.log_info(f"Loaded {len(self._dataset)} samples")
        except Exception as e:
            raise DatasetError(f"Failed to load dataset {self.dataset_name}: {e}")
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset samples."""
        for item in self._dataset:
            yield {
                'URL': item.get('URL', ''),
                'TEXT': item.get('TEXT', ''),
                'metadata': item
            }
    
    def __len__(self) -> int:
        """Get total number of samples."""
        return len(self._dataset) if self._dataset else 0
    
    def skip(self, n: int) -> 'REACTDatasetHandler':
        """Skip first n samples."""
        self._dataset = self._dataset.skip(n)
        return self


class HuggingFaceDatasetHandler(DatasetHandler):
    """Generic handler for HuggingFace datasets."""
    
    def __init__(self, 
                 dataset_name: str,
                 split: str = 'train',
                 image_column: str = 'image',
                 text_column: str = 'text',
                 streaming: bool = False):
        """
        Initialize HuggingFace dataset handler.
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split to use
            image_column: Name of image column
            text_column: Name of text column
            streaming: Whether to use streaming mode
        """
        super().__init__("HuggingFaceDatasetHandler")
        self.dataset_name = dataset_name
        self.split = split
        self.image_column = image_column
        self.text_column = text_column
        self.streaming = streaming
        self._dataset = None
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the dataset from HuggingFace."""
        try:
            self.log_info(f"Loading dataset: {self.dataset_name}")
            self._dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                streaming=self.streaming
            )
            if not self.streaming:
                self.log_info(f"Loaded {len(self._dataset)} samples")
        except Exception as e:
            raise DatasetError(f"Failed to load dataset {self.dataset_name}: {e}")
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset samples."""
        for item in self._dataset:
            # Handle different dataset formats
            image_data = item.get(self.image_column)
            text_data = item.get(self.text_column, '')
            
            # Convert image to URL if it's a PIL Image
            if hasattr(image_data, 'save'):
                # It's a PIL Image, we need to handle it differently
                # For now, skip these as they need special handling
                continue
            
            yield {
                'URL': image_data if isinstance(image_data, str) else '',
                'TEXT': text_data,
                'metadata': item
            }
    
    def __len__(self) -> int:
        """Get total number of samples."""
        if self.streaming:
            return -1  # Unknown length for streaming datasets
        return len(self._dataset) if self._dataset else 0
