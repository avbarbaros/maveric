"""I/O utility functions."""

import json
import csv
import requests
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from tqdm import tqdm


def load_json(path: Union[str, Path]) -> Any:
    """
    Load JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2):
    """
    Save data to JSON file.

    Args:
        data: Data to save
        path: Output path
        indent: Indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def save_json_atomic(data: Any, path: Union[str, Path], indent: int = 2, timeout: Optional[float] = None):
    """
    Save data to JSON file atomically to prevent corruption.

    Uses atomic write pattern: write to temp file, then rename.
    This prevents partial writes on network filesystems that could cause
    the process to hang indefinitely.

    Args:
        data: Data to save
        path: Output path
        indent: Indentation level
        timeout: Optional timeout for the write operation (in seconds)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory as target (ensures same filesystem)
    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f'.{path.name}.',
        suffix='.tmp'
    )

    try:
        # Write to temp file with timeout if specified
        with os.fdopen(fd, 'w') as f:
            if timeout is not None:
                # For timeout support, we'd need signal-based timeout
                # For now, just do the write (TODO: add proper timeout)
                json.dump(data, f, indent=indent)
            else:
                json.dump(data, f, indent=indent)

        # Atomic rename (works on both POSIX and Windows)
        os.replace(temp_path, path)

    except Exception as e:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except:
            pass
        raise e


def load_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load CSV file.
    
    Args:
        path: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        DataFrame
    """
    return pd.read_csv(path, **kwargs)


def save_csv(data: pd.DataFrame, path: Union[str, Path], **kwargs):
    """
    Save DataFrame to CSV.
    
    Args:
        data: DataFrame to save
        path: Output path
        **kwargs: Additional arguments for to_csv
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data.to_csv(path, index=False, **kwargs)


def download_file(url: str, 
                  output_path: Union[str, Path],
                  chunk_size: int = 8192,
                  timeout: int = 30) -> Path:
    """
    Download file from URL.
    
    Args:
        url: URL to download from
        output_path: Output file path
        chunk_size: Download chunk size
        timeout: Request timeout
        
    Returns:
        Path to downloaded file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return output_path


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
