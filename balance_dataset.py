#!/usr/bin/env python3
"""
Standalone script to balance manually cleaned training datasets.
Can be run directly: python balance_dataset.py --input ./data --output ./balanced
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from maveric.utils.balance_cli import main

if __name__ == '__main__':
    main()
