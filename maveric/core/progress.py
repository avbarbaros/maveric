"""Progress tracking and real-time statistics for MAVERIC operations."""

import time
import threading
from typing import Dict, Optional


class RealTimeStats:
    """Real-time statistics display for download/cache operations."""
    
    def __init__(self, update_interval: float = 2.0, enable_display: bool = True):
        """
        Initialize real-time stats tracker.
        
        Args:
            update_interval: Seconds between display updates
            enable_display: Whether to show real-time updates
        """
        self.stats = {}
        self.lock = threading.Lock()
        self.last_update = time.time()
        self.update_interval = update_interval
        self.enable_display = enable_display
        
    def update_stats(self, new_stats: Dict):
        """Update statistics (called by cache manager and retriever)."""
        with self.lock:
            # Merge new stats with existing stats to preserve all fields
            self.stats.update(new_stats)
            
            if self.enable_display:
                current_time = time.time()
                # Only display if enough time has passed
                if current_time - self.last_update >= self.update_interval:
                    self._display_stats()
                    self.last_update = current_time
                
    def _display_stats(self):
        """Display current statistics."""
        if not self.stats:
            return
            
        successful = self.stats.get('downloads_successful', 0)
        failed = self.stats.get('downloads_failed', 0)
        
        # Create status line with consistent format
        status_parts = []
        
        # Always show downloads with batch position if available
        batch_size = self.stats.get('batch_size', None)
        current_batch_position = self.stats.get('current_batch_position', None)
        
        if batch_size and current_batch_position is not None:
            status_parts.append(f"✅ Downloads: {current_batch_position} / {batch_size}")
        elif successful > 0:
            status_parts.append(f"✅ Downloads: {successful}")
        else:
            status_parts.append(f"✅ Downloads: 0")
        
        # Always show failed count
        status_parts.append(f"❌ Failed: {failed}")
        
        # Always show current index information if available
        current_index = self.stats.get('current_index', None)
        total_samples = self.stats.get('total_samples', None)
        
        if current_index is not None and total_samples is not None:
            status_parts.append(f"📍 Index: {current_index} / {total_samples}")
        else:
            status_parts.append(f"📍 Index: - / -")
            
        if status_parts:
            status_line = " | ".join(status_parts)
            print(f"\r[STATS] {status_line}", end="", flush=True)
            
    def final_display(self):
        """Display final statistics and move to new line."""
        if self.enable_display:
            print()  # New line after progress
            
    def get_current_stats(self) -> Dict:
        """Get current statistics snapshot."""
        with self.lock:
            return self.stats.copy()