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
                # Force immediate display if cache_hits changed (important events)
                cache_hits_changed = 'cache_hits' in new_stats
                time_elapsed = current_time - self.last_update >= self.update_interval

                # Display if: time interval passed OR important stat changed
                if time_elapsed or cache_hits_changed:
                    self._display_stats()
                    self.last_update = current_time
                
    def _display_stats(self):
        """Display current statistics."""
        if not self.stats:
            return

        successful = self.stats.get('downloads_successful', 0)
        failed = self.stats.get('downloads_failed', 0)
        cache_hits = self.stats.get('cache_hits', 0)

        # Create status line with consistent format
        status_parts = []

        # Show processed count with batch position if available
        batch_size = self.stats.get('batch_size', None)
        current_batch_position = self.stats.get('current_batch_position', None)

        if batch_size and current_batch_position is not None:
            status_parts.append(f"✅ Processed: {current_batch_position} / {batch_size}")
        elif current_batch_position is not None:
            status_parts.append(f"✅ Processed: {current_batch_position}")
        else:
            status_parts.append(f"✅ Processed: 0")

        # ALWAYS show cache hits (even if 0) for consistency
        status_parts.append(f"⚡ Cache Hits: {cache_hits}")

        # ALWAYS show downloads (even if 0) for consistency and to verify: Processed = Cache Hits + Downloads
        status_parts.append(f"📥 Downloads: {successful}")

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