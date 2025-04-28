from collections import OrderedDict
from threading import Lock
from .prompt_cache import PromptCache

class PromptCachePool:
    """
    A pool of PromptCache objects, keyed by conversation ID, with LRU eviction.
    """

    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self.pool: OrderedDict[str, PromptCache] = OrderedDict()
        self.lock = Lock()

    def get(self, key: str) -> PromptCache:
        """
        Retrieve the PromptCache for `key`, creating it if missing.
        Evict oldest cache when pool size exceeds capacity.
        """
        with self.lock:
            if key in self.pool:
                # mark as recently used
                self.pool.move_to_end(key)
                return self.pool[key]
            # create new cache
            cache = PromptCache()
            self.pool[key] = cache
            # evict least recently used
            if len(self.pool) > self.capacity:
                old_key, old_cache = self.pool.popitem(last=False)
                # free VRAM
                old_cache.cache.clear()
            return cache

    def touch(self, key: str):
        """Mark `key` as recently used without creating."""
        with self.lock:
            if key in self.pool:
                self.pool.move_to_end(key)
