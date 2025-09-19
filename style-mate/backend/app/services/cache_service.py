"""
High-performance caching service using Redis
"""

import redis.asyncio as redis
import hashlib
import json
import pickle
from typing import Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

class CacheService:
    """
    Redis-based caching with fallback to in-memory cache
    """
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}
        self.stats = {"hits": 0, "misses": 0}
        self._connect()
    
    def _connect(self):
        """Connect to Redis"""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=False,
                socket_connect_timeout=5
            )
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis connection failed, using memory cache: {e}")
            self.redis_client = None
    
    def generate_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, bytes):
            return f"img_{hashlib.md5(data).hexdigest()}"
        elif isinstance(data, str):
            return f"str_{hashlib.md5(data.encode()).hexdigest()}"
        else:
            return f"obj_{hashlib.md5(str(data).encode()).hexdigest()}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    self.stats["hits"] += 1
                    return pickle.loads(value)
            else:
                # Fallback to memory cache
                if key in self.memory_cache:
                    self.stats["hits"] += 1
                    return self.memory_cache[key]
            
            self.stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL"""
        try:
            serialized = pickle.dumps(value)
            
            if self.redis_client:
                await self.redis_client.set(key, serialized, ex=ttl)
            else:
                # Store in memory cache
                self.memory_cache[key] = value
                # Simple size limit for memory cache
                if len(self.memory_cache) > 1000:
                    # Remove oldest entries
                    keys_to_remove = list(self.memory_cache.keys())[:100]
                    for k in keys_to_remove:
                        del self.memory_cache[k]
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def delete(self, key: str):
        """Delete value from cache"""
        try:
            if self.redis_client:
                await self.redis_client.delete(key)
            else:
                self.memory_cache.pop(key, None)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
    
    async def clear(self):
        """Clear all cache"""
        try:
            if self.redis_client:
                await self.redis_client.flushdb()
            else:
                self.memory_cache.clear()
            self.stats = {"hits": 0, "misses": 0}
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    async def get_stats(self, key: str) -> int:
        """Get statistics"""
        try:
            if self.redis_client:
                value = await self.redis_client.get(f"stats:{key}")
                return int(value) if value else 0
            return 0
        except:
            return 0
    
    async def incr_stats(self, key: str):
        """Increment statistics counter"""
        try:
            if self.redis_client:
                await self.redis_client.incr(f"stats:{key}")
        except:
            pass
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.stats["hits"] + self.stats["misses"]
        if total == 0:
            return 0.0
        return (self.stats["hits"] / total) * 100
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()