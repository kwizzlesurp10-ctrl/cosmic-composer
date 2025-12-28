"""
Model caching utilities to avoid repeated loading.
"""
import os
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Global model cache with checkpoint path as key
_model_cache: Dict[str, 'CosmicTransformer'] = {}


def get_cached_model(checkpoint_path: str) -> 'CosmicTransformer':
    """
    Get model instance with caching to avoid repeated loading.
    
    Args:
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Cached or newly loaded model instance
    """
    global _model_cache
    if checkpoint_path not in _model_cache:
        from models.transformer import CosmicTransformer
        logger.info(f"Loading model from {checkpoint_path}")
        _model_cache[checkpoint_path] = CosmicTransformer.load(checkpoint_path)
    return _model_cache[checkpoint_path]
