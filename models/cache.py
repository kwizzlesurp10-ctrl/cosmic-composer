"""
Model caching utilities to avoid repeated loading.
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global model cache
_model_cache: Optional['CosmicTransformer'] = None


def get_cached_model(checkpoint_path: str) -> 'CosmicTransformer':
    """
    Get model instance with caching to avoid repeated loading.
    
    Args:
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Cached or newly loaded model instance
    """
    global _model_cache
    if _model_cache is None:
        from models.transformer import CosmicTransformer
        logger.info(f"Loading model from {checkpoint_path}")
        _model_cache = CosmicTransformer.load(checkpoint_path)
    return _model_cache
