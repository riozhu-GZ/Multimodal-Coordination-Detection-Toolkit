"""
Multimodal Coordination Detection Toolkit
==========================================

Detects coordinated inauthentic behaviour on social media by finding accounts
that post similar multimodal content (text + images) within time windows.

Public API
----------
>>> from multimodal_coordination import detect_coordination
>>> results = detect_coordination(posts_df, output_dir="./results", precomputed_embeddings=True)
>>> print(results)  # {"TiCNet": "./results/TiCNet.graphml", "LiCNet": "./results/LiCNet.graphml"}
"""

from .pipeline import detect_coordination

__all__ = ["detect_coordination"]
__version__ = "0.1.0"
