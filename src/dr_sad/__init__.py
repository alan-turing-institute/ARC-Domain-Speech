"""
ARC-Domain-Speech: ARC project on domain robust speech activity detection.
"""

from __future__ import annotations

from importlib.metadata import version

__all__ = (
    "__version__",
    "analysis",
    "annotation",
    "data",
    "models",
    "pyannet",
    "segment",
    "training",
)
__version__ = version(__name__)

from . import analysis, annotation, data, models, pyannet, segment, training
