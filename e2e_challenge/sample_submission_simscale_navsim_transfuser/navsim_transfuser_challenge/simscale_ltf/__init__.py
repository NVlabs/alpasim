"""Inference-only adaptation of the Apache-2.0 SimScale LTF model."""

from .config import TransfuserConfig
from .model import TransfuserModel

__all__ = ["TransfuserConfig", "TransfuserModel"]
