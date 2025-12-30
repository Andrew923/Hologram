"""
View module for hologram rendering.

This module provides abstract and concrete renderer implementations
for generating holographic volume data.
"""
from .renderer import Renderer
from .sphere import SphereRenderer
from .depth_renderer import DepthRenderer

__all__ = ['Renderer', 'SphereRenderer', 'DepthRenderer']
