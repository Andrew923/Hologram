"""
Abstract base class for hologram volume renderers.
"""
from abc import ABC, abstractmethod
import numpy as np


class Renderer(ABC):
    """
    Abstract base class for generating hologram slice data.
    
    Subclasses must implement the `render_slice` method to generate
    the actual pixel data for each slice of the holographic volume.
    """
    
    def __init__(self, num_slices: int, width: int, height: int):
        """
        Initialize the renderer with volume dimensions.
        
        Args:
            num_slices: Number of slices in a full rotation (e.g., 120)
            width: Width of each slice in pixels (e.g., 64)
            height: Height of each slice in pixels (e.g., 32)
        """
        self.num_slices = num_slices
        self.width = width
        self.height = height
    
    @abstractmethod
    def render_slice(self, slice_index: int) -> np.ndarray:
        """
        Generate the image data for a single slice.
        
        Args:
            slice_index: The index of the slice to render (0 to num_slices-1)
            
        Returns:
            A numpy array of shape (height, width, 3) with BGR color data (uint8)
        """
        pass
    
    def render_all_slices(self) -> list:
        """
        Pre-render all slices for the volume.
        
        Returns:
            A list of numpy arrays, one for each slice
        """
        return [self.render_slice(i) for i in range(self.num_slices)]
