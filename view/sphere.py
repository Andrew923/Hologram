"""
Sphere renderer implementation for hologram display.
"""
import numpy as np
import math

from .renderer import Renderer


class SphereRenderer(Renderer):
    """
    Renders a 3D sphere with a diagonal color gradient.
    
    The sphere is mathematically sliced at each angle, creating
    circular cross-sections with HSV-based coloring.
    """
    
    def __init__(self, num_slices: int, width: int, height: int, radius: int = None):
        """
        Initialize the sphere renderer.
        
        Args:
            num_slices: Number of slices in a full rotation
            width: Width of each slice in pixels
            height: Height of each slice in pixels
            radius: Radius of the sphere in pixels (default: min(width, height) // 2 - 2)
        """
        super().__init__(num_slices, width, height)
        
        self.center_x = width // 2
        self.center_y = height // 2
        self.radius = radius if radius is not None else min(width, height) // 2 - 2
    
    def render_slice(self, slice_index: int) -> np.ndarray:
        """
        Generate a slice of the sphere at the given angle.
        
        The sphere is rendered with a diagonal gradient based on
        pixel position and slice angle.
        
        Args:
            slice_index: The index of the slice to render (0 to num_slices-1)
            
        Returns:
            A numpy array of shape (height, width, 3) with BGR color data
        """
        # Create empty canvas
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Calculate angle for this slice (0 to 2PI) - useful for 3D effects
        theta = (slice_index / self.num_slices) * 2 * math.pi
        
        # Check every pixel to see if it's inside the sphere
        for y in range(self.height):
            for x in range(self.width):
                # Calculate distance from center of panel
                dist_sq = (x - self.center_x)**2 + (y - self.center_y)**2
                
                if dist_sq <= self.radius**2:
                    # Pixel is inside the sphere
                    
                    # Diagonal Gradient: Hue changes with X + Y + Angle
                    hue = int((x * 2 + y * 4 + (slice_index * 2)) % 180)
                    
                    # Simple color map (blue-ish gradient)
                    img[y, x] = [hue, 255 - hue, 200]
        
        return img
