"""
Depth-based renderer for hologram display.

Uses Depth Anything V2 to create a 3D parallax effect from camera input.
The view angle slowly oscillates between -45° and +45° across slices.
"""
import cv2
import numpy as np
import sys
import os

from .renderer import Renderer

# Add control module to path for depth estimation
SCRIPTS_PATH = os.path.dirname(os.path.dirname(__file__))
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, SCRIPTS_PATH)

from control.depth import DepthCamera


class DepthRenderer(Renderer):
    """
    Renders a depth-projected 3D view of camera input.
    
    Uses depth estimation to displace pixels based on their depth,
    creating a parallax effect. The viewing angle slowly oscillates
    between -45° and +45° to create a 3D effect.
    
    All slices show the same frame but with slightly different viewing
    angles, creating the illusion of 3D when displayed on the hologram.
    """
    
    # Available color schemes
    COLOR_SCHEMES = ['rgb', 'hologram', 'matrix']
    
    def __init__(
        self,
        num_slices: int,
        width: int,
        height: int,
        camera_src: int = 0,
        encoder: str = 'vits',
        input_size: int = 196,
        depth_scale: float = 0.4,
        depth_threshold: float = 0.35,
        oscillation_cycles: float = 0.5,
        max_angle_deg: float = 45.0,
        flip_horizontal: bool = True,
        checkpoint_dir: str = None,
        color_scheme: str = 'hologram'
    ):
        """
        Initialize the depth renderer.
        
        Args:
            num_slices: Number of slices in a full rotation (e.g., 120)
            width: Width of each slice in pixels (e.g., 64)
            height: Height of each slice in pixels (e.g., 32)
            camera_src: Camera device index
            encoder: Depth model encoder ('vits', 'vitb', 'vitl', 'vitg')
            input_size: Input size for depth model (smaller = faster)
            depth_scale: How much depth affects displacement (0.0-1.0)
            depth_threshold: Background removal threshold (0-1, higher = more removed)
            oscillation_cycles: How many left-right cycles per full rotation
            max_angle_deg: Maximum viewing angle in degrees (±)
            flip_horizontal: Mirror the camera input
            checkpoint_dir: Directory containing depth model checkpoints
            color_scheme: Color palette ('rgb', 'hologram', 'matrix')
                - 'rgb': Original colors with outlines, blacks lifted to dark gray
                - 'hologram': Blue-tinted like Star Wars holograms
                - 'matrix': White-to-green gradient like The Matrix
        """
        super().__init__(num_slices, width, height)
        
        self.depth_scale = depth_scale
        self.depth_threshold = depth_threshold
        self.oscillation_cycles = oscillation_cycles
        self.max_angle_rad = np.radians(max_angle_deg)
        self.color_scheme = color_scheme if color_scheme in self.COLOR_SCHEMES else 'rgb'
        
        # Initialize depth camera
        self.depth_camera = DepthCamera(
            src=camera_src,
            encoder=encoder,
            input_size=input_size,
            output_size=None,  # Will resize ourselves
            flip_horizontal=flip_horizontal,
            checkpoint_dir=checkpoint_dir
        )
        
        # Cache for current frame data
        self._current_depth = None
        self._current_frame = None
        self._last_valid_output = None
        
        # Precompute coordinate grids (lazily initialized)
        self._grid_cache = {}
        
    def _get_grids(self, h: int, w: int):
        """Get or create cached coordinate grids for projection."""
        key = (h, w)
        if key not in self._grid_cache:
            y_img = np.linspace(-1, 1, h, dtype=np.float32)
            x_img = np.linspace(-1, 1, w, dtype=np.float32)
            xx, yy = np.meshgrid(x_img, y_img)
            # Base theta maps x position to angle (±90 degrees range for parallax)
            theta_base = xx * (np.pi * 0.5)
            self._grid_cache[key] = (xx, yy, theta_base)
        return self._grid_cache[key]
    
    def _update_frame(self):
        """Fetch the latest frame and depth from camera."""
        ret, depth, frame = self.depth_camera.read()
        if ret and depth is not None and frame is not None:
            # Invert depth: Depth Anything outputs higher values for closer objects,
            # but our projection expects lower values = closer
            self._current_depth = 1.0 - depth
            self._current_frame = frame
    
    def _calculate_angle(self, slice_index: int) -> float:
        """
        Calculate the viewing angle for a given slice.
        
        The angle oscillates smoothly between -max_angle and +max_angle
        using a sine wave across the slices.
        
        Args:
            slice_index: Current slice index (0 to num_slices-1)
            
        Returns:
            Viewing angle in radians
        """
        # Progress through the oscillation (0 to 2π * cycles)
        progress = (slice_index / self.num_slices) * 2 * np.pi * self.oscillation_cycles
        # Sine wave oscillation
        angle = np.sin(progress) * self.max_angle_rad
        return angle
    
    def _apply_color_scheme(self, colors: np.ndarray, depth_values: np.ndarray) -> np.ndarray:
        """
        Apply the selected color scheme to the pixel colors.
        
        Args:
            colors: BGR colors array of shape (N, 3)
            depth_values: Normalized depth values (0-1) for intensity mapping
            
        Returns:
            Processed BGR colors
        """
        if self.color_scheme == 'rgb':
            # Lift blacks to dark gray so they show on LED matrix
            # Calculate luminance
            luminance = (0.114 * colors[:, 0] + 0.587 * colors[:, 1] + 0.299 * colors[:, 2])
            # Find dark pixels (luminance < 30)
            dark_mask = luminance < 30
            # Lift dark pixels to dark gray (40, 40, 40)
            colors = colors.copy()
            colors[dark_mask] = np.maximum(colors[dark_mask], 40)
            return colors
            
        elif self.color_scheme == 'hologram':
            # Star Wars style blue hologram
            # Convert to grayscale intensity
            intensity = (0.114 * colors[:, 0] + 0.587 * colors[:, 1] + 0.299 * colors[:, 2]) / 255.0
            # Boost contrast and add depth-based glow (closer = slightly brighter)
            intensity = intensity * (0.7 + depth_values * 0.3)
            intensity = np.clip(intensity, 0, 1)
            
            # Map to blue-cyan hologram colors (BGR format)
            # Base blue tint with cyan highlights for brighter areas
            out_colors = np.zeros_like(colors)
            out_colors[:, 0] = (80 + intensity * 175).astype(np.uint8)   # B: strong blue base
            out_colors[:, 1] = (40 + intensity * 200).astype(np.uint8)   # G: cyan tint increases with brightness
            out_colors[:, 2] = (intensity * 60).astype(np.uint8)         # R: minimal, just for whites
            return out_colors
            
        elif self.color_scheme == 'matrix':
            # The Matrix style green with white highlights
            # Convert to grayscale intensity
            intensity = (0.114 * colors[:, 0] + 0.587 * colors[:, 1] + 0.299 * colors[:, 2]) / 255.0
            
            # Boost contrast significantly
            intensity = np.clip((intensity - 0.2) * 1.5, 0, 1)
            
            # Map to green gradient (BGR format)
            out_colors = np.zeros_like(colors)
            # Green varies from dark (30) to bright (255)
            out_colors[:, 1] = (30 + intensity * 225).astype(np.uint8)  # G: 30-255
            # Add some blue and red for brighter areas (white/cyan highlights)
            highlight = np.clip((intensity - 0.5) * 2.0, 0, 1)
            out_colors[:, 0] = (highlight * 120).astype(np.uint8)  # B: subtle cyan
            out_colors[:, 2] = (highlight * 80).astype(np.uint8)   # R: minimal
            return out_colors
        
        return colors
    
    def _add_outline(self, output: np.ndarray, depth_map: np.ndarray = None) -> np.ndarray:
        """
        Add silhouette outline to the rendered output based on depth discontinuities.
        Only outlines the boundary between foreground and background, not internal edges.
        
        Args:
            output: BGR image
            depth_map: Optional depth map for edge detection
            
        Returns:
            Image with silhouette outline added
        """
        if self.color_scheme != 'rgb':
            return output
        
        # Create a binary mask of rendered pixels (non-black)
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        mask = (gray > 0).astype(np.uint8) * 255
        
        # Find the outer contour only (silhouette)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw only the outermost contours with thin lines
        outline_color = (70, 70, 70)  # Subtle dark gray
        cv2.drawContours(output, contours, -1, outline_color, 1)
        
        return output
    
    def _project_with_depth(
        self,
        frame: np.ndarray,
        depth: np.ndarray,
        theta: float,
        out_width: int,
        out_height: int
    ) -> np.ndarray:
        """
        Project frame using depth information to create parallax effect.
        
        Args:
            frame: BGR input frame
            depth: Normalized depth map (0-1, higher = closer after inversion)
            theta: Viewing angle in radians
            out_width: Output width
            out_height: Output height
            
        Returns:
            Projected BGR image
        """
        # Resize inputs to working resolution
        work_h, work_w = 64, 64  # Internal working resolution
        depth_small = cv2.resize(depth, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
        frame_small = cv2.resize(frame, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
        
        # Get coordinate grids
        xx, yy, theta_base = self._get_grids(work_h, work_w)
        
        # Background mask: keep foreground
        # After depth inversion in _update_frame: lower values = closer
        # Keep pixels where inverted depth < threshold (i.e., close objects)
        fg_mask = depth_small < self.depth_threshold
        
        # Create output image (black background)
        output = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        
        if not np.any(fg_mask):
            # No foreground detected
            return output
        
        # Get foreground pixels
        depth_fg = depth_small[fg_mask]
        xx_fg = xx[fg_mask]
        yy_fg = yy[fg_mask]
        theta_base_fg = theta_base[fg_mask]
        colors_fg = frame_small[fg_mask]  # BGR colors
        
        # Calculate radius based on depth (closer = larger displacement)
        radius = 1.0 - depth_fg * self.depth_scale
        
        # Apply view rotation
        theta_total = theta_base_fg + theta
        
        # Cylindrical to projected coordinates
        sin_theta = np.sin(theta_total)
        cos_theta = np.cos(theta_total)
        
        # Project to 2D
        x_proj = radius * sin_theta
        y_proj = yy_fg
        z_proj = radius * cos_theta
        
        # Simple perspective
        perspective = 1.0 / (1.0 + (1.0 - z_proj) * 0.3)
        x_proj *= perspective
        y_proj *= perspective
        
        # Map to output coordinates
        out_x = ((x_proj + 1) * 0.5 * (out_width - 1)).astype(np.int32)
        out_y = ((y_proj + 1) * 0.5 * (out_height - 1)).astype(np.int32)
        
        # Filter valid coordinates
        valid = (out_x >= 0) & (out_x < out_width) & (out_y >= 0) & (out_y < out_height)
        
        if not np.any(valid):
            return output
        
        out_x = out_x[valid]
        out_y = out_y[valid]
        z_proj = z_proj[valid]
        colors_valid = colors_fg[valid]
        
        # Sort by depth (painter's algorithm: far to near)
        sort_idx = np.argsort(z_proj)
        out_x = out_x[sort_idx]
        out_y = out_y[sort_idx]
        z_sorted = z_proj[sort_idx]
        colors_sorted = colors_valid[sort_idx]
        
        # Apply color scheme
        # Convert z_proj to depth-like values (0=far, 1=close) for color processing
        depth_for_color = 1.0 - (z_sorted - z_sorted.min()) / (z_sorted.max() - z_sorted.min() + 1e-8)
        colors_processed = self._apply_color_scheme(colors_sorted, depth_for_color)
        
        # Draw pixels (painter's algorithm ensures closer pixels overwrite farther)
        output[out_y, out_x] = colors_processed
        
        # Add outlines for RGB mode
        output = self._add_outline(output)
        
        return output
    
    def render_slice(self, slice_index: int) -> np.ndarray:
        """
        Generate a depth-projected slice at the given index.
        
        Args:
            slice_index: The index of the slice to render (0 to num_slices-1)
            
        Returns:
            A numpy array of shape (height, width, 3) with BGR color data
        """
        # Update frame data from camera
        self._update_frame()
        
        # Calculate viewing angle for this slice
        theta = self._calculate_angle(slice_index)
        
        # Check if we have valid data
        if self._current_depth is None or self._current_frame is None:
            # Return black frame or last valid output
            if self._last_valid_output is not None:
                return self._last_valid_output.copy()
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Project the frame
        output = self._project_with_depth(
            self._current_frame,
            self._current_depth,
            theta,
            self.width,
            self.height
        )
        
        # Cache valid output
        if np.any(output > 0):
            self._last_valid_output = output.copy()
        
        return output
    
    def get_fps(self) -> float:
        """Get the current depth estimation FPS."""
        return self.depth_camera.get_fps()
    
    def release(self):
        """Release camera and depth estimator resources."""
        self.depth_camera.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


# Example usage and testing
if __name__ == '__main__':
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Depth Renderer Test')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--encoder', type=str, default='vits',
                        choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--input-size', type=int, default=196)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--num-slices', type=int, default=120)
    parser.add_argument('--color-scheme', type=str, default='rgb',
                        choices=['rgb', 'hologram', 'matrix'],
                        help='Color palette (rgb=original+outlines, hologram=Star Wars blue, matrix=green)')
    
    args = parser.parse_args()
    
    print(f"Initializing DepthRenderer...")
    print(f"  Camera: {args.camera}")
    print(f"  Encoder: {args.encoder}")
    print(f"  Output: {args.width}x{args.height}, {args.num_slices} slices")
    print(f"  Color scheme: {args.color_scheme}")
    
    with DepthRenderer(
        num_slices=args.num_slices,
        width=args.width,
        height=args.height,
        camera_src=args.camera,
        encoder=args.encoder,
        input_size=args.input_size,
        color_scheme=args.color_scheme
    ) as renderer:
        
        print("Warming up depth estimation...")
        time.sleep(2)
        
        print("Rendering slices... Press 'q' to quit")
        
        while True:
            # Render a few slices to show the oscillation
            for slice_idx in range(args.num_slices):
                frame = renderer.render_slice(slice_idx)
                
                # Scale up for visualization
                display = cv2.resize(frame, (640, 320), interpolation=cv2.INTER_NEAREST)
                
                # Add info
                fps = renderer.get_fps()
                angle_deg = np.degrees(renderer._calculate_angle(slice_idx))
                cv2.putText(display, f'Slice: {slice_idx}/{args.num_slices}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, f'Angle: {angle_deg:.1f}°', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, f'Depth FPS: {fps:.1f}', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Depth Renderer', display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                # Simulate the hologram timing
                time.sleep(0.0002)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
