"""
Depth estimation module using Depth Anything V2.

Provides a threaded depth map generator that works with the FastVideoCapture
camera module to produce real-time depth maps at a specified resolution.
"""

import cv2
import numpy as np
import threading
import torch
import sys
import os

# Add Depth-Anything-V2 to path
DEPTH_ANYTHING_PATH = os.path.join(os.path.dirname(__file__), '..', 'test', 'Depth-Anything-V2')
if DEPTH_ANYTHING_PATH not in sys.path:
    sys.path.insert(0, DEPTH_ANYTHING_PATH)

from depth_anything_v2.dpt import DepthAnythingV2


class DepthEstimator:
    """
    Real-time depth estimation using Depth Anything V2.
    
    Runs inference in a background thread, always providing the most recent
    depth map for a given camera source.
    """
    
    MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    def __init__(
        self,
        camera,
        encoder='vits',
        input_size=196,
        output_size=None,
        checkpoint_dir=None,
        normalize=True
    ):
        """
        Initialize the depth estimator.
        
        Args:
            camera: A camera object with a read() method (e.g., FastVideoCapture)
            encoder: Model encoder size ('vits', 'vitb', 'vitl', 'vitg')
            input_size: Input size for the model (smaller = faster, try 196, 252, 518)
            output_size: Output depth map size as (width, height), None for original size
            checkpoint_dir: Directory containing model checkpoints
            normalize: If True, normalize depth to 0-1 range
        """
        self.camera = camera
        self.input_size = input_size
        self.output_size = output_size
        self.normalize = normalize
        
        # Thread control
        self.running = True
        self.lock = threading.Lock()
        self.depth = None
        self.frame = None
        self.fps = 0.0
        self._last_time = None
        
        # Determine checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(DEPTH_ANYTHING_PATH, 'checkpoints')
        
        # Initialize model
        self.device = self._get_device()
        self.model = self._load_model(encoder, checkpoint_dir)
        
        # Start processing thread
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
    
    def _get_device(self):
        """Determine the best available device."""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def _load_model(self, encoder, checkpoint_dir):
        """Load the depth estimation model."""
        import glob
        
        if encoder not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown encoder '{encoder}'. Choose from: {list(self.MODEL_CONFIGS.keys())}")
        
        model = DepthAnythingV2(**self.MODEL_CONFIGS[encoder])
        
        # Find checkpoint file
        patterns = [
            os.path.join(checkpoint_dir, f'depth_anything_v2_{encoder}.pth'),
            os.path.join(checkpoint_dir, f'depth_anything_v2_{encoder}.pth*'),
        ]
        
        checkpoint_path = None
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                checkpoint_path = matches[0]
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Could not find checkpoint for encoder '{encoder}' in {checkpoint_dir}. "
                f"Please download the model weights."
            )
        
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model = model.to(self.device).eval()
        
        return model
    
    def _process_loop(self):
        """Background thread that continuously processes frames."""
        import time
        
        while self.running:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue
            
            # Run depth estimation
            with torch.inference_mode():
                depth = self.model.infer_image(frame, self.input_size)
            
            # Normalize if requested
            if self.normalize:
                d_min, d_max = depth.min(), depth.max()
                depth = ((depth - d_min) / (d_max - d_min + 1e-8)).astype(np.float32)
            
            # Resize to output size if specified
            if self.output_size is not None:
                depth = cv2.resize(depth, self.output_size, interpolation=cv2.INTER_LINEAR)
            
            # Calculate FPS
            current_time = time.time()
            if self._last_time is not None:
                self.fps = 1.0 / (current_time - self._last_time + 1e-8)
            self._last_time = current_time
            
            # Update shared data
            with self.lock:
                self.depth = depth
                self.frame = frame
    
    def read(self):
        """
        Get the most recent depth map and corresponding frame.
        
        Returns:
            tuple: (success_flag, depth_map, frame)
                - success_flag: True if valid data is available
                - depth_map: Normalized depth map (0-1 if normalize=True)
                - frame: The RGB frame used to generate the depth map
        """
        with self.lock:
            if self.depth is None:
                return False, None, None
            return True, self.depth.copy(), self.frame.copy()
    
    def read_depth(self):
        """
        Get only the most recent depth map.
        
        Returns:
            tuple: (success_flag, depth_map)
        """
        with self.lock:
            if self.depth is None:
                return False, None
            return True, self.depth.copy()
    
    def get_fps(self):
        """Get the current processing FPS."""
        return self.fps
    
    def release(self):
        """Stop the processing thread."""
        self.running = False
        self._thread.join(timeout=2.0)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class DepthCamera:
    """
    Convenience class that combines FastVideoCapture and DepthEstimator.
    
    Provides a simple interface for getting depth maps from a camera.
    """
    
    def __init__(
        self,
        src=0,
        encoder='vits',
        input_size=196,
        output_size=None,
        flip_horizontal=True,
        checkpoint_dir=None
    ):
        """
        Initialize depth camera.
        
        Args:
            src: Camera source (device index or video file path)
            encoder: Model encoder size ('vits', 'vitb', 'vitl', 'vitg')
            input_size: Input size for the model
            output_size: Output depth map size as (width, height)
            flip_horizontal: If True, flip frames horizontally (mirror effect)
            checkpoint_dir: Directory containing model checkpoints
        """
        from .camera import FastVideoCapture
        
        self.camera = FastVideoCapture(src=src, flip_horizontal=flip_horizontal)
        self.depth_estimator = DepthEstimator(
            camera=self.camera,
            encoder=encoder,
            input_size=input_size,
            output_size=output_size,
            checkpoint_dir=checkpoint_dir
        )
    
    def read(self):
        """
        Get the most recent depth map and frame.
        
        Returns:
            tuple: (success_flag, depth_map, frame)
        """
        return self.depth_estimator.read()
    
    def read_depth(self):
        """
        Get only the most recent depth map.
        
        Returns:
            tuple: (success_flag, depth_map)
        """
        return self.depth_estimator.read_depth()
    
    def get_fps(self):
        """Get the current processing FPS."""
        return self.depth_estimator.get_fps()
    
    def release(self):
        """Release all resources."""
        self.depth_estimator.release()
        self.camera.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


# Example usage and testing
if __name__ == '__main__':
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Depth Estimation Module Test')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--encoder', type=str, default='vits', 
                        choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--input-size', type=int, default=196)
    parser.add_argument('--output-width', type=int, default=None)
    parser.add_argument('--output-height', type=int, default=None)
    
    args = parser.parse_args()
    
    output_size = None
    if args.output_width and args.output_height:
        output_size = (args.output_width, args.output_height)
    
    print(f"Starting depth camera (encoder={args.encoder}, input_size={args.input_size})")
    
    with DepthCamera(
        src=args.camera,
        encoder=args.encoder,
        input_size=args.input_size,
        output_size=output_size,
        flip_horizontal=True
    ) as depth_cam:
        
        print("Warming up...")
        time.sleep(2)
        
        print("Press 'q' to quit")
        
        while True:
            ret, depth, frame = depth_cam.read()
            
            if not ret:
                continue
            
            # Visualize depth
            depth_vis = (depth * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            
            # Add FPS text
            fps = depth_cam.get_fps()
            cv2.putText(depth_color, f'FPS: {fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show side by side
            combined = np.hstack([frame, depth_color])
            cv2.imshow('Depth Camera', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
