import cv2
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from .camera import FastVideoCapture


class HandPoseDetector(ABC):
    """Abstract base class for hand pose detection models."""
    
    # Hand skeleton connections (trt_pose joint order)
    # Format: (joint1_idx, joint2_idx)
    HAND_CONNECTIONS = [
        # Thumb
        (0, 4), (4, 3), (3, 2), (2, 1),
        # Index
        (0, 8), (8, 7), (7, 6), (6, 5),
        # Middle
        (0, 12), (12, 11), (11, 10), (10, 9),
        # Ring
        (0, 16), (16, 15), (15, 14), (14, 13),
        # Pinky
        (0, 20), (20, 19), (19, 18), (18, 17),
    ]
    
    # Fingertip joint indices
    FINGERTIP_INDICES = [1, 5, 9, 13, 17]  # Thumb, Index, Middle, Ring, Pinky tips
    
    # Joint names in trt_pose order
    JOINT_NAMES = [
        "wrist",           # 0
        "thumb_tip",       # 1
        "thumb_ip",        # 2
        "thumb_mcp",       # 3
        "thumb_cmc",       # 4
        "index_tip",       # 5
        "index_dip",       # 6
        "index_pip",       # 7
        "index_mcp",       # 8
        "middle_tip",      # 9
        "middle_dip",      # 10
        "middle_pip",      # 11
        "middle_mcp",      # 12
        "ring_tip",        # 13
        "ring_dip",        # 14
        "ring_pip",        # 15
        "ring_mcp",        # 16
        "pinky_tip",       # 17
        "pinky_dip",       # 18
        "pinky_pip",       # 19
        "pinky_mcp",       # 20
    ]
    
    NUM_JOINTS = 21
    
    def __init__(self, camera: Optional[FastVideoCapture] = None, camera_src: int = 0):
        """
        Initialize the hand pose detector.
        
        Args:
            camera: Optional pre-existing FastVideoCapture instance
            camera_src: Camera source index if creating new capture
        """
        self._owns_camera = camera is None
        self.camera = camera if camera is not None else FastVideoCapture(camera_src, flip_horizontal=True)
        self._model_loaded = False
    
    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the model weights and initialize inference.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[List[float]]:
        """
        Detect hand landmarks in the given frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of 21 joints, each as [x, y] normalized to 0-1 range.
            Returns empty list if no hand detected.
        """
        pass
    
    def detect_from_camera(self) -> Tuple[Optional[np.ndarray], List[List[float]]]:
        """
        Read a frame from the camera and detect hand landmarks.
        
        Returns:
            Tuple of (frame, joints) where frame is the captured image
            and joints is the list of detected landmarks.
        """
        ret, frame = self.camera.read()
        if not ret or frame is None:
            return None, []
        
        joints = self.detect(frame)
        return frame, joints
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._model_loaded
    
    def release(self):
        """Release resources."""
        if self._owns_camera and self.camera is not None:
            self.camera.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class TRTPoseHandDetector(HandPoseDetector):
    """Hand pose detector using TRT_POSE (optimized for NVIDIA Jetson)."""
    
    DEFAULT_MODEL_WEIGHTS = '/data/trt_pose_hand/model/hand_pose_resnet18_att_244_244.pth'
    DEFAULT_HAND_POSE_JSON = '/data/trt_pose_hand/preprocess/hand_pose.json'
    
    def __init__(
        self,
        model_weights: str = DEFAULT_MODEL_WEIGHTS,
        hand_pose_json: str = DEFAULT_HAND_POSE_JSON,
        camera: Optional[FastVideoCapture] = None,
        camera_src: int = 0,
        cmap_threshold: float = 0.15,
        link_threshold: float = 0.15,
    ):
        """
        Initialize TRT_POSE hand detector.
        
        Args:
            model_weights: Path to model weights file
            hand_pose_json: Path to hand pose JSON config
            camera: Optional pre-existing FastVideoCapture instance
            camera_src: Camera source index if creating new capture
            cmap_threshold: Confidence threshold for keypoint detection
            link_threshold: Threshold for skeleton link detection
        """
        super().__init__(camera, camera_src)
        
        self.model_weights = model_weights
        self.hand_pose_json = hand_pose_json
        self.cmap_threshold = cmap_threshold
        self.link_threshold = link_threshold
        
        # Will be set during load_model()
        self.model = None
        self.parse_objects = None
        self.mean = None
        self.std = None
        self.topology = None
    
    def load_model(self) -> bool:
        """Load TRT_POSE model."""
        try:
            import torch
            import trt_pose.coco
            import trt_pose.models
            from trt_pose.parse_objects import ParseObjects
            
            # Load topology
            with open(self.hand_pose_json, 'r') as f:
                hand_pose = json.load(f)
            self.topology = trt_pose.coco.coco_category_to_topology(hand_pose)
            
            # Load model
            num_keypoints = len(hand_pose['keypoints'])
            num_links = 2 * len(hand_pose['skeleton'])
            self.model = trt_pose.models.resnet18_baseline_att(num_keypoints, num_links)
            self.model = self.model.cuda().eval()
            self.model.load_state_dict(torch.load(self.model_weights))
            
            # Initialize parser
            self.parse_objects = ParseObjects(
                self.topology,
                cmap_threshold=self.cmap_threshold,
                link_threshold=self.link_threshold
            )
            
            # Hardware transforms
            self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
            self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
            
            self._model_loaded = True
            return True
            
        except Exception as e:
            print(f"Failed to load TRT_POSE model: {e}")
            self._model_loaded = False
            return False
    
    def _preprocess(self, frame: np.ndarray):
        """Preprocess frame for TRT_POSE inference."""
        import torch
        import torchvision.transforms as transforms
        import PIL.Image
        
        device = torch.device('cuda')
        
        # Resize to model input (224x224)
        image = cv2.resize(frame, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        
        return image[None, ...]
    
    def _parse_output(self, counts, objects, peaks) -> List[List[float]]:
        """Parse TRT_POSE output to joint list."""
        joints = []
        count = int(counts[0])
        
        if count == 0:
            return []
        
        obj = objects[0][0]  # Get first hand only
        for j in range(obj.shape[0]):
            k = int(obj[j])
            if k >= 0:
                peak = peaks[0][j][k]
                x = float(peak[1])
                y = float(peak[0])
                joints.append([x, y])
            else:
                joints.append([0.0, 0.0])
        
        return joints
    
    def detect(self, frame: np.ndarray) -> List[List[float]]:
        """Detect hand landmarks using TRT_POSE."""
        if not self._model_loaded:
            return []
        
        data = self._preprocess(frame)
        cmap, paf = self.model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf)
        
        return self._parse_output(counts, objects, peaks)


class MediaPipeHandDetector(HandPoseDetector):
    """Hand pose detector using MediaPipe (cross-platform, CPU-friendly)."""
    
    # Mapping from trt_pose index to MediaPipe index for compatibility
    TRT_TO_MEDIAPIPE = [
        0,   # 0: wrist -> wrist
        4,   # 1: thumb tip
        3,   # 2: thumb IP
        2,   # 3: thumb MCP
        1,   # 4: thumb CMC
        8,   # 5: index tip
        7,   # 6: index DIP
        6,   # 7: index PIP
        5,   # 8: index MCP
        12,  # 9: middle tip
        11,  # 10: middle DIP
        10,  # 11: middle PIP
        9,   # 12: middle MCP
        16,  # 13: ring tip
        15,  # 14: ring DIP
        14,  # 15: ring PIP
        13,  # 16: ring MCP
        20,  # 17: pinky tip
        19,  # 18: pinky DIP
        18,  # 19: pinky PIP
        17,  # 20: pinky MCP
    ]
    
    def __init__(
        self,
        camera: Optional[FastVideoCapture] = None,
        camera_src: int = 0,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize MediaPipe hand detector.
        
        Args:
            camera: Optional pre-existing FastVideoCapture instance
            camera_src: Camera source index if creating new capture
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum detection confidence threshold
            min_tracking_confidence: Minimum tracking confidence threshold
        """
        super().__init__(camera, camera_src)
        
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Will be set during load_model()
        self.hands = None
    
    def load_model(self) -> bool:
        """Load MediaPipe hand detection model."""
        try:
            import mediapipe as mp
            
            mp_hands = mp.solutions.hands
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            self._model_loaded = True
            return True
            
        except Exception as e:
            print(f"Failed to load MediaPipe model: {e}")
            self._model_loaded = False
            return False
    
    def _convert_landmarks(self, hand_landmarks) -> List[List[float]]:
        """
        Convert MediaPipe landmarks to trt_pose joint order.
        
        Returns joints in trt_pose order for compatibility with other code.
        """
        joints = []
        for trt_idx in range(self.NUM_JOINTS):
            mp_idx = self.TRT_TO_MEDIAPIPE[trt_idx]
            lm = hand_landmarks.landmark[mp_idx]
            joints.append([lm.x, lm.y])  # Already normalized 0-1
        
        return joints
    
    def detect(self, frame: np.ndarray) -> List[List[float]]:
        """Detect hand landmarks using MediaPipe."""
        if not self._model_loaded:
            return []
        
        # Resize and convert for MediaPipe
        img_resized = cv2.resize(frame, (224, 224))
        rgb_frame = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            return self._convert_landmarks(hand_landmarks)
        
        return []
    
    def release(self):
        """Release MediaPipe resources."""
        if self.hands is not None:
            self.hands.close()
        super().release()


def create_detector(
    model_type: str = 'trt_pose',
    camera: Optional[FastVideoCapture] = None,
    camera_src: int = 0,
    **kwargs
) -> HandPoseDetector:
    """
    Factory function to create a hand pose detector.
    
    Args:
        model_type: 'trt_pose' or 'mediapipe'
        camera: Optional pre-existing FastVideoCapture instance
        camera_src: Camera source index if creating new capture
        **kwargs: Additional arguments passed to the detector constructor
        
    Returns:
        HandPoseDetector instance (model not yet loaded)
    """
    if model_type == 'trt_pose':
        return TRTPoseHandDetector(camera=camera, camera_src=camera_src, **kwargs)
    elif model_type == 'mediapipe':
        return MediaPipeHandDetector(camera=camera, camera_src=camera_src, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'trt_pose' or 'mediapipe'.")
