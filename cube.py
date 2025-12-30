import cv2
import json
import curses
import math
import numpy as np

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
WIDTH = 224
HEIGHT = 224

# MODEL SELECTION: 'trt_pose' or 'mediapipe'
USE_MODEL = 'mediapipe'

# TRT_POSE paths
MODEL_WEIGHTS = '/data/trt_pose_hand/model/hand_pose_resnet18_att_244_244.pth'
HAND_POSE_JSON = '/data/trt_pose_hand/preprocess/hand_pose.json'

# ---------------------------------------------------------
# 2. MODEL SETUP
# ---------------------------------------------------------

if USE_MODEL == 'trt_pose':
    import torch
    import trt_pose.coco
    import trt_pose.models
    from trt_pose.parse_objects import ParseObjects
    import torchvision.transforms as transforms
    import PIL.Image

    with open(HAND_POSE_JSON, 'r') as f:
        hand_pose = json.load(f)

    topology = trt_pose.coco.coco_category_to_topology(hand_pose)
    num_parts = len(hand_pose['keypoints'])
    num_links = len(hand_pose['skeleton'])

    # Load Model
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    model.load_state_dict(torch.load(MODEL_WEIGHTS))

    # Data Preprocessing
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    parse_objects = ParseObjects(topology, cmap_threshold=0.15, link_threshold=0.15)

    def preprocess(image):
        device = torch.device('cuda')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    class PreprocessData:
        def __init__(self, topology, num_parts):
            self.topology = topology
            self.num_parts = num_parts

        def joints_inference(self, image, counts, objects, peaks):
            joints = []
            height = image.shape[0]
            width = image.shape[1]
            count = int(counts[0])
            for i in range(count):
                obj = objects[0][i]
                C = obj.shape[0]
                for j in range(C):
                    k = int(obj[j])
                    if k >= 0:
                        peak = peaks[0][j][k]
                        x = round(float(peak[1]) * width)
                        y = round(float(peak[0]) * height)
                        joints.append([x, y])
                    else:
                        joints.append([0, 0])
            return joints

    preprocessdata = PreprocessData(topology, num_parts)

elif USE_MODEL == 'mediapipe':
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # MediaPipe hand landmark indices (for reference):
    # 0=WRIST
    # Thumb: 1=CMC, 2=MCP, 3=IP, 4=TIP
    # Index: 5=MCP, 6=PIP, 7=DIP, 8=TIP
    # Middle: 9=MCP, 10=PIP, 11=DIP, 12=TIP
    # Ring: 13=MCP, 14=PIP, 15=DIP, 16=TIP
    # Pinky: 17=MCP, 18=PIP, 19=DIP, 20=TIP

    def mediapipe_to_joints(hand_landmarks, width, height):
        """Convert MediaPipe landmarks to trt_pose-compatible joint format.
        
        Maps MediaPipe indices to trt_pose_hand indices:
        trt_pose: 0=palm, 1=thumb_tip...4=thumb_base, 5=index_tip...8=index_base, etc.
        MediaPipe: 0=wrist, 4=thumb_tip, 8=index_tip, 12=middle_tip, 16=ring_tip, 20=pinky_tip
        """
        joints = []
        
        # Mapping from trt_pose index to MediaPipe index
        # trt_pose order: palm, thumb(tip->base), index(tip->base), middle, ring, pinky
        # MediaPipe order: wrist, thumb(base->tip), index(base->tip), etc.
        trt_to_mp = [
            0,   # 0: palm -> wrist
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
        
        for trt_idx in range(21):
            mp_idx = trt_to_mp[trt_idx]
            lm = hand_landmarks.landmark[mp_idx]
            x = int(lm.x * width)
            y = int(lm.y * height)
            joints.append([x, y])
        
        return joints

# ---------------------------------------------------------
# 3. 3D CUBE LOGIC
# ---------------------------------------------------------
def get_rotation_matrix(angle_x, angle_y, angle_z):
    # Rotation matrices for 3D space
    cx, sx = math.cos(angle_x), math.sin(angle_x)
    cy, sy = math.cos(angle_y), math.sin(angle_y)
    cz, sz = math.cos(angle_z), math.sin(angle_z)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return np.dot(Rz, np.dot(Ry, Rx))

def draw_line(stdscr, p1, p2, char='#'):
    # Bresenham's Line Algorithm for ASCII
    x0, y0 = p1
    x1, y1 = p2
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        try:
            stdscr.addch(y0, x0, char)
        except: pass # Clip if out of bounds

        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

# Define Cube Vertices (Centered at 0,0,0)
cube_vertices = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                          [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
# Define edges (indices of vertices)
cube_edges = [(0,1), (1,2), (2,3), (3,0),
              (4,5), (5,6), (6,7), (7,4),
              (0,4), (1,5), (2,6), (3,7)]

# ---------------------------------------------------------
# 4. HELPER FUNCTIONS FOR HAND TRACKING
# ---------------------------------------------------------
def get_joint_normalized(joints, idx):
    """Return joint as normalized coordinates (0-1), or None if invalid"""
    if idx < len(joints) and joints[idx] != [0, 0]:
        return (joints[idx][0] / WIDTH, joints[idx][1] / HEIGHT)
    return None

def calculate_pinch_distance(joints):
    """
    Calculate distance between thumb tip and index finger tip.
    Returns normalized distance (0-1 range approximately).
    """
    thumb_tip = get_joint_normalized(joints, 1)   # Thumb tip
    index_tip = get_joint_normalized(joints, 5)   # Index finger tip
    
    if thumb_tip is None or index_tip is None:
        return None
    
    dx = thumb_tip[0] - index_tip[0]
    dy = thumb_tip[1] - index_tip[1]
    dist = math.sqrt(dx*dx + dy*dy)
    return dist

# ---------------------------------------------------------
# 5. MAIN LOOP
# ---------------------------------------------------------
def main_loop(stdscr):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    curses.curs_set(0)
    stdscr.nodelay(True)

    # Cube state
    rows, cols = stdscr.getmaxyx()
    base_scale = min(rows, cols) // 6
    
    # Smoothed rotation values
    smooth_rot_x, smooth_rot_y, smooth_rot_z = 0.0, 0.0, 0.0
    smooth_scale = base_scale
    smoothing_factor = 0.3

    while True:
        ret, frame = cap.read()
        if not ret: continue

        # 1. MIRROR FRAME
        frame = cv2.flip(frame, 1)

        img_resized = cv2.resize(frame, (WIDTH, HEIGHT))
        
        # Get joints based on selected model
        if USE_MODEL == 'trt_pose':
            data = preprocess(img_resized)
            cmap, paf = model(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            counts, objects, peaks = parse_objects(cmap, paf)
            joints = preprocessdata.joints_inference(img_resized, counts, objects, peaks)
        elif USE_MODEL == 'mediapipe':
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                # Use the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                joints = mediapipe_to_joints(hand_landmarks, WIDTH, HEIGHT)
            else:
                joints = []

        # TUI Setup
        stdscr.clear()
        rows, cols = stdscr.getmaxyx()
        stdscr.addstr(0, 0, f"ASCII AR HAND CONTROL [{USE_MODEL}] | Tilt: Rotate | Pinch: Scale | 'q': Quit")

        center_x, center_y = cols // 2, rows // 2
        scale = smooth_scale

        # 2. CALCULATE HAND CONTROL
        # Joint indices for trt_pose_hand:
        # 0=palm
        # Thumb: 1=tip, 2, 3, 4=base
        # Index: 5=tip, 6, 7, 8=base
        # Middle: 9=tip, 10, 11, 12=base
        # Ring: 13=tip, 14, 15, 16=base
        # Pinky: 17=tip, 18, 19, 20=base
        if len(joints) > 0 and joints[0] != [0, 0]:
            # Get key landmarks (normalized 0-1)
            palm = get_joint_normalized(joints, 0)
            index_base = get_joint_normalized(joints, 8)    # Index finger base/MCP
            middle_base = get_joint_normalized(joints, 12)  # Middle finger base/MCP
            pinky_base = get_joint_normalized(joints, 20)   # Pinky base/MCP
            
            if palm and middle_base:
                # Calculate hand orientation for cube rotation (matching reference style)
                
                # X rotation: based on hand tilt (palm to middle finger Y difference)
                hand_tilt_y = middle_base[1] - palm[1]
                rotation_x = hand_tilt_y * math.pi * 2
                
                # Y rotation: based on hand horizontal position
                hand_center_x = (palm[0] + middle_base[0]) / 2
                rotation_y = (hand_center_x - 0.5) * math.pi * 2
                
                # Z rotation: based on hand roll (index to pinky angle)
                if index_base and pinky_base:
                    dx = pinky_base[0] - index_base[0]
                    dy = pinky_base[1] - index_base[1]
                    rotation_z = math.atan2(dy, dx)
                else:
                    rotation_z = smooth_rot_z  # Keep previous value
                
                # Smooth the rotations
                smooth_rot_x += (rotation_x - smooth_rot_x) * smoothing_factor
                smooth_rot_y += (rotation_y - smooth_rot_y) * smoothing_factor
                smooth_rot_z += (rotation_z - smooth_rot_z) * smoothing_factor
                
                # Calculate cube position based on hand center
                center_x = int(hand_center_x * cols)
                center_y = int(((palm[1] + middle_base[1]) / 2) * rows)
            
            # Scale: based on thumb-index pinch distance
            pinch_dist = calculate_pinch_distance(joints)
            if pinch_dist is not None:
                # Map pinch distance to scale
                # Close pinch ~0.05, spread apart ~0.3
                min_pinch = 0.03
                max_pinch = 0.25
                normalized_pinch = (pinch_dist - min_pinch) / (max_pinch - min_pinch)
                normalized_pinch = max(0.0, min(1.0, normalized_pinch))
                normalized_pinch = pinch_dist
                
                # Map to scale range
                min_scale = 3
                max_scale = min(rows, cols) // 2
                target_scale = min_scale + normalized_pinch * (max_scale - min_scale)
                smooth_scale += (target_scale - smooth_scale) * smoothing_factor
                scale = smooth_scale

            # Debug info
            stdscr.addstr(1, 0, f"Rot X:{math.degrees(smooth_rot_x):5.1f}° Y:{math.degrees(smooth_rot_y):5.1f}° Z:{math.degrees(smooth_rot_z):5.1f}° | Scale: {pinch_dist if pinch_dist else 0}")

        # 3. RENDER CUBE
        rot_matrix = get_rotation_matrix(smooth_rot_x, smooth_rot_y, smooth_rot_z)
        projected_points = []

        # Project Vertices
        for v in cube_vertices:
            rotated = np.dot(v, rot_matrix)
            # Simple Orthographic Projection with aspect ratio correction
            x = int(rotated[0] * scale * 2 + center_x)  # *2 for char aspect ratio
            y = int(rotated[1] * scale + center_y)
            projected_points.append((x, y))

        # Draw Edges
        for edge in cube_edges:
            p1 = projected_points[edge[0]]
            p2 = projected_points[edge[1]]
            draw_line(stdscr, p1, p2, '*')

        # 4. DRAW FINGER MARKERS FOR DEBUGGING
        # Finger tip indices: Thumb=1, Index=5, Middle=9, Ring=13, Pinky=17
        finger_tips = [1, 5, 9, 13, 17]
        finger_chars = ['T', 'I', 'M', 'R', 'P']  # Characters for each finger
        
        if len(joints) > 0:
            for i, tip_idx in enumerate(finger_tips):
                tip = get_joint_normalized(joints, tip_idx)
                if tip is not None:
                    fx = int(tip[0] * cols)
                    fy = int(tip[1] * rows)
                    # Clamp to screen bounds
                    fx = max(1, min(fx, cols - 2))
                    fy = max(2, min(fy, rows - 2))  # Start from row 2 to avoid header
                    try:
                        stdscr.addch(fy, fx, finger_chars[i])
                    except curses.error:
                        pass
            
            # Also draw palm marker
            palm = get_joint_normalized(joints, 0)
            if palm is not None:
                px = int(palm[0] * cols)
                py = int(palm[1] * rows)
                px = max(1, min(px, cols - 2))
                py = max(2, min(py, rows - 2))
                try:
                    stdscr.addch(py, px, 'O')
                except curses.error:
                    pass

        # Check Quit
        if stdscr.getch() == ord('q'): break
        stdscr.refresh()

    cap.release()
    if USE_MODEL == 'mediapipe':
        hands.close()

try:
    curses.wrapper(main_loop)
except Exception as e:
    print(f"Error: {e}")

