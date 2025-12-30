import os
# Set pure-Python protobuf implementation for compatibility with locally built mediapipe
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import json
import torch
import trt_pose.coco
import trt_pose.models
from trt_pose.parse_objects import ParseObjects
import torchvision.transforms as transforms
import PIL.Image
import curses
import sys
import time
import mediapipe as mp

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
WIDTH = 224
HEIGHT = 224
# Make sure this matches your actual path
MODEL_WEIGHTS = '/data/trt_pose_hand/model/hand_pose_resnet18_att_244_244.pth'
HAND_POSE_JSON = '/data/trt_pose_hand/preprocess/hand_pose.json'

# Model selection: 'trt_pose' or 'mediapipe'
MODELS = ['trt_pose', 'mediapipe']

# ---------------------------------------------------------
# 2. TRT_POSE MODEL SETUP
# ---------------------------------------------------------
print("Initializing TRT_Pose Model... (Please wait)")
with open(HAND_POSE_JSON, 'r') as f:
    hand_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(hand_pose)
num_parts = len(hand_pose['keypoints'])
num_links = len(hand_pose['skeleton'])

# Load Raw PyTorch model (Skipping TRT for compatibility/speed of setup)
trt_model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
trt_model.load_state_dict(torch.load(MODEL_WEIGHTS))
print("TRT_Pose Model Loaded!")

# ---------------------------------------------------------
# 3. MEDIAPIPE MODEL SETUP
# ---------------------------------------------------------
print("Initializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
mediapipe_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,  # 0=Lite, 1=Full
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("MediaPipe Hands Loaded!")

# ---------------------------------------------------------
# 4. DATA PREPROCESSING (TRT_POSE)
# ---------------------------------------------------------
# Data Preprocessing tools
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
        K = self.topology.shape[0]
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

# ---------------------------------------------------------
# 5. INFERENCE FUNCTIONS
# ---------------------------------------------------------
def run_trt_pose_inference(frame, img_resized):
    """Run TRT_Pose hand inference and return joints list."""
    data = preprocess(img_resized)
    cmap, paf = trt_model(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)
    joints = preprocessdata.joints_inference(img_resized, counts, objects, peaks)
    return joints

def run_mediapipe_inference(frame, img_resized):
    """Run MediaPipe hand inference and return joints list."""
    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    results = mediapipe_model.process(rgb_frame)
    
    joints = []
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * WIDTH)
            y = int(landmark.y * HEIGHT)
            joints.append([x, y])
    return joints

# ---------------------------------------------------------
# 6. CURSES TUI LOOP
# ---------------------------------------------------------
def main_loop(stdscr):
    # Setup Camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    # Setup Curses
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(True) # Don't block waiting for keypress
    stdscr.clear()

    # Model selection
    current_model_idx = 0
    
    # FPS tracking
    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()

    while True:
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            stdscr.addstr(0, 0, "Error: Camera read failed.")
            stdscr.refresh()
            continue

        # Resize frame
        img_resized = cv2.resize(frame, (WIDTH, HEIGHT))
        
        # Inference based on selected model
        current_model = MODELS[current_model_idx]
        if current_model == 'trt_pose':
            joints = run_trt_pose_inference(frame, img_resized)
        else:  # mediapipe
            joints = run_mediapipe_inference(frame, img_resized)
        
        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start_time = time.time()

        # ------------------ DRAW TUI ------------------
        stdscr.clear()

        # 1. Get Terminal Dimensions
        rows, cols = stdscr.getmaxyx()

        # 2. Draw Header with FPS and Model info
        header = f"HAND TRACKER [{current_model.upper()}] | FPS: {fps:.1f} | 'm'=switch model, 'q'=quit | {WIDTH}x{HEIGHT}"
        stdscr.addstr(0, 0, header[:cols-1])

        # 3. Draw Hand Points
        if len(joints) > 0:
            wrist = joints[0]
            if wrist != [0,0]:
                stdscr.addstr(1, 0, f"HAND DETECTED! Wrist X: {wrist[0]} Y: {wrist[1]} | Total joints: {len(joints)}")

            # Draw ASCII representation of the hand
            # Map 224x224 image coordinates to Terminal Rows/Cols
            # We reserve rows 3+ for drawing
            draw_height = rows - 4
            draw_width = cols - 2

            for idx, j in enumerate(joints):
                if j == [0,0]: continue

                # Normalize (0-1) then scale to terminal
                term_x = int((j[0] / WIDTH) * draw_width)
                term_y = int((j[1] / HEIGHT) * draw_height) + 3 # +3 to skip header

                # Bounds check
                if 0 <= term_y < rows and 0 <= term_x < cols - 2:
                    try:
                        # Display joint number instead of 'O'
                        label = str(idx)
                        stdscr.addstr(term_y, term_x, label)
                    except:
                        pass # Ignore edge cases
        else:
            stdscr.addstr(1, 0, "No Hand Detected...")

        # 4. Check for Quit or Model Toggle
        key = stdscr.getch()
        if key == ord('q'):
            break
        elif key == ord('m'):
            current_model_idx = (current_model_idx + 1) % len(MODELS)

        stdscr.refresh()

    cap.release()
    mediapipe_model.close()

# Wrapper to handle cleanup automatically
try:
    curses.wrapper(main_loop)
except Exception as e:
    print(f"Crashed: {e}")
