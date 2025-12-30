import cv2
import math
import numpy as np
import curses
import time

from control import FastVideoCapture, TRTPoseHandDetector, MediaPipeHandDetector, DirectUDP

# ---------------------------------------------------------
# ONE EURO FILTER (for smooth, low-latency filtering)
# ---------------------------------------------------------
class OneEuroFilter:
    """
    One Euro Filter for reducing jitter while maintaining responsiveness.
    - min_cutoff: Minimum cutoff frequency. Lower = more smoothing, higher latency.
    - beta: Speed coefficient. Higher = less lag during fast movements.
    - d_cutoff: Cutoff frequency for derivative estimation.
    """
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
    
    def _smoothing_factor(self, cutoff, dt):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)
    
    def _exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev
    
    def __call__(self, x, t=None):
        if t is None:
            t = time.time()
        
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        
        dt = t - self.t_prev
        if dt <= 0:
            dt = 1e-6  # Avoid division by zero
        
        # Estimate derivative
        dx = (x - self.x_prev) / dt
        
        # Smooth the derivative
        a_d = self._smoothing_factor(self.d_cutoff, dt)
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)
        
        # Adaptive cutoff based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Smooth the signal
        a = self._smoothing_factor(cutoff, dt)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)
        
        # Store for next iteration
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat
    
    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

# ---------------------------------------------------------
# NETWORK CONFIGURATION
# ---------------------------------------------------------
ESP_IP = "192.168.1.228"
ESP_PORT = 4210

# To use ephemeral WiFi (ESP32 SoftAP), change to:
# from control import EphemeralWiFiUDP
# transport = EphemeralWiFiUDP(ESP_IP, ESP_PORT, wifi_ssid="Hologram", wifi_password="andrew923")
transport = DirectUDP(ESP_IP, ESP_PORT)

# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------
WIDTH = 64   # Matrix Width
HEIGHT = 32  # Matrix Height

# MODEL SELECTION: 'trt_pose' or 'mediapipe' (can switch at runtime with 'm' key)
LOAD_BOTH_MODELS = True  # Set False to save memory (only loads active_model)
active_model = 'trt_pose'  # Starting model

# ---------------------------------------------------------
# RLE COMPRESSION LOGIC
# ---------------------------------------------------------
def compress_frame_rle(frame, frame_id):
    """
    Takes a 64x32 BGR frame, downsamples to RGB332, and RLE compresses it.
    Prepends a 1-byte frame ID for packet ordering.
    Returns: Bytes object with format [Frame_ID, RLE_Data...]
    """
    # Start with frame ID header
    compressed = bytearray([frame_id & 0xFF])
    
    # 1. Convert BGR to RGB332 (8-bit color)
    # R (3 bits) | G (3 bits) | B (2 bits)
    # We use bit shifting to quantize
    b, g, r = cv2.split(frame)
    r = (r >> 5).astype(np.uint8) << 5  # Keep top 3 bits, shift to pos 7-5
    g = (g >> 5).astype(np.uint8) << 2  # Keep top 3 bits, shift to pos 4-2
    b = (b >> 6).astype(np.uint8)       # Keep top 2 bits, shift to pos 1-0
    rgb332 = r | g | b 
    
    # Flatten to 1D array
    pixels = rgb332.flatten()
    
    # 2. RLE Compression
    # Logic: Loop through pixels, count identical neighbors
    # Output format: [Frame_ID, Count, Color, Count, Color...]
    if len(pixels) == 0: return compressed
    
    count = 0
    current_color = pixels[0]
    
    for pixel in pixels:
        if pixel == current_color and count < 255:
            count += 1
        else:
            compressed.append(count)
            compressed.append(current_color)
            current_color = pixel
            count = 1
            
    # Append last run
    compressed.append(count)
    compressed.append(current_color)
    
    return compressed

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
def main_loop(stdscr):
    global active_model
    
    # Connect network transport
    if not transport.connect():
        raise ConnectionError(f"Failed to connect to {transport.host}:{transport.port}")
    
    # Initialize shared camera
    cap = FastVideoCapture(0, flip_horizontal=True)
    
    # Initialize detectors (they share the camera)
    trt_detector = None
    mediapipe_detector = None
    trt_pose_available = False
    mediapipe_available = False
    
    if LOAD_BOTH_MODELS or active_model == 'trt_pose':
        trt_detector = TRTPoseHandDetector(camera=cap)
        trt_pose_available = trt_detector.load_model()
        if trt_pose_available:
            print("TRT_POSE loaded successfully")
    
    if LOAD_BOTH_MODELS or active_model == 'mediapipe':
        mediapipe_detector = MediaPipeHandDetector(camera=cap)
        mediapipe_available = mediapipe_detector.load_model()
        if mediapipe_available:
            print("MediaPipe loaded successfully")

    curses.curs_set(0)
    stdscr.nodelay(True)
    
    # Initialize colors for curses
    curses.start_color()
    curses.use_default_colors()
    # Define color pairs for hue visualization
    curses.init_pair(1, curses.COLOR_RED, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_GREEN, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)
    curses.init_pair(5, curses.COLOR_BLUE, -1)
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)

    # One Euro Filters for each joint (21 joints x 2 coordinates)
    joint_filters_x = [OneEuroFilter(min_cutoff=1.0, beta=0.5, d_cutoff=1.0) for _ in range(21)]
    joint_filters_y = [OneEuroFilter(min_cutoff=1.0, beta=0.5, d_cutoff=1.0) for _ in range(21)]
    
    # Smoothed joints storage
    smooth_joints = [[0, 0] for _ in range(21)]
    
    # Debug view toggle
    show_debug_view = True
    
    # Frame ID counter (0-255, wraps around)
    frame_id = 0
    
    # Hand skeleton connections from detector class
    HAND_CONNECTIONS = TRTPoseHandDetector.HAND_CONNECTIONS
    
    # Network stats
    bytes_sent = 0
    packets_sent = 0
    start_time = time.time()
    last_stats_time = start_time
    bytes_per_second = 0
    packets_per_second = 0
    fps = 0
    frame_count = 0
    inference_time_ms = 0

    while True:
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret or frame is None: 
            continue
        
        # Get joints based on selected model
        inference_start = time.time()
        joints = []
        
        if active_model == 'trt_pose' and trt_pose_available:
            joints = trt_detector.detect(frame)
        elif active_model == 'mediapipe' and mediapipe_available:
            joints = mediapipe_detector.detect(frame)
        
        inference_time_ms = (time.time() - inference_start) * 1000
        
        # Create the "Canvas" for the matrix (64x32)
        matrix_canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        
        # TUI Setup
        stdscr.clear()
        rows, cols = stdscr.getmaxyx()
        debug_status = "ON" if show_debug_view else "OFF"
        model_status = f"{active_model}"
        if not ((active_model == 'trt_pose' and trt_pose_available) or (active_model == 'mediapipe' and mediapipe_available)):
            model_status += " (N/A)"
        stdscr.addstr(0, 0, f"2D HAND [{model_status}] | Debug:{debug_status}(SPACE) | Switch(m) | 'q':Quit")
        
        hand_detected = False
        num_valid_joints = 0
        
        if len(joints) >= 21:
            wrist = joints[0]
            
            # Check if wrist is valid
            if wrist[0] != 0 or wrist[1] != 0:
                hand_detected = True
                current_time = time.time()
                
                # Apply One Euro Filter to each joint and scale to matrix size
                for i in range(21):
                    if joints[i][0] != 0 or joints[i][1] != 0:
                        # Scale to matrix coordinates
                        target_x = joints[i][0] * WIDTH
                        target_y = joints[i][1] * HEIGHT
                        
                        # Apply smoothing
                        smooth_joints[i][0] = int(joint_filters_x[i](target_x, current_time))
                        smooth_joints[i][1] = int(joint_filters_y[i](target_y, current_time))
                        num_valid_joints += 1
                
                # Draw hand skeleton on canvas
                # Color: white for bones
                bone_color = (255, 255, 255)
                joint_color = (0, 255, 0)  # Green for joints
                tip_color = (0, 0, 255)    # Red for fingertips
                
                # Draw connections (bones)
                for conn in HAND_CONNECTIONS:
                    j1, j2 = conn
                    p1 = (smooth_joints[j1][0], smooth_joints[j1][1])
                    p2 = (smooth_joints[j2][0], smooth_joints[j2][1])
                    # Only draw if both points are valid (non-zero)
                    if (p1[0] != 0 or p1[1] != 0) and (p2[0] != 0 or p2[1] != 0):
                        cv2.line(matrix_canvas, p1, p2, bone_color, 1)
                
                # Draw joints (small circles)
                fingertip_indices = [1, 5, 9, 13, 17]  # Thumb, Index, Middle, Ring, Pinky tips
                for i in range(21):
                    x, y = smooth_joints[i]
                    if x != 0 or y != 0:
                        # Use different color for fingertips
                        color = tip_color if i in fingertip_indices else joint_color
                        # Draw as single pixel or small dot
                        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                            matrix_canvas[y, x] = color

        # Compress and Send with Frame ID
        payload = compress_frame_rle(matrix_canvas, frame_id)
        payload_size = len(payload)
        
        net_error = None
        try:
            # Send packet (ESP32 will use frame_id to ignore out-of-order packets)
            transport.send(payload)
            bytes_sent += payload_size
            packets_sent += 1
        except Exception as e:
            net_error = str(e)
        
        # Increment frame ID (wrap at 255)
        frame_id = (frame_id + 1) & 0xFF
        
        # Update network stats every second
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - last_stats_time
        if elapsed >= 1.0:
            bytes_per_second = bytes_sent / elapsed
            packets_per_second = packets_sent / elapsed
            fps = frame_count / elapsed
            bytes_sent = 0
            packets_sent = 0
            frame_count = 0
            last_stats_time = current_time

        # Debug info - Line 1: Hand status
        status = "HAND" if hand_detected else "----"
        wrist_pos = f"({smooth_joints[0][0]:2d},{smooth_joints[0][1]:2d})" if hand_detected else "(--,--)"
        stdscr.addstr(1, 0, f"{status} | Wrist:{wrist_pos} Joints:{num_valid_joints}/21")
        
        # Debug info - Line 2: Network & performance stats
        net_stats = f"NET: {bytes_per_second/1024:.1f}KB/s {packets_per_second:.0f}pkt/s | FPS:{fps:.1f} Inf:{inference_time_ms:.1f}ms | Pkt:{payload_size}B FID:{frame_id}"
        if net_error:
            net_stats = f"NET ERROR: {net_error}"
        stdscr.addstr(2, 0, net_stats[:cols-1])
        
        # Draw ASCII representation of the matrix canvas (if enabled)
        if show_debug_view:
            preview_start_row = 4
            scale_x = 2  # chars per pixel width
            scale_y = 1  # chars per pixel height
            
            # Use green color for hand skeleton display
            color_pair = 3  # Green
            
            # Draw border
            border_width = WIDTH * scale_x + 2
            if preview_start_row < rows and border_width < cols:
                stdscr.addstr(preview_start_row, 0, "+" + "-" * (WIDTH * scale_x) + "+")
            
            for y in range(HEIGHT):
                row_y = preview_start_row + 1 + y * scale_y
                if row_y >= rows - 1:
                    break
                
                line = "|"
                for x in range(WIDTH):
                    # Check if this pixel is lit (any channel > 0)
                    if matrix_canvas[y, x].any():
                        line += "██"
                    else:
                        line += "  "
                line += "|"
                
                if len(line) < cols:
                    try:
                        if matrix_canvas[y].any():
                            stdscr.addstr(row_y, 0, line, curses.color_pair(color_pair))
                        else:
                            stdscr.addstr(row_y, 0, line)
                    except curses.error:
                        pass
            
            # Bottom border
            bottom_row = preview_start_row + 1 + HEIGHT * scale_y
            if bottom_row < rows and border_width < cols:
                try:
                    stdscr.addstr(bottom_row, 0, "+" + "-" * (WIDTH * scale_x) + "+")
                except curses.error:
                    pass

        # Check input
        key = stdscr.getch()
        if key == ord('q'):
            break
        elif key == ord(' '):
            show_debug_view = not show_debug_view
        elif key == ord('m'):
            # Toggle between models
            if active_model == 'trt_pose' and mediapipe_available:
                active_model = 'mediapipe'
            elif active_model == 'mediapipe' and trt_pose_available:
                active_model = 'trt_pose'
            
        stdscr.refresh()

    cap.release()
    if mediapipe_detector is not None:
        mediapipe_detector.release()
    transport.disconnect()

try:
    curses.wrapper(main_loop)
except Exception as e:
    print(f"Error: {e}")