import cv2
import numpy as np
import struct
import time

from control import DirectUDP, EphemeralWiFiUDP
from view import DepthRenderer

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

# Network Mode: Set to True to use ESP32 SoftAP, False for direct LAN connection
USE_EPHEMERAL_WIFI = False

# Direct LAN settings (when USE_EPHEMERAL_WIFI = False)
DIRECT_ESP_IP = "192.168.1.228"

# ESP32 SoftAP settings (when USE_EPHEMERAL_WIFI = True)
SOFTAP_ESP_IP = "192.168.4.1"
WIFI_SSID = "Hologram"
WIFI_PASS = "andrew923"

# Common settings
ESP_PORT = 4210

# Options: 0 = RLE (Column-Major Delta), 1 = Sparse Voxel List
COMPRESSION_MODE = 0

# Volume Config
NUM_SLICES = 120
WIDTH = 64
HEIGHT = 32

# ---------------------------------------------------------
# 2. COMPRESSION LOGIC
# ---------------------------------------------------------
def to_rgb332(img):
    """Downsamples BGR image to 8-bit RGB332"""
    # Img is (H, W, 3)
    b, g, r = cv2.split(img)
    r = (r >> 5).astype(np.uint8) << 5
    g = (g >> 5).astype(np.uint8) << 2
    b = (b >> 6).astype(np.uint8)
    return r | g | b

def compress_sparse(rgb332_flat):
    """Protocol 1: [X, Y, Color, ...]"""
    payload = bytearray()
    # Iterate linear, convert to X,Y
    for i, color in enumerate(rgb332_flat):
        if color != 0: # Only send non-black pixels
            x = i % WIDTH
            y = i // WIDTH
            payload.append(x)
            payload.append(y)
            payload.append(color)
    return payload

def compress_rle_column_major(rgb332_img):
    """Protocol 0: [Count, Color, Count, Color...] (Column Major)"""
    payload = bytearray()
    
    # 1. Transpose for Column-Major Scanning (scan down columns)
    # This keeps vertical lines (like a standing human) contiguous
    img_T = cv2.transpose(rgb332_img)
    pixels = img_T.flatten()
    
    # 2. RLE Encoding
    count = 0
    current_val = int(pixels[0])
    
    for px in pixels:
        if px == current_val and count < 255:
            count += 1
        else:
            payload.append(count)
            payload.append(current_val)
            current_val = int(px)
            count = 1
    # Final run
    payload.append(count)
    payload.append(current_val)
    
    return payload

# ---------------------------------------------------------
# 3. MAIN LOOP
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # Create transport based on configuration
    if USE_EPHEMERAL_WIFI:
        transport = EphemeralWiFiUDP(
            host=SOFTAP_ESP_IP,
            port=ESP_PORT,
            wifi_ssid=WIFI_SSID,
            wifi_password=WIFI_PASS,
            hidden=True
        )
    else:
        transport = DirectUDP(host=DIRECT_ESP_IP, port=ESP_PORT)
    
    # Create renderer
    renderer = renderer = DepthRenderer(
        NUM_SLICES, WIDTH, HEIGHT,
        camera_src=0,
        encoder='vits',
        input_size=196,
        depth_scale=0.4,       # How much depth affects displacement
        depth_threshold=0.7,   # Background removal (0-1)
        oscillation_cycles=0.5, # How many L-R cycles per rotation
        max_angle_deg=45.0     # Maximum angle ±
    )
    
    # Connect and run
    with transport:
        print(f"[*] Starting Stream to {transport.host}:{transport.port}")
        print(f"[*] Mode: {'SPARSE' if COMPRESSION_MODE else 'RLE (Column-Major)'}")
        print(f"[*] Network: {'EphemeralWiFi' if USE_EPHEMERAL_WIFI else 'Direct LAN'}")
        print(f"[*] Renderer: {renderer.__class__.__name__}")
        
        try:
            while True:
                # We simulate one full rotation (frame 0 to 119)
                # In a real app, this would be tied to Input/Game Logic
                
                for slice_id in range(NUM_SLICES):
                    
                    # 1. Generate Slice
                    frame_bgr = renderer.render_slice(slice_id)
                    frame_332 = to_rgb332(frame_bgr)
                    
                    # 2. Compress
                    payload = b''
                    
                    if COMPRESSION_MODE == 1:
                        # Sparse Mode
                        flat = frame_332.flatten()
                        data = compress_sparse(flat)
                        payload = data
                    else:
                        # RLE Mode (Column-Major)
                        data = compress_rle_column_major(frame_332)
                        payload = data
                        
                    # 4. Construct Header (2 Bytes)
                    # Byte 0: [Proto(1) | Reserved(7)]
                    # Byte 1: SliceID
                    
                    flags = 0
                    if COMPRESSION_MODE == 1: flags |= 0x80 # Bit 7
                    
                    header = struct.pack('BB', flags, slice_id)
                    packet = header + payload
                    
                    if len(packet) > 1400:
                        print(f"WARNING: Packet size {len(packet)} bytes exceeds MTU!")
                        
                    # 5. Send
                    try:
                        transport.send(packet)
                    except Exception as e:
                        print(f"Send Error: {e}")

                    # 6. Throttle
                    # A full rotation takes ~66ms (900RPM). 
                    # 66ms / 120 slices = 0.5ms per slice.
                    # We go slightly faster to fill buffer.
                    time.sleep(0.0002) 

                # Optional: print status every rotation
                # print(".", end="", flush=True)

        except KeyboardInterrupt:
            print("\n[*] Stopping...")

    print("[*] Done.")