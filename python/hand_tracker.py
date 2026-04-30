#!/usr/bin/env python3
"""
hand_tracker.py — MediaPipe hand tracking sidecar.

Captures from /dev/video0, runs MediaPipe Hands (21 landmarks),
and writes results to POSIX shared memory via SharedMemWriter seqlock.

Joint order written directly as MediaPipe native indices (0-20):
  0=WRIST
  1=THUMB_CMC, 2=THUMB_MCP, 3=THUMB_IP,  4=THUMB_TIP
  5=INDEX_MCP, 6=INDEX_PIP, 7=INDEX_DIP,  8=INDEX_TIP
  9=MID_MCP,  10=MID_PIP,  11=MID_DIP,  12=MID_TIP
 13=RING_MCP, 14=RING_PIP, 15=RING_DIP,  16=RING_TIP
 17=PINKY_MCP,18=PINKY_PIP,19=PINKY_DIP, 20=PINKY_TIP
No remapping is applied — C++ reads these indices directly.
"""

import signal
import sys
import time

import cv2
import mediapipe as mp

from shared_mem_writer import SharedMemWriter

# Camera settings
CAM_WIDTH   = 224
CAM_HEIGHT  = 224
CAM_BUFFER  = 1   # small buffer = low latency
CAM_FPS     = 30  # cap frame rate to reduce camera firmware stress

CAM_RECONNECT_DELAY = 1.5   # seconds to wait before reconnect attempt
CAM_RECONNECT_TRIES = 10    # max consecutive reconnect attempts before exiting

# Shared memory retry parameters
SHM_RETRY_INTERVAL = 0.5   # seconds between retries
SHM_RETRY_TIMEOUT  = 15.0  # seconds to wait for C++ to create shm

# -----------------------------------------------------------------------
# Signal handling
# -----------------------------------------------------------------------
_running = True

def _sigterm(signum, frame):
    global _running
    _running = False

signal.signal(signal.SIGTERM, _sigterm)
signal.signal(signal.SIGINT,  _sigterm)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    global _running

    # 1. Wait for C++ to create the shared memory (it does so before fork)
    writer = SharedMemWriter()
    deadline = time.monotonic() + SHM_RETRY_TIMEOUT
    while time.monotonic() < deadline:
        if writer.open():
            print("hand_tracker: attached to shared memory", flush=True)
            break
        print("hand_tracker: waiting for shared memory...", flush=True)
        time.sleep(SHM_RETRY_INTERVAL)
    else:
        print("hand_tracker: timed out waiting for shared memory, exiting", flush=True)
        sys.exit(1)

    # 2. Open camera — try /dev/video0 and /dev/video1, use whichever opens
    def open_camera():
        for idx in (0, 1):
            c = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if c.isOpened():
                c.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
                c.set(cv2.CAP_PROP_BUFFERSIZE,   CAM_BUFFER)
                c.set(cv2.CAP_PROP_FPS,          CAM_FPS)
                print(f"hand_tracker: opened /dev/video{idx}", flush=True)
                return c
            c.release()
        return None

    cam_deadline = time.monotonic() + 30.0
    cap = None
    while cap is None and time.monotonic() < cam_deadline and _running:
        cap = open_camera()
        if cap is None:
            print("hand_tracker: waiting for camera...", flush=True)
            time.sleep(1.0)

    if cap is None:
        print("hand_tracker: could not open camera after 30 s, exiting", flush=True)
        writer.close()
        sys.exit(1)

    # 3. Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("hand_tracker: running", flush=True)

    # Pre-allocate zero arrays for when no hand is detected
    zero_lm = [0.0] * 21

    consecutive_failures = 0
    reconnect_attempts = 0

    while _running:
        ret, frame = cap.read()

        if not ret or frame is None:
            consecutive_failures += 1
            if consecutive_failures >= 10:
                if reconnect_attempts >= CAM_RECONNECT_TRIES:
                    print("hand_tracker: could not reconnect camera, exiting", flush=True)
                    break
                reconnect_attempts += 1
                print(f"hand_tracker: camera lost, reconnect attempt {reconnect_attempts}/{CAM_RECONNECT_TRIES}", flush=True)
                cap.release()
                time.sleep(CAM_RECONNECT_DELAY)
                new_cap = open_camera()
                if new_cap is not None:
                    cap = new_cap
                    consecutive_failures = 0
            else:
                time.sleep(0.05)  # avoid hammering V4L2 on transient errors
            continue
        consecutive_failures = 0
        reconnect_attempts = 0

        # Mirror horizontally (matches cube.py and 2d_send.py)
        frame = cv2.flip(frame, 1)

        # MediaPipe requires RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)
        t = time.monotonic()

        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0].landmark
            # Write MediaPipe native order directly — no remapping
            lm_x = [lms[i].x for i in range(21)]
            lm_y = [lms[i].y for i in range(21)]
            writer.write(lm_x, lm_y, hand_detected=True, timestamp=t)
        else:
            writer.write(zero_lm, zero_lm, hand_detected=False, timestamp=t)

    # 4. Clean shutdown
    hands.close()
    cap.release()
    writer.close()
    print("hand_tracker: exited cleanly", flush=True)


if __name__ == "__main__":
    main()
