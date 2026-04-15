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
CAM_INDEX   = 0
CAM_WIDTH   = 224
CAM_HEIGHT  = 224
CAM_BUFFER  = 1   # small buffer = low latency

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

    # 2. Open camera — retry for up to 30 s (device may appear after container starts)
    cap = None
    cam_deadline = time.monotonic() + 30.0
    while time.monotonic() < cam_deadline and _running:
        c = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        if c.isOpened():
            cap = c
            break
        c.release()
        print("hand_tracker: waiting for /dev/video0...", flush=True)
        time.sleep(1.0)

    if cap is None or not cap.isOpened():
        print("hand_tracker: could not open camera after 30 s, exiting", flush=True)
        writer.close()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   CAM_BUFFER)

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
    MAX_CONSECUTIVE_FAILURES = 10

    while _running:
        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"hand_tracker: camera read failed {MAX_CONSECUTIVE_FAILURES} times in a row, exiting", flush=True)
                break
            time.sleep(0.05)  # avoid hammering V4L2 on transient errors
            continue
        consecutive_failures = 0

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
