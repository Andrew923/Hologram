#!/usr/bin/env python3
"""
calibrate_hand.py — Measure the user's index bone length for depth
                     unprojection.

Opens the camera, runs MediaPipe Hands to find the wrist (0) → index
MCP (5) segment, and asks the user to hold their palm flat at a known
distance.  From the observed pixel length and the camera focal length
(loaded from config/camera.json), it computes the physical bone length
and writes `user_index_bone_m` into the same file.

Usage:
    python3 python/calibrate_hand.py [options]

Options:
    --config  PATH     camera.json path (default: config/camera.json)
    --distance-m N     Distance from camera to palm in meters (default: 0.50)
    --device  N        Video device index (default: 0)
    --samples N        How many measurements to average (default: 30)

Instructions:
  1. Run calibrate_camera.py first so config/camera.json has fx/fy.
  2. Measure the distance from the camera lens to where you'll hold
     your palm (e.g. arm's length, ~50 cm).
  3. Hold your open palm facing the camera at that distance. Keep it
     steady until the script collects enough samples.
"""

import argparse
import json
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Hand bone-length calibration")
    p.add_argument("--config", default="config/camera.json")
    p.add_argument("--distance-m", type=float, default=0.50,
                   help="Known camera-to-palm distance in meters (default: 0.50)")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--samples", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()

    # Load camera intrinsics.
    if not os.path.isfile(args.config):
        print(f"ERROR: {args.config} not found. "
              "Run calibrate_camera.py first.", file=sys.stderr)
        sys.exit(1)

    with open(args.config) as f:
        cfg = json.load(f)

    fx = cfg.get("fx")
    if not fx or fx <= 0:
        print("ERROR: fx missing or invalid in config.", file=sys.stderr)
        sys.exit(1)

    img_w = cfg.get("image_width", 640)
    img_h = cfg.get("image_height", 480)

    print(f"Camera: {img_w}x{img_h}  fx={fx:.1f}")
    print(f"Hold your open palm {args.distance_m:.2f}m from the camera.")
    print("Press 'q' to abort.\n")

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"ERROR: cannot open device {args.device}", file=sys.stderr)
        sys.exit(1)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    measurements = []

    while len(measurements) < args.samples:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        vis = frame.copy()
        if result.multi_hand_landmarks:
            lms = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(vis, lms, mp_hands.HAND_CONNECTIONS)

            wrist = lms.landmark[0]
            idx_mcp = lms.landmark[5]
            px_wrist = np.array([wrist.x * img_w, wrist.y * img_h])
            px_mcp   = np.array([idx_mcp.x * img_w, idx_mcp.y * img_h])
            L_px = float(np.linalg.norm(px_mcp - px_wrist))

            if L_px > 5.0:
                measurements.append(L_px)
                n = len(measurements)
                if n % 5 == 0:
                    avg = np.mean(measurements)
                    print(f"  Sample {n}/{args.samples}  avg bone px = {avg:.1f}")

        status = f"Samples: {len(measurements)}/{args.samples}"
        cv2.putText(vis, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Hand Calibration", vis)

        if (cv2.waitKey(30) & 0xFF) == ord('q'):
            print("Aborted.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(1)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    avg_px = float(np.mean(measurements))
    # L_ref = Z_known * L_px / f
    bone_m = args.distance_m * avg_px / fx
    print(f"\n  Average bone pixel length: {avg_px:.1f} px")
    print(f"  At distance {args.distance_m:.2f} m with fx={fx:.1f}")
    print(f"  → user_index_bone_m = {bone_m:.4f} m")

    cfg["user_index_bone_m"] = round(bone_m, 4)
    with open(args.config, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nSaved to {args.config}")


if __name__ == "__main__":
    main()
