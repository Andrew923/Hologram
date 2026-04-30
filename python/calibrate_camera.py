#!/usr/bin/env python3
"""
calibrate_camera.py — OpenCV checkerboard camera calibration.

Captures frames from the webcam, finds checkerboard corners, and runs
cv2.calibrateCamera to produce intrinsic parameters.  Writes the result
to config/camera.json (merging into the existing file if present, so
user_index_bone_m set by calibrate_hand.py is preserved).

Usage:
    python3 python/calibrate_camera.py [options]

Options:
    --squares WxH      Inner corner count (default: 9x6)
    --size-mm N        Physical square edge length in mm (default: 25)
    --frames  N        How many checkerboard frames to capture (default: 20)
    --device  N        Video device index (default: 0)
    --output  PATH     Output JSON path (default: config/camera.json)

Hold a printed checkerboard pattern in front of the camera; the script
captures a frame each time all corners are detected.  When enough frames
are collected, calibration runs automatically and the result is saved.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Camera intrinsic calibration")
    p.add_argument("--squares", default="9x6",
                   help="Inner corner count WxH (default: 9x6)")
    p.add_argument("--size-mm", type=float, default=25.0,
                   help="Square side length in mm (default: 25)")
    p.add_argument("--frames", type=int, default=20,
                   help="Number of frames to capture (default: 20)")
    p.add_argument("--device", type=int, default=0,
                   help="Video capture device index (default: 0)")
    p.add_argument("--output", default="config/camera.json",
                   help="Output JSON path (default: config/camera.json)")
    p.add_argument("--headless", action="store_true",
                   help="Run without display (no cv2.imshow)")
    return p.parse_args()


def main():
    args = parse_args()

    w, h = [int(x) for x in args.squares.split("x")]
    board_size = (w, h)
    square_mm = args.size_mm
    target_frames = args.frames

    # Prepare object points for one board (Z=0 plane).
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2) * square_mm

    obj_points = []   # 3D world points per frame
    img_points = []   # 2D image points per frame

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"ERROR: cannot open video device {args.device}", file=sys.stderr)
        sys.exit(1)

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {img_w}x{img_h}  Board: {w}x{h}  "
          f"Square: {square_mm:.1f}mm  Target: {target_frames} frames")
    if args.headless:
        print("Headless mode: auto-capturing, no display.")
    else:
        print("Show the checkerboard to the camera. "
              "Press 'q' to abort, 's' to skip a frame.")

    last_capture = 0.0
    while len(obj_points) < target_frames:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        vis = frame.copy()
        if found:
            cv2.drawChessboardCorners(vis, board_size, corners, found)

            # Auto-capture at most once per second to get varied viewpoints.
            now = time.time()
            if now - last_capture >= 1.0:
                refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                obj_points.append(objp)
                img_points.append(refined)
                last_capture = now
                n = len(obj_points)
                print(f"  Captured {n}/{target_frames}")

        if not args.headless:
            status = (f"Captured: {len(obj_points)}/{target_frames}  "
                      f"{'[FOUND]' if found else '[searching...]'}")
            cv2.putText(vis, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Calibration", vis)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("Aborted.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(1)

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()

    print(f"\nRunning calibrateCamera on {len(obj_points)} frames ...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (img_w, img_h), None, None)
    print(f"  RMS reprojection error: {ret:.4f}")
    print(f"  fx={mtx[0,0]:.2f}  fy={mtx[1,1]:.2f}  "
          f"cx={mtx[0,2]:.2f}  cy={mtx[1,2]:.2f}")
    print(f"  dist_coeffs={dist.ravel().tolist()}")

    # Merge with existing config (preserve user_index_bone_m if present).
    existing = {}
    if os.path.isfile(args.output):
        with open(args.output) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                pass

    existing["image_width"]  = img_w
    existing["image_height"] = img_h
    existing["fx"] = float(mtx[0, 0])
    existing["fy"] = float(mtx[1, 1])
    existing["cx"] = float(mtx[0, 2])
    existing["cy"] = float(mtx[1, 2])
    existing["dist_coeffs"] = [float(d) for d in dist.ravel()[:5]]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
