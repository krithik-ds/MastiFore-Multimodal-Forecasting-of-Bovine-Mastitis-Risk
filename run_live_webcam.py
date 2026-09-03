#!/usr/bin/env python3
"""
MastiFore Live Fullscreen Computer Vision Runner
SIH 26109: Bovine Mastitis Early Forecasting System

Features:
- YOLOv8 + ByteTrack ID persistence
- Differential Relative Jaw Motion tracking (cancels hand/camera shake on static photos)
- Spine kyphosis posture measurement (Sprecher pain index)
- Ground-clearance / Leg pillar posture classifier (Standing vs Resting)
- Press 'f' key for true native edge-to-edge fullscreen
- Press 'q' or 'ESC' to exit
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import time
from cv_module.cow_vision_triage import CowVisionTriageEngine

WINDOW_TITLE = "MastiFore - AI Cattle Vision Triage [Press 'f' Fullscreen | 'q' Exit]"


def main():
    print("=" * 70)
    print(" MastiFore: Live Barn CCTV Vision Screening (SIH 26109)")
    print("=" * 70)
    print(" • Press 'f' to toggle True Edge-to-Edge Native Fullscreen")
    print(" • Press 'q' or 'ESC' to exit")
    print("=" * 70)

    # Initialize CV Triage Engine
    engine = CowVisionTriageEngine(conf_threshold=0.25)

    # Open Camera with macOS AVFoundation backend
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open camera device 0.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Create Resizable Window
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

    is_fullscreen = False

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Run MastiFore Computer Vision Triage Engine
        annotated_frame, diagnostics = engine.analyze_frame(frame)

        # Show Output
        cv2.imshow(WINDOW_TITLE, annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

    cap.release()
    cv2.destroyAllWindows()
    print("[MastiFore] Vision stream closed cleanly.")


if __name__ == "__main__":
    main()
