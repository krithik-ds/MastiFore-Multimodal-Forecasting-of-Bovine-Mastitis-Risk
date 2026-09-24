"""
Synthetic Demo Video Generator for Cow Triage Testing
Generates a realistic test video (.mp4) with 2 simulated cows:
- Cow 1: Healthy cow (Flat spine 174°, active rumination chewing)
- Cow 2: Subclinical Mastitis Suspect cow (Arched spine 146°, depressed chewing)
"""

import os
import math
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def generate_sih_demo_video(output_path="demo_cows.mp4", duration_sec=6, fps=25):
    if cv2 is None:
        print("[Error] OpenCV is required. Please install it using 'pip install opencv-python'.")
        return

    w, h = 1280, 720
    total_frames = int(duration_sec * fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"[DemoGenerator] Rendering {total_frames} frames of simulated cattle barn video to {output_path}...")

    for frame_i in range(total_frames):
        t = frame_i / float(fps)
        # Barn background
        frame = np.full((h, w, 3), (210, 225, 230), dtype=np.uint8)
        # Floor (hay/barn floor)
        cv2.rectangle(frame, (0, int(h * 0.65)), (w, h), (140, 175, 190), -1)

        # ----------------- Cow 1: Healthy (Left Side) -----------------
        c1_x, c1_y = 150, 280
        # Body (flat top spine)
        cv2.ellipse(frame, (c1_x + 180, c1_y + 120), (140, 85), 0, 0, 360, (230, 230, 230), -1)
        # Flat spine line
        pts1 = np.array([
            [c1_x + 50, c1_y + 50],
            [c1_x + 180, c1_y + 48],
            [c1_x + 300, c1_y + 52],
            [c1_x + 300, c1_y + 160],
            [c1_x + 50, c1_y + 160]
        ], np.int32)
        cv2.fillPoly(frame, [pts1], (40, 40, 40))
        # Head & Muzzle (Rhythmic chewing motion)
        jaw_osc1 = int(math.sin(t * 7.0) * 6)  # Active chewing ~65 chews/min
        cv2.circle(frame, (c1_x + 30, c1_y + 80), 38, (45, 45, 45), -1)
        cv2.ellipse(frame, (c1_x + 10, c1_y + 100 + jaw_osc1), (22, 16), 0, 0, 360, (180, 180, 200), -1)
        # Legs
        cv2.rectangle(frame, (c1_x + 70, c1_y + 140), (c1_x + 95, c1_y + 260), (30, 30, 30), -1)
        cv2.rectangle(frame, (c1_x + 250, c1_y + 140), (c1_x + 275, c1_y + 260), (30, 30, 30), -1)
        # Udder
        cv2.ellipse(frame, (c1_x + 210, c1_y + 180), (25, 20), 0, 0, 360, (180, 190, 220), -1)

        # ----------------- Cow 2: Mastitis Suspect (Right Side) -----------------
        c2_x, c2_y = 700, 270
        # Arched spine (kyphosis upward arch)
        pts2 = np.array([
            [c2_x + 60, c2_y + 80],
            [c2_x + 180, c2_y + 20],  # Arched upwards!
            [c2_x + 310, c2_y + 85],
            [c2_x + 310, c2_y + 170],
            [c2_x + 60, c2_y + 170]
        ], np.int32)
        cv2.fillPoly(frame, [pts2], (120, 80, 50))
        # Body lower
        cv2.ellipse(frame, (c2_x + 185, c2_y + 130), (135, 75), 0, 0, 360, (140, 95, 60), -1)
        # Head (Depressed / lethargic, minimal jaw motion)
        jaw_osc2 = int(math.sin(t * 1.5) * 1)  # Low rumination ~20 chews/min
        cv2.circle(frame, (c2_x + 40, c2_y + 100), 38, (120, 80, 50), -1)
        cv2.ellipse(frame, (c2_x + 20, c2_y + 120 + jaw_osc2), (20, 14), 0, 0, 360, (180, 160, 160), -1)
        # Legs
        cv2.rectangle(frame, (c2_x + 80, c2_y + 150), (c2_x + 105, c2_y + 270), (100, 65, 40), -1)
        cv2.rectangle(frame, (c2_x + 260, c2_y + 150), (c2_x + 285, c2_y + 270), (100, 65, 40), -1)
        # Swollen Udder (Enlarged & reddish)
        cv2.ellipse(frame, (c2_x + 220, c2_y + 185), (32, 26), 0, 0, 360, (140, 140, 220), -1)

        # Environment Info Overlay
        cv2.putText(frame, "SIH 26109: Smart Barn Multi-Cow CV Stream", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 50, 70), 2)
        cv2.putText(frame, f"Simulated Barn Camera #1 | Frame: {frame_i+1}/{total_frames}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 100, 110), 1)

        writer.write(frame)

    writer.release()
    print(f"[DemoGenerator] Successfully generated demo video at: {output_path}")


if __name__ == "__main__":
    generate_sih_demo_video("demo_barn_feed.mp4")
