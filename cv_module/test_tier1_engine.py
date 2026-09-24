import os
import sys
import cv2
import numpy as np
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cv_module.cow_vision_triage import CowVisionTriageEngine
from cv_module.cross_camera_fusion import CrossCameraFusionEngine


def run_tests():
    print("Testing Tier-1 Vision & Cross-Camera Fusion Modules...")
    engine = CowVisionTriageEngine(conf_threshold=0.25)
    fusion = CrossCameraFusionEngine()

    passed = 0
    total = 6

    # Test 1: Video Ingestion & Target Detection
    sample_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    cv2.rectangle(sample_frame, (400, 250), (900, 600), (45, 55, 60), -1)

    t0 = time.time()
    annotated, diags = engine.analyze_frame(sample_frame)
    latency = (time.time() - t0) * 1000

    if len(diags) >= 1:
        print(f"✓ Test 1: Frame ingestion & target detection (latency: {latency:.1f}ms)")
        passed += 1
    else:
        print("✗ Test 1: Target detection failed")

    # Test 2: Standing Posture
    standing_img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(standing_img, (150, 100), (450, 240), (255, 255, 255), -1)
    cv2.rectangle(standing_img, (180, 240), (210, 360), (255, 255, 255), -1)
    cv2.rectangle(standing_img, (390, 240), (420, 360), (255, 255, 255), -1)

    res_s = engine._classify_posture_and_behavior(standing_img, 600, 400, {"is_arched": False}, {"is_eating": False, "eating_acc": 0}, 0.85)
    if res_s["posture"] == "Standing":
        print("✓ Test 2: Standing posture classification verified")
        passed += 1
    else:
        print(f"✗ Test 2: Expected Standing, got {res_s['posture']}")

    # Test 3: Resting / Lying Posture
    resting_img = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.ellipse(resting_img, (300, 180), (220, 75), 0, 0, 360, (255, 255, 255), -1)

    res_r = engine._classify_posture_and_behavior(resting_img, 600, 300, {"is_arched": False}, {"is_eating": False, "eating_acc": 0}, 0.85)
    if res_r["posture"] == "Resting":
        print("✓ Test 3: Resting posture classification verified")
        passed += 1
    else:
        print(f"✗ Test 3: Expected Resting, got {res_r['posture']}")

    # Test 4: Spine Kyphosis Angle Measurement
    spine_flat = engine._analyze_spine_curvature(standing_img, 600, 400, 50, 50, 550, 350, 1280, 720)
    # Verify method handles occlusion and valid lateral curves
    if "kyphosis_angle" in spine_flat and "posture_label" in spine_flat:
        print("✓ Test 4: Spine kyphosis posture analysis verified")
        passed += 1
    else:
        print("✗ Test 4: Spine analysis failed")

    # Test 5: Static Image Noise Rejection (Differential Optical Flow)
    metric = engine._track_differential_chews("TEST_COW", standing_img, 600, 400, "left")
    if metric["total_chews"] == 0 and not metric["is_eating"]:
        print("✓ Test 5: Static noise rejection verified (no false chews)")
        passed += 1
    else:
        print("✗ Test 5: Detected false chews on static image")

    # Test 6: Cross-Camera Multi-View Fusion
    cam1_data = [{"cow_id": "COW_103", "chews_per_min": 38.0, "total_chews": 14, "is_eating": True}]
    cam2_data = [{"cow_id": "COW_103", "posture": "Standing", "spine_kyphosis_angle": 156.0, "is_spine_arched": True}]
    fused = fusion.fuse_camera_frames(cam1_data, cam2_data)[0]

    cam2_only = [{"cow_id": "COW_104", "posture": "Resting", "spine_kyphosis_angle": 175.0, "is_eating": True, "chews_per_min": 60.0, "total_chews": 100}]
    fused_fallback = fusion.fuse_camera_frames([], cam2_only)[0]

    if fused["rumination_cpm"] == 38.0 and fused["spine_kyphosis_angle"] == 156.0 and fused_fallback["posture"] == "Resting":
        print("✓ Test 6: Multi-camera fusion and fallback verified")
        passed += 1
    else:
        print("✗ Test 6: Multi-camera fusion output mismatch")

    print(f"\nAll tests complete: {passed}/{total} passed.")


if __name__ == "__main__":
    run_tests()
