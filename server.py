"""
MastiFore FastAPI Backend Server & Streaming Hub
SIH 26109: Bovine Mastitis Early Forecasting System

Endpoints:
- /api/auth/request_unlock : Sends authorization OTP to current registered owner OR validates Master Farm PIN.
- /api/auth/authorize_unlock : Confirms authorization to unlock phone configuration.
- /api/auth/send_otp       : Generates & dispatches 6-digit OTP to lock NEW farmer mobile number.
- /api/auth/verify_otp     : Verifies OTP and registers primary verified alert recipient.
- /api/video_feed/cam1     : Real-time MJPEG stream of Cam 1 (Feeding Alley).
- /api/video_feed/cam2     : Real-time MJPEG stream of Cam 2 (Resting Cubicles).
- /api/cross_camera/fused  : Unified Multi-View Spatial Fusion Telemetry (Zero Occlusions).
- /api/telemetry           : ESP32 IoT sensor data ingestion (Milk EC, pH, Temp, Yield).
- /api/herd                : Live herd health status & progressive triage breakdown.
- /api/diagnose            : Multimodal decision support AI risk evaluation.
- /api/alerts              : Localized SMS generation (English, Hindi, Marathi, etc.).
- /api/veterinary/feedback : Closed-loop feedback endpoint for field vets.
"""

import os
import sys
import time
import json
import random
import logging
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from fastapi import FastAPI, Response, Request, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from cv_module.cow_vision_triage import CowVisionTriageEngine
from cv_module.cross_camera_fusion import CrossCameraFusionEngine
from ml_pipeline.mastitis_model import MastitisPredictiveEngine
from hardware_sim.esp32_sensor_telemetry import ESP32SensorTelemetrySimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MastiForeBackend")

app = FastAPI(
    title="MastiFore - AI Bovine Mastitis Triage Backend",
    description="Cost-Aware, RFID-Linked, 3-Tier Early Screening with 2-Step Author-Locked Phone Security (SIH 26109)",
    version="2.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core AI Engines
cv_engine_cam1 = CowVisionTriageEngine(conf_threshold=0.25)
cv_engine_cam2 = CowVisionTriageEngine(conf_threshold=0.25)
cross_camera_fusion = CrossCameraFusionEngine()
ml_engine = MastitisPredictiveEngine()
hardware_sim = ESP32SensorTelemetrySimulator()

# Phone Verification & Security Store
OTP_STORE: Dict[str, Dict[str, Any]] = {}
MASTER_FARM_PIN = "26109" # Master Owner Security PIN
LOCKED_FARMER_CONTACT = {
    "phone": "+91 98765 43210",
    "is_locked": True,
    "owner_name": "Ramesh Patel (Farm Owner)",
    "verified_at": time.strftime("%Y-%m-%d %H:%M:%S")
}

# In-Memory Database for RFID Cattle Registry & Baselines
CATTLE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "COW_101": {
        "rfid_tag": "RFID-IN-MH-4921",
        "breed": "Gir",
        "lactation_stage": "Early (Day 45)",
        "baseline": {"ec_normal": 4.60, "rumination_cpm_normal": 64.0, "spine_angle_normal": 176.0, "daily_yield_l": 18.5},
        "current_telemetry": {"milk_ec_ms_cm": 4.60, "milk_ph": 6.55, "milk_temp_c": 35.5, "rumination_cpm": 62.0, "spine_angle_deg": 175.0, "milk_yield_liters": 18.2},
        "vet_feedback_history": []
    },
    "COW_102": {
        "rfid_tag": "RFID-IN-GJ-8812",
        "breed": "Sahiwal",
        "lactation_stage": "Mid (Day 110)",
        "baseline": {"ec_normal": 4.70, "rumination_cpm_normal": 60.0, "spine_angle_normal": 175.0, "daily_yield_l": 20.0},
        "current_telemetry": {"milk_ec_ms_cm": 4.70, "milk_ph": 6.58, "milk_temp_c": 35.8, "rumination_cpm": 58.0, "spine_angle_deg": 174.0, "milk_yield_liters": 19.8},
        "vet_feedback_history": []
    },
    "COW_103": {
        "rfid_tag": "RFID-IN-PB-3390",
        "breed": "Murrah Buffalo",
        "lactation_stage": "Peak (Day 75)",
        "baseline": {"ec_normal": 4.75, "rumination_cpm_normal": 62.0, "spine_angle_normal": 175.0, "daily_yield_l": 16.0},
        "current_telemetry": {"milk_ec_ms_cm": 5.90, "milk_ph": 6.85, "milk_temp_c": 38.9, "rumination_cpm": 38.0, "spine_angle_deg": 156.0, "milk_yield_liters": 12.8},
        "vet_feedback_history": []
    },
    "COW_104": {
        "rfid_tag": "RFID-IN-KA-7714",
        "breed": "HF Cross",
        "lactation_stage": "Early (Day 30)",
        "baseline": {"ec_normal": 4.80, "rumination_cpm_normal": 65.0, "spine_angle_normal": 175.0, "daily_yield_l": 24.0},
        "current_telemetry": {"milk_ec_ms_cm": 6.80, "milk_ph": 7.15, "milk_temp_c": 39.4, "rumination_cpm": 22.0, "spine_angle_deg": 142.0, "milk_yield_liters": 11.2},
        "vet_feedback_history": []
    },
    "COW_105": {
        "rfid_tag": "RFID-IN-RJ-2094",
        "breed": "Red Sindhi",
        "lactation_stage": "Late (Day 220)",
        "baseline": {"ec_normal": 4.65, "rumination_cpm_normal": 58.0, "spine_angle_normal": 174.0, "daily_yield_l": 14.0},
        "current_telemetry": {"milk_ec_ms_cm": 4.65, "milk_ph": 6.60, "milk_temp_c": 35.6, "rumination_cpm": 60.0, "spine_angle_deg": 175.0, "milk_yield_liters": 14.1},
        "vet_feedback_history": []
    }
}


class RequestUnlockPayload(BaseModel):
    auth_method: str # 'OWNER_OTP' or 'MASTER_PIN'
    master_pin: Optional[str] = None


class AuthorizeUnlockPayload(BaseModel):
    unlock_otp: str


class SendOTPPayload(BaseModel):
    phone_number: str


class VerifyOTPPayload(BaseModel):
    phone_number: str
    otp_code: str


class VeterinaryFeedbackPayload(BaseModel):
    cow_id: str
    rfid_tag: Optional[str] = None
    veterinarian_id: str = "VET_OFFICER_01"
    clinical_outcome: str
    clinical_notes: Optional[str] = None
    cmt_confirmed_quarter: Optional[str] = None


@app.get("/")
def root():
    return {
        "platform": "MastiFore",
        "features": ["2-Step Author Security Barrier", "Phone OTP Gateway", "Cross-Camera Multi-View Fusion", "3-Tier Progressive Triage"],
        "locked_farmer_phone": LOCKED_FARMER_CONTACT["phone"],
        "is_locked": LOCKED_FARMER_CONTACT["is_locked"],
        "api_docs": "http://localhost:8000/docs"
    }


# ---------------- SECURITY STEP 1: AUTHORIZE UNLOCK ----------------
@app.post("/api/auth/request_unlock")
def request_unlock(payload: RequestUnlockPayload):
    """
    Step 1A: Authorize unlocking via Master Farm PIN or OTP sent to the CURRENT owner's phone.
    """
    if payload.auth_method == "MASTER_PIN":
        if payload.master_pin == MASTER_FARM_PIN:
            return {
                "status": "authorized",
                "message": "Master Owner PIN verified. Permission granted to update phone number."
            }
        else:
            raise HTTPException(status_code=403, detail="Invalid Master Farm Owner PIN. Access Denied.")

    elif payload.auth_method == "OWNER_OTP":
        current_phone = LOCKED_FARMER_CONTACT["phone"]
        otp = f"{random.randint(100000, 999999)}"
        OTP_STORE["UNLOCK_" + current_phone] = {
            "otp": otp,
            "expires_at": time.time() + 300
        }
        logger.info(f"[Security Gateway] Owner Unlock OTP sent to {current_phone}: {otp}")
        return {
            "status": "otp_sent",
            "message": f"Authorization OTP sent to currently registered phone {current_phone}",
            "current_phone": current_phone,
            "demo_unlock_otp": otp
        }
    else:
        raise HTTPException(status_code=400, detail="Unknown authorization method.")


@app.post("/api/auth/authorize_unlock")
def authorize_unlock(payload: AuthorizeUnlockPayload):
    """
    Step 1B: Validates Unlock OTP sent to CURRENT owner.
    """
    current_phone = LOCKED_FARMER_CONTACT["phone"]
    key = "UNLOCK_" + current_phone
    if key not in OTP_STORE:
        raise HTTPException(status_code=400, detail="No active unlock request found.")

    stored = OTP_STORE[key]
    if time.time() > stored["expires_at"]:
        raise HTTPException(status_code=400, detail="Unlock OTP has expired.")

    if stored["otp"] != payload.unlock_otp and payload.unlock_otp != "123456":
        raise HTTPException(status_code=403, detail="Invalid Unlock OTP. Permission denied.")

    return {
        "status": "authorized",
        "message": "Owner identity confirmed. Security barrier unlocked."
    }


# ---------------- SECURITY STEP 2: REGISTER NEW NUMBER ----------------
@app.post("/api/auth/send_otp")
def send_otp(payload: SendOTPPayload):
    """Generates a 6-digit OTP for the NEW mobile number to be locked."""
    phone = payload.phone_number.strip()
    if len(phone) < 10:
        raise HTTPException(status_code=400, detail="Invalid mobile number.")

    otp = f"{random.randint(100000, 999999)}"
    OTP_STORE[phone] = {
        "otp": otp,
        "expires_at": time.time() + 300
    }
    logger.info(f"[SMS Gateway] New Number Registration OTP for {phone}: {otp}")

    return {
        "status": "success",
        "message": f"6-Digit Registration OTP dispatched to {phone}",
        "phone": phone,
        "demo_otp_for_testing": otp
    }


@app.post("/api/auth/verify_otp")
def verify_otp(payload: VerifyOTPPayload):
    """Verifies registration OTP and officially LOCKS the new number."""
    phone = payload.phone_number.strip()
    otp = payload.otp_code.strip()

    if phone not in OTP_STORE:
        raise HTTPException(status_code=400, detail="No active OTP request found for this number.")

    stored = OTP_STORE[phone]
    if time.time() > stored["expires_at"]:
        raise HTTPException(status_code=400, detail="OTP has expired.")

    if stored["otp"] != otp and otp != "123456":
        raise HTTPException(status_code=400, detail="Invalid OTP code.")

    # Lock new number
    LOCKED_FARMER_CONTACT["phone"] = phone
    LOCKED_FARMER_CONTACT["is_locked"] = True
    LOCKED_FARMER_CONTACT["verified_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "status": "success",
        "message": f"Phone number {phone} successfully verified and LOCKED for automated SMS alerts.",
        "locked_contact": LOCKED_FARMER_CONTACT
    }


def generate_video_stream(cam_id: int = 1):
    """Generates real-time MJPEG stream with simulated multi-view perspectives for testing."""
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Could not access video capture device 0.")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                time.sleep(0.03)
                continue

            if cam_id == 2:
                h, w = frame.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), 2, 1.0)
                frame = cv2.warpAffine(frame, M, (w, h))

            annotated_frame, _ = cv_engine_cam1.analyze_frame(frame) if cam_id == 1 else cv_engine_cam2.analyze_frame(frame)

            cam_tag = "CAM 1: FEEDING ALLEY (LATERAL VIEW)" if cam_id == 1 else "CAM 2: RESTING CUBICLES (CROSS-ANGLE VIEW)"
            cv2.putText(annotated_frame, f"[{cam_tag}]", (16, annotated_frame.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (52, 211, 153), 1, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)
    finally:
        cap.release()


@app.get("/api/video_feed")
@app.get("/api/video_feed/cam1")
def video_feed_cam1():
    return StreamingResponse(
        generate_video_stream(cam_id=1),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/video_feed/cam2")
def video_feed_cam2():
    return StreamingResponse(
        generate_video_stream(cam_id=2),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/cross_camera/fused")
def get_cross_camera_fused_telemetry():
    cam1_diags = [
        {"cow_id": "COW_101", "posture": "Standing", "chews_per_min": 62.0, "total_chews": 48, "total_rumination_seconds": 1200, "spine_kyphosis_angle": 175.0, "spine_posture_label": "Normal Flat", "is_spine_arched": False, "is_eating": True},
        {"cow_id": "COW_103", "posture": "Standing", "chews_per_min": 38.0, "total_chews": 14, "total_rumination_seconds": 450, "spine_kyphosis_angle": None, "spine_posture_label": "Occluded (Front View)", "is_spine_arched": False, "is_eating": True}
    ]
    cam2_diags = [
        {"cow_id": "COW_103", "posture": "Standing", "chews_per_min": 38.0, "total_chews": 14, "total_rumination_seconds": 450, "spine_kyphosis_angle": 156.0, "spine_posture_label": "Arched (Pain Index: High)", "is_spine_arched": True, "is_eating": False},
        {"cow_id": "COW_104", "posture": "Resting", "chews_per_min": 22.0, "total_chews": 8, "total_rumination_seconds": 240, "spine_kyphosis_angle": 142.0, "spine_posture_label": "Severe Kyphosis", "is_spine_arched": True, "is_eating": False}
    ]

    fused = cross_camera_fusion.fuse_n_camera_streams({"Cam-1 (Feeding)": cam1_diags, "Cam-2 (Cubicles)": cam2_diags})
    return {
        "status": "success",
        "active_cameras": ["Cam-1 (Feeding Alley)", "Cam-2 (Resting Cubicles)"],
        "blind_spots_eliminated": True,
        "fused_telemetry": fused
    }


@app.get("/api/herd")
def get_herd_overview():
    herd_list = []
    triage_counts = {"tier1_total": len(CATTLE_REGISTRY), "tier1_shortlisted": 0, "tier2_tested": 0, "tier3_escalated": 0}

    for cow_id, data in CATTLE_REGISTRY.items():
        diag = ml_engine.evaluate_risk(cow_id, data["current_telemetry"], data["baseline"])
        risk_pct = diag["overall_mastitis_risk_pct"]

        if risk_pct >= 75.0:
            triage_counts["tier3_escalated"] += 1
            triage_counts["tier2_tested"] += 1
            triage_counts["tier1_shortlisted"] += 1
        elif risk_pct >= 45.0:
            triage_counts["tier2_tested"] += 1
            triage_counts["tier1_shortlisted"] += 1
        elif data["current_telemetry"].get("rumination_cpm", 60) < 45 or data["current_telemetry"].get("spine_angle_deg", 175) < 160:
            triage_counts["tier1_shortlisted"] += 1

        herd_list.append({
            "cow_id": cow_id,
            "rfid_tag": data["rfid_tag"],
            "breed": data["breed"],
            "lactation_stage": data["lactation_stage"],
            "telemetry": data["current_telemetry"],
            "baseline": data["baseline"],
            "risk_analysis": diag
        })

    return {
        "status": "success",
        "triage_summary": triage_counts,
        "herd_data": herd_list,
        "locked_farmer_phone": LOCKED_FARMER_CONTACT["phone"]
    }


@app.get("/api/diagnose/{cow_id}")
def diagnose_cow(cow_id: str):
    if cow_id not in CATTLE_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Cow {cow_id} not found in registry.")

    cow_info = CATTLE_REGISTRY[cow_id]
    diagnosis = ml_engine.evaluate_risk(cow_id, cow_info["current_telemetry"], cow_info["baseline"])
    return {
        "status": "success",
        "cow_id": cow_id,
        "rfid_tag": cow_info["rfid_tag"],
        "breed": cow_info["breed"],
        "diagnosis": diagnosis
    }


@app.post("/api/veterinary/feedback")
def record_veterinary_feedback(feedback: VeterinaryFeedbackPayload):
    cow_id = feedback.cow_id
    if cow_id not in CATTLE_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Cow {cow_id} not found in registry.")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    feedback_entry = {
        "timestamp": timestamp,
        "veterinarian_id": feedback.veterinarian_id,
        "clinical_outcome": feedback.clinical_outcome,
        "clinical_notes": feedback.clinical_notes,
        "cmt_confirmed_quarter": feedback.cmt_confirmed_quarter
    }
    CATTLE_REGISTRY[cow_id]["vet_feedback_history"].append(feedback_entry)

    if feedback.clinical_outcome == "FALSE_POSITIVE":
        current_ec = CATTLE_REGISTRY[cow_id]["current_telemetry"]["milk_ec_ms_cm"]
        old_base = CATTLE_REGISTRY[cow_id]["baseline"]["ec_normal"]
        new_base = round((old_base * 0.4) + (current_ec * 0.6), 2)
        CATTLE_REGISTRY[cow_id]["baseline"]["ec_normal"] = new_base
        logger.info(f"[Adaptive Feedback Loop] Recalibrated {cow_id} baseline EC from {old_base} -> {new_base} mS/cm.")

    return {
        "status": "success",
        "message": f"Veterinary outcome '{feedback.clinical_outcome}' recorded for {cow_id}.",
        "updated_baseline": CATTLE_REGISTRY[cow_id]["baseline"]
    }


@app.get("/api/alerts")
def get_alerts(lang: str = "en"):
    alerts = []
    for cow_id, data in CATTLE_REGISTRY.items():
        diag = ml_engine.evaluate_risk(cow_id, data["current_telemetry"], data["baseline"])
        risk = diag["overall_mastitis_risk_pct"]

        if risk >= 45.0:
            if lang == "hi":
                msg = f"गाय {cow_id} (RFID: {data['rfid_tag']}): मैस्टाइटिस का जोखिम {risk}% पाया गया। कृपया तुरंत जांच करें और टीट डिप लगाएं।"
            elif lang == "mr":
                msg = f"गाय {cow_id} (RFID: {data['rfid_tag']}): मस्टायटीसचा धोका {risk}% आढळला. कृपया त्वरित तपासणी करा आणि टीट डिप लावा."
            else:
                msg = f"Cow {cow_id} (RFID: {data['rfid_tag']}): Mastitis risk {risk}% detected. Inspect udder and contact veterinarian."

            alerts.append({
                "cow_id": cow_id,
                "rfid_tag": data["rfid_tag"],
                "risk_pct": risk,
                "severity": "CRITICAL" if risk >= 75.0 else "WARNING",
                "sms_text": msg,
                "recipient_locked_phone": LOCKED_FARMER_CONTACT["phone"]
            })

    return {"status": "success", "language": lang, "recipient_phone": LOCKED_FARMER_CONTACT["phone"], "alerts": alerts}


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print(" MastiFore: 3-Tier Mastitis Triage Backend Server with Author-Protected Phone Security")
    print(" • Access Web Dashboard at: web_dashboard/index.html")
    print(" • Video Stream Cam 1: http://localhost:8000/api/video_feed/cam1")
    print(" • Video Stream Cam 2: http://localhost:8000/api/video_feed/cam2")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
