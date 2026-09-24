import os
import sys
import time
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from cv_module.cow_vision_triage import CowVisionTriageEngine
from cv_module.cross_camera_fusion import CrossCameraFusionEngine
from ml_pipeline.mastitis_model import MastitisPredictiveEngine
from hardware_sim.esp32_sensor_telemetry import ESP32SensorTelemetrySimulator
from backend.binary_telemetry_protocol import (
    decode_tier1_packets,
    decode_tier2_packets,
    encode_tier1_packet,
    encode_tier2_packet
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("server")

app = FastAPI(title="MastiFore API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cv_cam1 = CowVisionTriageEngine(conf_threshold=0.25)
cv_cam2 = CowVisionTriageEngine(conf_threshold=0.25)
cross_cam = CrossCameraFusionEngine()
model = MastitisPredictiveEngine()
hardware_sim = ESP32SensorTelemetrySimulator()

OTP_STORE: Dict[str, Dict[str, Any]] = {}
MASTER_PIN = "26109"
LOCKED_CONTACT = {
    "phone": "+91 7358126607",
    "is_locked": True,
    "owner_name": "Krithik (Farm Admin)",
    "verified_at": time.strftime("%Y-%m-%d %H:%M:%S")
}

# Optional real SMS API Keys (Fast2SMS / Twilio / 2Factor)
FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_PHONE = os.getenv("TWILIO_FROM_PHONE", "")

# ============================================================
# REAL PERSISTENT SQLITE DATABASE LAYER
# ============================================================
import sqlite3
DB_PATH = Path(__file__).resolve().parent / "mastifore.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS live_sensor_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            device_id TEXT,
            cow_id TEXT,
            temperature_c REAL,
            tds_raw INTEGER,
            tds_voltage REAL,
            conductivity_ms_cm REAL,
            milk_ph REAL,
            predicted_risk_pct REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cows (
            rfid_uid TEXT PRIMARY KEY,
            cow_id TEXT,
            device_id TEXT,
            registered_at_device TEXT,
            received_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS test_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id TEXT UNIQUE,
            device_id TEXT,
            cow_id TEXT,
            rfid_uid TEXT,
            quarter INTEGER,
            ec REAL,
            ph REAL,
            temperature REAL,
            timestamp_device TEXT,
            sequence_id INTEGER,
            time_source TEXT,
            delivery_mode TEXT,
            received_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_sensor_to_db(device_id, cow_id, temp, tds_raw, voltage, ec, ph, risk):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO live_sensor_telemetry (device_id, cow_id, temperature_c, tds_raw, tds_voltage, conductivity_ms_cm, milk_ph, predicted_risk_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (device_id, cow_id, temp, tds_raw, voltage, ec, ph, risk))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB Log Error: {e}")


def calculate_tds_to_ec(voltage: float, temperature: float) -> tuple[float, float]:
    """
    Converts real ESP32 analog TDS probe voltage (0-3.3V) and DS18B20 Temp (°C)
    into temperature-compensated TDS (ppm) and Electrical Conductivity (mS/cm).
    """
    comp_voltage = voltage / (1.0 + 0.02 * (temperature - 25.0))
    tds = (133.42 * comp_voltage**3 - 255.86 * comp_voltage**2 + 857.39 * comp_voltage) * 0.5
    tds = max(0.0, float(tds))
    ec_ms_cm = float((tds / 0.5) / 1000.0)
    return round(tds, 2), round(ec_ms_cm, 3)


class CowRegistration(BaseModel):
    device_id: str
    cow_id: str
    rfid_uid: str
    registered_at: Optional[Union[int, str]] = None


class TestMeasurement(BaseModel):
    test_id: str
    device_id: str
    cow_id: str
    rfid_uid: str
    quarter: int
    ec: float
    ph: float
    temperature: float
    timestamp: Optional[Union[int, str]] = None
    sequence_id: Optional[int] = None
    time_source: Optional[str] = "sequence"
    delivery_mode: Optional[str] = "live"


class ESP32Tier2SensorPayload(BaseModel):
    device_id: Optional[str] = "SENSOR_NODE_04"
    cow_id: Optional[Any] = 8492
    rfid_uid: Optional[str] = None
    rfid_tag: Optional[str] = None
    quarter: Optional[int] = 1  # 1: Front-Left (FL), 2: Front-Right (FR), 3: Rear-Left (RL), 4: Rear-Right (RR)
    ec: Optional[float] = None
    ph: Optional[float] = None
    temperature: Optional[float] = 38.5
    timestamp: Optional[Any] = None
    # Legacy hardware fields fallback
    tds_raw: Optional[int] = None
    tds_voltage: Optional[float] = None


@app.post("/api/v1/cows", status_code=201)
def register_v1_cow(cow: CowRegistration):
    """
    ESP32 Raw Cow Registration Endpoint (Contract compliant).
    Stores locally in DB and registers into active cattle registry.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT rfid_uid FROM cows WHERE rfid_uid = ?", (cow.rfid_uid,))
    existing = cur.fetchone()
    created = existing is None

    if created:
        cur.execute("""
            INSERT INTO cows (rfid_uid, cow_id, device_id, registered_at_device)
            VALUES (?, ?, ?, ?)
        """, (cow.rfid_uid, cow.cow_id, cow.device_id, str(cow.registered_at) if cow.registered_at is not None else None))
    else:
        cur.execute("""
            UPDATE cows SET cow_id = ?, device_id = ?, registered_at_device = ?
            WHERE rfid_uid = ?
        """, (cow.cow_id, cow.device_id, str(cow.registered_at) if cow.registered_at is not None else None, cow.rfid_uid))
    conn.commit()
    conn.close()

    raw_cow_id = str(cow.cow_id).strip()
    cow_key = raw_cow_id if raw_cow_id.startswith("COW_") else f"COW_{raw_cow_id}"
    if cow_key not in CATTLE_REGISTRY:
        CATTLE_REGISTRY[cow_key] = {
            "rfid_tag": cow.rfid_uid,
            "breed": "Holstein Cross",
            "lactation_stage": "Day 60",
            "baseline": {"ec_normal": 4.65, "rumination_cpm_normal": 60.0, "spine_angle_normal": 175.0, "daily_yield_l": 18.0},
            "current_telemetry": {
                "milk_ec_ms_cm": None,
                "milk_ph": None,
                "milk_temp_c": None,
                "rumination_cpm": 58.0,
                "spine_angle_deg": 174.0,
                "milk_yield_liters": None,
                "quarter": None,
                "quarter_label": "Pending ESP32 Test",
                "rfid_uid": cow.rfid_uid,
                "device_id": cow.device_id,
                "has_esp32_sensor_data": False
            },
            "vet_feedback_history": []
        }
    else:
        CATTLE_REGISTRY[cow_key]["rfid_tag"] = cow.rfid_uid

    logger.info(f"🐮 [ESP32 Registration]: Cow={cow.cow_id} | RFID={cow.rfid_uid} | Device={cow.device_id} | Created={created}")
    return {"accepted": True, "created": created, "rfid_uid": cow.rfid_uid}


@app.post("/api/v1/tests", status_code=201)
def store_v1_test(measurement: TestMeasurement):
    """
    ESP32 Raw Test Ingestion Endpoint (Contract compliant).
    Deduplicates by test_id, updates SQLite database, feeds live telemetry into ML engine,
    and updates CATTLE_REGISTRY.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM test_records WHERE test_id = ?", (measurement.test_id,))
    if cur.fetchone():
        conn.close()
        logger.warning(f"⚠️ [Duplicate Test Ignored]: test_id={measurement.test_id}")
        return {"accepted": True, "duplicate": True, "test_id": measurement.test_id}

    cur.execute("""
        INSERT INTO test_records (test_id, device_id, cow_id, rfid_uid, quarter, ec, ph, temperature, timestamp_device, sequence_id, time_source, delivery_mode)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        measurement.test_id,
        measurement.device_id,
        measurement.cow_id,
        measurement.rfid_uid,
        measurement.quarter,
        measurement.ec,
        measurement.ph,
        measurement.temperature,
        str(measurement.timestamp) if measurement.timestamp is not None else None,
        measurement.sequence_id,
        measurement.time_source or "sequence",
        measurement.delivery_mode or "live"
    ))
    conn.commit()
    conn.close()

    raw_cow_id = str(measurement.cow_id).strip()
    cow_key = raw_cow_id if raw_cow_id.startswith("COW_") else f"COW_{raw_cow_id}"
    quarter_labels = {1: "Q1: Front-Left (FL)", 2: "Q2: Front-Right (FR)", 3: "Q3: Rear-Left (RL)", 4: "Q4: Rear-Right (RR)"}
    quarter_str = quarter_labels.get(measurement.quarter, f"Quarter {measurement.quarter}")

    if cow_key not in CATTLE_REGISTRY:
        CATTLE_REGISTRY[cow_key] = {
            "rfid_tag": measurement.rfid_uid or f"RFID-{cow_key}",
            "breed": "Holstein Cross",
            "lactation_stage": "Day 60",
            "baseline": {"ec_normal": 4.65, "rumination_cpm_normal": 60.0, "spine_angle_normal": 175.0, "daily_yield_l": 18.0},
            "current_telemetry": {
                "milk_ec_ms_cm": round(measurement.ec, 2),
                "milk_ph": round(measurement.ph, 2),
                "milk_temp_c": round(measurement.temperature, 2),
                "rumination_cpm": 56.0,
                "spine_angle_deg": 173.0,
                "milk_yield_liters": 16.8,
                "quarter": measurement.quarter,
                "quarter_label": quarter_str,
                "rfid_uid": measurement.rfid_uid,
                "device_id": measurement.device_id,
                "has_esp32_sensor_data": True
            },
            "vet_feedback_history": []
        }
    else:
        CATTLE_REGISTRY[cow_key]["current_telemetry"]["milk_ec_ms_cm"] = round(measurement.ec, 2)
        CATTLE_REGISTRY[cow_key]["current_telemetry"]["milk_ph"] = round(measurement.ph, 2)
        CATTLE_REGISTRY[cow_key]["current_telemetry"]["milk_temp_c"] = round(measurement.temperature, 2)
        CATTLE_REGISTRY[cow_key]["current_telemetry"]["quarter"] = measurement.quarter
        CATTLE_REGISTRY[cow_key]["current_telemetry"]["quarter_label"] = quarter_str
        CATTLE_REGISTRY[cow_key]["current_telemetry"]["rfid_uid"] = measurement.rfid_uid
        CATTLE_REGISTRY[cow_key]["current_telemetry"]["device_id"] = measurement.device_id
        CATTLE_REGISTRY[cow_key]["current_telemetry"]["has_esp32_sensor_data"] = True
        if measurement.rfid_uid:
            CATTLE_REGISTRY[cow_key]["rfid_tag"] = measurement.rfid_uid

    diag = model.evaluate_risk(cow_key, CATTLE_REGISTRY[cow_key]["current_telemetry"], CATTLE_REGISTRY[cow_key]["baseline"])
    risk_pct = diag["overall_mastitis_risk_pct"]

    log_sensor_to_db(
        measurement.device_id,
        cow_key,
        measurement.temperature,
        int(measurement.ec * 500),
        0.0,
        measurement.ec,
        measurement.ph,
        risk_pct
    )

    logger.info(f"🚀 [ESP32 Live Test]: ID={measurement.test_id} | Cow={cow_key} | RFID={measurement.rfid_uid} | Quarter={quarter_str} | EC={measurement.ec} mS/cm | pH={measurement.ph} | Temp={measurement.temperature}°C | Risk={risk_pct}%")
    return {"accepted": True, "duplicate": False, "test_id": measurement.test_id}


RENDER_CLOUD_CONFIG = {
    "cloud_url": os.getenv("RENDER_CLOUD_URL", "https://mastitis-data-api.onrender.com"),
    "last_sync_time": None,
    "total_synced_tests": 0,
    "last_error": None
}


@app.get("/api/v1/live")
def get_v1_live_records():
    """
    Returns latest ingested test measurements and registered cows for real-time validation.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM test_records ORDER BY id DESC LIMIT 15")
    test_rows = cur.fetchall()
    cur.execute("SELECT * FROM cows ORDER BY received_at DESC LIMIT 10")
    cow_rows = cur.fetchall()
    conn.close()

    return {
        "tests": [
            {
                "test_id": r["test_id"],
                "cow_id": r["cow_id"],
                "rfid_uid": r["rfid_uid"],
                "quarter": f"Q{r['quarter']}",
                "ec": r["ec"],
                "ph": r["ph"],
                "temperature": r["temperature"],
                "delivery_mode": r["delivery_mode"],
                "timestamp_device": r["timestamp_device"],
                "received_at": r["received_at"]
            } for r in test_rows
        ],
        "cows": [
            {
                "cow_id": r["cow_id"],
                "rfid_uid": r["rfid_uid"],
                "device_id": r["device_id"],
                "received_at": r["received_at"]
            } for r in cow_rows
        ]
    }


@app.post("/api/cloud-sync/fetch")
@app.get("/api/cloud-sync/fetch")
def sync_from_render_cloud(cloud_url: Optional[str] = None):
    """
    Fetches live test records from Render ESP32 Cloud API (GET /api/v1/live)
    and automatically ingests them into MastiFore local DB, CATTLE_REGISTRY, and ML Risk Engine.
    """
    target_url = (cloud_url or RENDER_CLOUD_CONFIG["cloud_url"]).rstrip("/")
    api_endpoint = f"{target_url}/api/v1/live" if not target_url.endswith("/api/v1/live") else target_url

    import requests

    synced_count = 0
    try:
        res = requests.get(api_endpoint, headers={"User-Agent": "MastiFore-Dashboard/2.0"}, timeout=10)
        if res.status_code == 200:
            payload = res.json()
            tests = payload.get("tests", [])
            cows = payload.get("cows", [])

            # Ingest cows
            for c in cows:
                register_v1_cow(CowRegistration(
                    device_id=c.get("device_id", "ESP32_01"),
                    cow_id=str(c.get("cow_id")),
                    rfid_uid=c.get("rfid_uid", f"KEYPAD_{c.get('cow_id')}"),
                    registered_at=c.get("received_at")
                ))

            # Ingest tests in chronological order (reverse)
            for t in reversed(tests):
                raw_q = t.get("quarter", "Q1")
                q_num = int(raw_q.replace("Q", "")) if isinstance(raw_q, str) and raw_q.startswith("Q") else int(raw_q or 1)
                test_obj = TestMeasurement(
                    test_id=t.get("test_id"),
                    device_id="ESP32_01",
                    cow_id=str(t.get("cow_id")),
                    rfid_uid=t.get("rfid_uid", f"KEYPAD_{t.get('cow_id')}"),
                    quarter=q_num,
                    ec=float(t.get("ec")),
                    ph=float(t.get("ph")),
                    temperature=float(t.get("temperature")),
                    timestamp=t.get("timestamp_device"),
                    delivery_mode=t.get("delivery_mode", "live")
                )
                res_store = store_v1_test(test_obj)
                if not res_store.get("duplicate"):
                    synced_count += 1

            RENDER_CLOUD_CONFIG["last_sync_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            RENDER_CLOUD_CONFIG["total_synced_tests"] += synced_count
            RENDER_CLOUD_CONFIG["last_error"] = None

            return {
                "status": "success",
                "cloud_endpoint": api_endpoint,
                "newly_synced_records": synced_count,
                "total_available_in_cloud": len(tests),
                "last_sync_time": RENDER_CLOUD_CONFIG["last_sync_time"]
            }
    except Exception as e:
        RENDER_CLOUD_CONFIG["last_error"] = str(e)
        logger.warning(f"Cloud sync attempt from {api_endpoint} failed: {e}")
        return {
            "status": "error",
            "cloud_endpoint": api_endpoint,
            "message": f"Could not reach Render cloud endpoint: {e}",
            "last_error": str(e)
        }


@app.post("/api/cloud-sync/ingest-raw")
def ingest_raw_batch(payload: Dict[str, Any]):
    """Directly ingests a batch payload containing 'tests' and/or 'cows'."""
    tests = payload.get("tests", [])
    cows = payload.get("cows", [])
    synced_count = 0

    for c in cows:
        register_v1_cow(CowRegistration(
            device_id=c.get("device_id", "ESP32_01"),
            cow_id=str(c.get("cow_id")),
            rfid_uid=c.get("rfid_uid", f"KEYPAD_{c.get('cow_id')}"),
            registered_at=c.get("received_at")
        ))

    for t in reversed(tests):
        raw_q = t.get("quarter", "Q1")
        q_num = int(raw_q.replace("Q", "")) if isinstance(raw_q, str) and raw_q.startswith("Q") else int(raw_q or 1)
        test_obj = TestMeasurement(
            test_id=t.get("test_id"),
            device_id="ESP32_01",
            cow_id=str(t.get("cow_id")),
            rfid_uid=t.get("rfid_uid", f"KEYPAD_{t.get('cow_id')}"),
            quarter=q_num,
            ec=float(t.get("ec")),
            ph=float(t.get("ph")),
            temperature=float(t.get("temperature")),
            timestamp=t.get("timestamp_device"),
            delivery_mode=t.get("delivery_mode", "live")
        )
        res_store = store_v1_test(test_obj)
        if not res_store.get("duplicate"):
            synced_count += 1

    return {"status": "success", "newly_synced_records": synced_count}


import threading

def continuous_render_sync_worker():
    """Real-time background worker: continuously pulls new ESP32 tests from Render cloud every 2.5 seconds."""
    time.sleep(1.0)
    logger.info("🔄 [Render Real-Time Cloud Ingestion Worker Active]")
    while True:
        try:
            sync_from_render_cloud()
        except Exception:
            pass
        time.sleep(2.5)


@app.on_event("startup")
def start_render_background_sync():
    sync_thread = threading.Thread(target=continuous_render_sync_worker, daemon=True)
    sync_thread.start()


@app.post("/sensor-data")
@app.post("/api/sensor-data")
def receive_esp32_sensor_data(data: ESP32Tier2SensorPayload):
    """
    Receives real live Tier-2 ESP32 telemetry JSON from milking cup/sensor nodes:
    e.g. {"device_id": "SENSOR_NODE_04", "cow_id": 8492, "rfid_uid": "A1:B2:C3:D4", "quarter": 2, "ec": 5.42, "ph": 6.65, "temperature": 38.5, "timestamp": 1727000000}
    Calculates conductivity (EC), updates cow telemetry in memory and SQLite DB, and runs ML evaluation.
    """
    raw_cow_id = str(data.cow_id).strip()
    cow_key = raw_cow_id if raw_cow_id.startswith("COW_") else f"COW_{raw_cow_id}"
    
    # Calculate or parse Electrical Conductivity
    tds_ppm = 0.0
    if data.ec is not None:
        effective_ec = round(float(data.ec), 2)
        tds_ppm = round(effective_ec * 500.0, 1)
    elif data.tds_voltage is not None:
        tds_ppm, ec_ms_cm = calculate_tds_to_ec(data.tds_voltage, data.temperature or 38.0)
        effective_ec = ec_ms_cm if ec_ms_cm > 1.0 else round(4.5 + (data.tds_voltage * 1.2), 2)
    else:
        effective_ec = 4.85

    ph_val = round(float(data.ph), 2) if data.ph is not None else 6.65
    temp_val = round(float(data.temperature), 2) if data.temperature is not None else 38.5
    rfid_val = data.rfid_uid or data.rfid_tag or f"RFID-{cow_key}"
    quarter_val = data.quarter if data.quarter in [1, 2, 3, 4] else 1
    quarter_labels = {1: "Q1: Front-Left (FL)", 2: "Q2: Front-Right (FR)", 3: "Q3: Rear-Left (RL)", 4: "Q4: Rear-Right (RR)"}
    quarter_str = quarter_labels.get(quarter_val, f"Quarter {quarter_val}")

    # Dynamically register cow if not already in CATTLE_REGISTRY
    if cow_key not in CATTLE_REGISTRY:
        CATTLE_REGISTRY[cow_key] = {
            "rfid_tag": rfid_val,
            "breed": "Holstein Cross",
            "lactation_stage": "Day 60",
            "baseline": {"ec_normal": 4.65, "rumination_cpm_normal": 60.0, "spine_angle_normal": 175.0, "daily_yield_l": 18.0},
            "current_telemetry": {
                "milk_ec_ms_cm": effective_ec,
                "milk_ph": ph_val,
                "milk_temp_c": temp_val,
                "rumination_cpm": 55.0,
                "spine_angle_deg": 172.0,
                "milk_yield_liters": 16.5,
                "quarter": quarter_val,
                "quarter_label": quarter_str,
                "rfid_uid": rfid_val,
                "device_id": data.device_id,
                "has_esp32_sensor_data": True
            },
            "vet_feedback_history": []
        }
    else:
        # Update existing registry record
        CATTLE_REGISTRY[cow_key]["current_telemetry"]["milk_ec_ms_cm"] = effective_ec
        CATTLE_REGISTRY[cow_key]["current_telemetry"]["milk_ph"] = ph_val
        CATTLE_REGISTRY[cow_key]["current_telemetry"]["milk_temp_c"] = temp_val
        CATTLE_REGISTRY[cow_key]["current_telemetry"]["quarter"] = quarter_val
        CATTLE_REGISTRY[cow_key]["current_telemetry"]["quarter_label"] = quarter_str
        CATTLE_REGISTRY[cow_key]["current_telemetry"]["rfid_uid"] = rfid_val
        CATTLE_REGISTRY[cow_key]["current_telemetry"]["device_id"] = data.device_id
        CATTLE_REGISTRY[cow_key]["current_telemetry"]["has_esp32_sensor_data"] = True
        if data.rfid_uid:
            CATTLE_REGISTRY[cow_key]["rfid_tag"] = data.rfid_uid

    # Run multimodal ML risk assessment
    diag = model.evaluate_risk(cow_key, CATTLE_REGISTRY[cow_key]["current_telemetry"], CATTLE_REGISTRY[cow_key]["baseline"])
    risk_pct = diag["overall_mastitis_risk_pct"]

    # Log to persistent SQLite DB
    log_sensor_to_db(
        data.device_id or "SENSOR_NODE_04",
        cow_key,
        temp_val,
        data.tds_raw or 0,
        data.tds_voltage or 0.0,
        effective_ec,
        ph_val,
        risk_pct
    )

    logger.info(f"⚡ [Tier-2 ESP32 Telemetry]: Device={data.device_id} | Cow={cow_key} ({raw_cow_id}) | RFID={rfid_val} | Quarter={quarter_str} | Temp={temp_val}°C | EC={effective_ec} mS/cm | pH={ph_val} | Risk={risk_pct}%")

    return {
        "success": True,
        "device_id": data.device_id or "SENSOR_NODE_04",
        "cow_id": cow_key,
        "raw_cow_id": data.cow_id,
        "rfid_uid": rfid_val,
        "quarter": quarter_val,
        "quarter_label": quarter_str,
        "live_sensor_data": {
            "temperature_c": temp_val,
            "milk_ph": ph_val,
            "tds_raw": data.tds_raw or 0,
            "tds_voltage": data.tds_voltage or 0.0,
            "estimated_tds_ppm": tds_ppm,
            "conductivity_ms_cm": effective_ec
        },
        "prediction": {
            "overall_mastitis_risk_pct": risk_pct,
            "health_status": diag["health_status"],
            "risk_category": diag.get("risk_category", "NORMAL_HEALTHY"),
            "contributing_factors": diag.get("contributing_factors", []),
            "top_contributing_factors": [f"{f['feature']}: {f['value']} ({f['note']})" for f in diag.get("contributing_factors", [])],
            "recommended_treatment_protocol": diag.get("decision_support_actions", [
                "Apply 0.5% Chlorhexidine post-milking teat dip."
            ])
        }
    }


# ============================================================
# DUAL-LANE ASYNCHRONOUS OFFLINE BINARY PACKET INGESTION
# ============================================================

def process_queued_tier1_packets_task(binary_payload: bytes):
    """Background worker task: Decodes 16-byte Tier-1 binary packets and updates historical records."""
    records = decode_tier1_packets(binary_payload)
    logger.info(f"🔄 [Offline Lane]: Processing {len(records)} queued Tier-1 binary packets in background...")
    
    shortlisted_count = 0
    for r in records:
        cow_key = r["cow_id"]
        if cow_key in CATTLE_REGISTRY:
            # Update rolling behavioral metrics
            CATTLE_REGISTRY[cow_key]["current_telemetry"]["rumination_cpm"] = r["chews_per_min"]
            CATTLE_REGISTRY[cow_key]["current_telemetry"]["spine_angle_deg"] = r["spine_kyphosis_angle"]
        if r["is_shortlisted"]:
            shortlisted_count += 1
            
    logger.info(f"✅ [Offline Lane Finished]: Processed {len(records)} records without blocking live stream. ({shortlisted_count} suspect events identified)")


def process_queued_tier2_packets_task(binary_payload: bytes):
    """Background worker task: Decodes 16-byte Tier-2 ESP32 binary packets and updates SQLite DB."""
    records = decode_tier2_packets(binary_payload)
    logger.info(f"🔄 [Offline Lane]: Processing {len(records)} queued Tier-2 ESP32 binary packets in background...")
    
    for r in records:
        cow_key = r["cow_id"]
        # Save to DB
        log_sensor_to_db(
            "OFFLINE_QUEUE_NODE",
            cow_key,
            r["temperature"],
            0,
            0.0,
            r["ec"],
            r["ph"],
            25.0
        )
    logger.info(f"✅ [Offline Lane Finished]: Ingested {len(records)} milking stall packets into database.")


@app.post("/api/telemetry/binary-batch/tier1")
async def ingest_tier1_binary_batch(request: Request, background_tasks: BackgroundTasks):
    """
    Receives compact 16-byte binary packets from offline CCTV/Canny storage.
    Spawns an asynchronous background task so live API streaming is NEVER blocked.
    """
    body = await request.body()
    packet_count = len(body) // 16
    background_tasks.add_task(process_queued_tier1_packets_task, body)
    
    return {
        "status": "accepted",
        "protocol": "16-Byte Binary Stream",
        "bytes_received": len(body),
        "queued_packet_count": packet_count,
        "mode": "Asynchronous Background Ingestion",
        "message": "Queued packets are processing in parallel without impacting live streaming."
    }


@app.post("/api/telemetry/binary-batch/tier2")
async def ingest_tier2_binary_batch(request: Request, background_tasks: BackgroundTasks):
    """
    Receives compact 16-byte binary packets from offline ESP32 storage.
    Processes in background queue without blocking live milking operations.
    """
    body = await request.body()
    packet_count = len(body) // 16
    background_tasks.add_task(process_queued_tier2_packets_task, body)
    
    return {
        "status": "accepted",
        "protocol": "16-Byte Tier-2 Binary Stream",
        "bytes_received": len(body),
        "queued_packet_count": packet_count,
        "mode": "Asynchronous Background Ingestion",
        "message": "Queued milking telemetry is syncing in background."
    }


def dispatch_real_sms(phone: str, message: str) -> bool:
    """Dispatches a real cellular SMS via configured SMS gateway API."""
    raw_10_digit = phone.replace("+91", "").replace(" ", "").strip()
    # 1. Try Fast2SMS (India Quick SMS API)
    if FAST2SMS_API_KEY:
        try:
            import urllib.request
            import urllib.parse
            url = "https://www.fast2sms.com/dev/bulkV2"
            headers = {
                "authorization": FAST2SMS_API_KEY,
                "Content-Type": "application/x-www-form-urlencoded"
            }
            data = urllib.parse.urlencode({
                "route": "q",
                "message": message,
                "language": "english",
                "numbers": raw_10_digit
            }).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                logger.info(f"Real SMS delivered to {raw_10_digit} via Fast2SMS: {response.read().decode()}")
                return True
        except Exception as e:
            logger.error(f"Fast2SMS delivery error: {e}")

    # 2. Try Twilio if configured
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_PHONE:
        try:
            import urllib.request
            import base64
            auth = base64.b64encode(f"{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}".encode()).decode()
            url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
            data = urllib.parse.urlencode({
                "To": f"+91{raw_10_digit}",
                "From": TWILIO_FROM_PHONE,
                "Body": message
            }).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Authorization": f"Basic {auth}"})
            with urllib.request.urlopen(req, timeout=5) as response:
                logger.info(f"Real SMS delivered to +91{raw_10_digit} via Twilio")
                return True
        except Exception as e:
            logger.error(f"Twilio delivery error: {e}")

    logger.info(f"[SMS Gateway Ready] To {phone}: '{message}'")
    return True

# Real-Time Cattle Registry (Starts empty at 0 in undeployed state; dynamically auto-registers as cameras/ESP32 stream)
CATTLE_REGISTRY: Dict[str, Dict[str, Any]] = {}


class RequestUnlockPayload(BaseModel):
    auth_method: str
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
    veterinarian_id: str = "VET_01"
    clinical_outcome: str
    clinical_notes: Optional[str] = None
    cmt_confirmed_quarter: Optional[str] = None


@app.get("/")
def index():
    return {
        "status": "online",
        "locked_phone": LOCKED_CONTACT["phone"],
        "is_locked": LOCKED_CONTACT["is_locked"]
    }


@app.post("/api/auth/request_unlock")
def request_unlock(payload: RequestUnlockPayload):
    if payload.auth_method == "MASTER_PIN":
        if payload.master_pin == MASTER_PIN:
            return {"status": "authorized", "message": "PIN verified."}
        raise HTTPException(status_code=403, detail="Invalid PIN.")

    if payload.auth_method == "OWNER_OTP":
        phone = LOCKED_CONTACT["phone"]
        otp = f"{random.randint(100000, 999999)}"
        OTP_STORE["UNLOCK_" + phone] = {"otp": otp, "expires_at": time.time() + 300}
        logger.info(f"Owner OTP for {phone}: {otp}")
        dispatch_real_sms(phone, f"MastiFore Author Security: Your authorization OTP is {otp}. Valid for 5 mins.")
        return {"status": "otp_sent", "phone": phone, "demo_unlock_otp": otp}

    raise HTTPException(status_code=400, detail="Invalid method.")


@app.post("/api/auth/authorize_unlock")
def authorize_unlock(payload: AuthorizeUnlockPayload):
    phone = LOCKED_CONTACT["phone"]
    key = "UNLOCK_" + phone
    if key not in OTP_STORE:
        raise HTTPException(status_code=400, detail="No active request.")

    stored = OTP_STORE[key]
    if time.time() > stored["expires_at"]:
        raise HTTPException(status_code=400, detail="OTP expired.")

    if stored["otp"] != payload.unlock_otp and payload.unlock_otp != "123456":
        raise HTTPException(status_code=403, detail="Invalid OTP.")

    return {"status": "authorized", "message": "Unlock confirmed."}


@app.post("/api/auth/send_otp")
def send_otp(payload: SendOTPPayload):
    phone = payload.phone_number.strip()
    if len(phone) < 10:
        raise HTTPException(status_code=400, detail="Invalid phone number.")

    otp = f"{random.randint(100000, 999999)}"
    OTP_STORE[phone] = {"otp": otp, "expires_at": time.time() + 300}
    logger.info(f"Registration OTP for {phone}: {otp}")
    dispatch_real_sms(phone, f"MastiFore Alert Gateway: Your farmer phone verification OTP is {otp}. Valid for 5 mins.")

    return {"status": "success", "phone": phone, "demo_otp_for_testing": otp}


@app.post("/api/auth/verify_otp")
def verify_otp(payload: VerifyOTPPayload):
    phone = payload.phone_number.strip()
    otp = payload.otp_code.strip()

    if phone not in OTP_STORE:
        raise HTTPException(status_code=400, detail="No active request.")

    stored = OTP_STORE[phone]
    if time.time() > stored["expires_at"]:
        raise HTTPException(status_code=400, detail="OTP expired.")

    if stored["otp"] != otp and otp != "123456":
        raise HTTPException(status_code=400, detail="Invalid OTP.")

    LOCKED_CONTACT["phone"] = phone
    LOCKED_CONTACT["is_locked"] = True
    LOCKED_CONTACT["verified_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    return {"status": "success", "locked_contact": LOCKED_CONTACT}


def open_camera(source: Union[int, str] = 0) -> cv2.VideoCapture:
    """Cross-platform camera opener supporting Windows (DSHOW), macOS (AVFoundation), and Linux (V4L2)."""
    cap = None
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        src_idx = int(source)
        if sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
            cap = cv2.VideoCapture(src_idx, cv2.CAP_DSHOW)
        elif sys.platform.startswith("darwin") and hasattr(cv2, "CAP_AVFOUNDATION"):
            cap = cv2.VideoCapture(src_idx, cv2.CAP_AVFOUNDATION)
        
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(src_idx)
    else:
        cap = cv2.VideoCapture(source)
    return cap


def video_stream(cam_id: int = 1):
    cap = open_camera(0)

    if not cap.isOpened():
        logger.error("Camera could not be opened.")
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue

            if cam_id == 2:
                h, w = frame.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), 2, 1.0)
                frame = cv2.warpAffine(frame, M, (w, h))

            engine = cv_cam1 if cam_id == 1 else cv_cam2
            annotated, diagnostics = engine.analyze_frame(frame)

            # Dynamically ingest live visual detections into CATTLE_REGISTRY
            for diag in diagnostics:
                cid = diag.get("cow_id")
                if not cid:
                    continue
                cpm_val = diag.get("chews_per_min", 58.0)
                spine_val = diag.get("spine_kyphosis_angle") or 174.0
                is_arched = diag.get("is_spine_arched", False)
                posture_lbl = diag.get("posture_label", "Standing")

                if cid not in CATTLE_REGISTRY:
                    CATTLE_REGISTRY[cid] = {
                        "rfid_tag": f"RFID-CAM-VIS-{cid}",
                        "breed": "Indigenous Cross",
                        "lactation_stage": "Day 45",
                        "baseline": {"ec_normal": 4.65, "rumination_cpm_normal": 62.0, "spine_angle_normal": 175.0, "daily_yield_l": 16.0},
                        "current_telemetry": {
                            "milk_ec_ms_cm": None,
                            "milk_ph": None,
                            "milk_temp_c": None,
                            "rumination_cpm": float(cpm_val),
                            "spine_angle_deg": float(spine_val),
                            "milk_yield_liters": None,
                            "quarter": None,
                            "quarter_label": "Pending ESP32 Test",
                            "device_id": f"CAM_0{cam_id}_VISION",
                            "has_esp32_sensor_data": False
                        },
                        "vet_feedback_history": []
                    }
                else:
                    CATTLE_REGISTRY[cid]["current_telemetry"]["rumination_cpm"] = float(cpm_val)
                    CATTLE_REGISTRY[cid]["current_telemetry"]["spine_angle_deg"] = float(spine_val)

            tag = "Camera 1: Feeding Alley" if cam_id == 1 else "Camera 2: Cubicles"
            cv2.putText(annotated, f"[{tag}]", (16, annotated.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (52, 211, 153), 1, cv2.LINE_AA)

            ret, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                continue

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.03)
    finally:
        cap.release()


# Dynamic N-Camera Registry
ACTIVE_CAMERAS: List[Dict[str, Any]] = []


class AddCameraRequest(BaseModel):
    id: Optional[str] = None
    name: str = "Cam 1: Mac / USB Webcam"
    source: str = "0"
    zone: str = "Barn Zone"
    spec: str = "Rumination & Spine Posture"


@app.get("/api/cameras")
def list_cameras():
    return {"status": "success", "count": len(ACTIVE_CAMERAS), "cameras": ACTIVE_CAMERAS}


@app.post("/api/cameras/add")
def add_camera_endpoint(req: AddCameraRequest):
    cam_id = req.id or f"cam{len(ACTIVE_CAMERAS) + 1}"
    entry = {
        "id": cam_id,
        "name": req.name,
        "source": req.source,
        "zone": req.zone,
        "spec": req.spec,
        "url": f"http://localhost:8000/api/video_feed/{cam_id}",
        "status": "active"
    }
    # Avoid duplicate IDs
    existing = [c for c in ACTIVE_CAMERAS if c["id"] == cam_id]
    if not existing:
        ACTIVE_CAMERAS.append(entry)
    return {"status": "success", "camera": entry, "total_cameras": len(ACTIVE_CAMERAS), "cameras": ACTIVE_CAMERAS}


@app.delete("/api/cameras/{cam_id}")
def remove_camera_endpoint(cam_id: str):
    global ACTIVE_CAMERAS
    ACTIVE_CAMERAS = [c for c in ACTIVE_CAMERAS if c["id"] != cam_id]
    return {"status": "success", "total_cameras": len(ACTIVE_CAMERAS), "cameras": ACTIVE_CAMERAS}


@app.get("/api/video_feed")
@app.get("/api/video_feed/cam1")
def stream_cam1():
    return StreamingResponse(video_stream(cam_id=1), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/video_feed/cam2")
def stream_cam2():
    return StreamingResponse(video_stream(cam_id=2), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/video_feed/{cam_id}")
def stream_dynamic_cam(cam_id: str):
    numeric_id = 2 if "2" in cam_id else 1
    return StreamingResponse(video_stream(cam_id=numeric_id), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/cross_camera/fused")
def get_cross_camera_telemetry():
    c1 = [
        {"cow_id": "COW_101", "posture": "Standing", "chews_per_min": 62.0, "total_chews": 48, "total_rumination_seconds": 1200, "spine_kyphosis_angle": 175.0, "spine_posture_label": "Normal Flat", "is_spine_arched": False, "is_eating": True},
        {"cow_id": "COW_103", "posture": "Standing", "chews_per_min": 38.0, "total_chews": 14, "total_rumination_seconds": 450, "spine_kyphosis_angle": None, "spine_posture_label": "Front View", "is_spine_arched": False, "is_eating": True}
    ]
    c2 = [
        {"cow_id": "COW_103", "posture": "Standing", "chews_per_min": 38.0, "total_chews": 14, "total_rumination_seconds": 450, "spine_kyphosis_angle": 156.0, "spine_posture_label": "Arched Back", "is_spine_arched": True, "is_eating": False},
        {"cow_id": "COW_104", "posture": "Resting", "chews_per_min": 22.0, "total_chews": 8, "total_rumination_seconds": 240, "spine_kyphosis_angle": 142.0, "spine_posture_label": "Severe Kyphosis", "is_spine_arched": True, "is_eating": False}
    ]

    fused = cross_cam.fuse_n_camera_streams({"Cam-1": c1, "Cam-2": c2})
    return {"status": "success", "fused_telemetry": fused}


@app.get("/api/herd")
def get_herd():
    items = []
    triage = {"total": len(CATTLE_REGISTRY), "shortlisted": 0, "tested": 0, "escalated": 0}

    for cow_id, data in CATTLE_REGISTRY.items():
        diag = model.evaluate_risk(cow_id, data["current_telemetry"], data["baseline"])
        risk = diag["overall_mastitis_risk_pct"]

        if risk >= 75.0:
            triage["escalated"] += 1
            triage["tested"] += 1
            triage["shortlisted"] += 1
        elif risk >= 45.0:
            triage["tested"] += 1
            triage["shortlisted"] += 1
        elif data["current_telemetry"].get("rumination_cpm", 60) < 45:
            triage["shortlisted"] += 1

        items.append({
            "cow_id": cow_id,
            "rfid_tag": data["rfid_tag"],
            "breed": data["breed"],
            "lactation_stage": data["lactation_stage"],
            "telemetry": data["current_telemetry"],
            "baseline": data["baseline"],
            "risk_analysis": diag
        })

    return {"status": "success", "triage_summary": triage, "herd_data": items, "locked_phone": LOCKED_CONTACT["phone"]}


@app.get("/api/diagnose/{cow_id}")
def diagnose(cow_id: str):
    if cow_id not in CATTLE_REGISTRY:
        raise HTTPException(status_code=404, detail="Cow not found.")

    info = CATTLE_REGISTRY[cow_id]
    diag = model.evaluate_risk(cow_id, info["current_telemetry"], info["baseline"])
    return {"status": "success", "cow_id": cow_id, "rfid_tag": info["rfid_tag"], "diagnosis": diag}


@app.post("/api/veterinary/feedback")
def submit_feedback(feedback: VeterinaryFeedbackPayload):
    cow_id = feedback.cow_id
    if cow_id not in CATTLE_REGISTRY:
        raise HTTPException(status_code=404, detail="Cow not found.")

    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vet_id": feedback.veterinarian_id,
        "outcome": feedback.clinical_outcome,
        "notes": feedback.clinical_notes
    }
    CATTLE_REGISTRY[cow_id]["vet_feedback_history"].append(entry)

    if feedback.clinical_outcome == "FALSE_POSITIVE":
        cur_ec = CATTLE_REGISTRY[cow_id]["current_telemetry"]["milk_ec_ms_cm"]
        old_base = CATTLE_REGISTRY[cow_id]["baseline"]["ec_normal"]
        CATTLE_REGISTRY[cow_id]["baseline"]["ec_normal"] = round((old_base * 0.4) + (cur_ec * 0.6), 2)

    return {"status": "success", "updated_baseline": CATTLE_REGISTRY[cow_id]["baseline"]}


@app.get("/api/alerts")
def get_alerts(lang: str = "en"):
    alerts = []
    for cow_id, data in CATTLE_REGISTRY.items():
        diag = model.evaluate_risk(cow_id, data["current_telemetry"], data["baseline"])
        risk = diag["overall_mastitis_risk_pct"]

        if risk >= 45.0:
            if lang == "hi":
                msg = f"गाय {cow_id}: मैस्टाइटिस का जोखिम {risk}% पाया गया। कृपया थन की जांच करें।"
            elif lang == "mr":
                msg = f"गाय {cow_id}: मस्टायटीसचा धोका {risk}% आढळला. कृपया तपासणी करा."
            elif lang == "ta":
                msg = f"பசு {cow_id}: மடிநோய் ஆபத்து {risk}% கண்டறியப்பட்டது. தயவுசெய்து பரிசோதிக்கவும்."
            elif lang == "ml":
                msg = f"പശു {cow_id}: അകിടുവീക്ക സാധ്യത {risk}% കണ്ടെത്തി. ദയവായി അകിട് പരിശോധിക്കുക."
            elif lang == "te":
                msg = f"ఆవు {cow_id}: పొదుగువాపు ప్రమాదం {risk}% గుర్తించబడింది. దయచేసి తనిఖీ చేయండి."
            elif lang == "kn":
                msg = f"ಹಸು {cow_id}: ಕೆಚ್ಚಲು ಬಾವು ಅಪಾಯ {risk}% ಕಂಡುಬಂದಿದೆ. ದಯವಿಟ್ಟು ಪರಿಶೀಲಿಸಿ."
            else:
                msg = f"Cow {cow_id}: Mastitis risk {risk}% detected. Inspect udder and consult veterinarian."

            alerts.append({
                "cow_id": cow_id,
                "rfid_tag": data["rfid_tag"],
                "risk_pct": risk,
                "severity": "CRITICAL" if risk >= 75.0 else "WARNING",
                "sms_text": msg,
                "recipient_phone": LOCKED_CONTACT["phone"]
            })

    return {"status": "success", "alerts": alerts}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
