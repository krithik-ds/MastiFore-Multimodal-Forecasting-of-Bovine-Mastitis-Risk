from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import math
from pathlib import Path

# ============================================================
# SENTRY 2 - FastAPI Sensor Server
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

from ml_pipeline.mastitis_model import MastitisPredictiveEngine


app = FastAPI(
    title="SENTRY 2 API",
    description="Live ESP32 + simulated milk data + mastitis risk engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# JSON SAFETY
# ============================================================

def make_json_safe(obj):
    """
    Convert NumPy/Pandas scalar values into normal Python
    values so FastAPI can serialize the response as JSON.
    """
    if obj is None:
        return None

    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass

    if isinstance(obj, dict):
        return {
            str(key): make_json_safe(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [make_json_safe(value) for value in obj]

    return obj


# ============================================================
# DATA
# ============================================================

cow_data = pd.read_csv(BASE_DIR / "cow_data.csv")
ph_data = pd.read_csv(BASE_DIR / "ph.csv")
milk_data = pd.read_csv(BASE_DIR / "milk_yield.csv")

data_index = 0

# Latest received/processed packet for live monitoring.
# This does not modify the ML model or its logic.
latest_sensor_state = None

# Backend-selected cow.
# ESP32 does NOT need to send cow_id.
active_cow_id = "COW_001"

# Load the existing risk engine from the supplied project.
mastitis_engine = MastitisPredictiveEngine()


# ============================================================
# REQUEST MODELS
# ============================================================

class CowSelection(BaseModel):
    cow_id: str


class SensorData(BaseModel):
    device_id: str
    temperature: float
    tds_raw: int
    tds_voltage: float


# ============================================================
# SENSOR CALCULATIONS
# ============================================================

def calculate_tds(voltage: float, temperature: float) -> float:
    """
    Standard TDS-meter conversion used by the current prototype.
    Temperature-compensated voltage -> ppm.
    """
    compensation_voltage = voltage / (
        1 + 0.02 * (temperature - 25.0)
    )

    tds = (
        133.42 * compensation_voltage ** 3
        - 255.86 * compensation_voltage ** 2
        + 857.39 * compensation_voltage
    ) * 0.5

    return max(0.0, float(tds))


def calculate_conductivity(tds: float) -> float:
    """
    Convert TDS ppm to conductivity in mS/cm using the
    current prototype's 0.5 conversion factor.
    """
    return float((tds / 0.5) / 1000.0)


# ============================================================
# BASIC ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "project": "SENTRY 2",
        "status": "online",
        "active_cow_id": active_cow_id,
    }


@app.get("/cows")
def get_cows():
    records = cow_data.to_dict(orient="records")

    return make_json_safe({
        "success": True,
        "cows": records,
    })


@app.get("/active-cow")
def get_active_cow():
    return {
        "success": True,
        "active_cow_id": active_cow_id,
    }


@app.post("/select-cow")
def select_cow(selection: CowSelection):
    global active_cow_id

    cow_id = selection.cow_id.strip()

    if cow_id not in cow_data["cow_id"].astype(str).values:
        raise HTTPException(
            status_code=404,
            detail=f"Cow ID '{cow_id}' not found in cow_data.csv",
        )

    active_cow_id = cow_id

    return {
        "success": True,
        "active_cow_id": active_cow_id,
    }


# ============================================================
# LIVE SENSOR MONITOR
# ============================================================

@app.get("/latest-sensor")
def latest_sensor():
    if latest_sensor_state is None:
        return {
            "success": True,
            "message": "No sensor data received yet."
        }

    return make_json_safe(latest_sensor_state)


# ============================================================
# LIVE SENSOR ENDPOINT
# ============================================================

@app.post("/sensor-data")
def receive_sensor_data(data: SensorData):
    global data_index, latest_sensor_state

    # --------------------------------------------------------
    # 1. Find currently selected cow
    # --------------------------------------------------------
    selected_rows = cow_data[
        cow_data["cow_id"].astype(str) == active_cow_id
    ]

    if selected_rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Active cow '{active_cow_id}' not found",
        )

    cow = selected_rows.iloc[0]

    # --------------------------------------------------------
    # 2. Loop simulated CSV data
    # --------------------------------------------------------
    ph_value = float(
        ph_data.iloc[data_index % len(ph_data)]["ph"]
    )

    milk_yield = float(
        milk_data.iloc[data_index % len(milk_data)]["milk_yield"]
    )

    data_index += 1

    # --------------------------------------------------------
    # 3. Convert live TDS -> conductivity
    # --------------------------------------------------------
    estimated_tds = calculate_tds(
        data.tds_voltage,
        data.temperature,
    )

    conductivity = calculate_conductivity(
        estimated_tds
    )

    # --------------------------------------------------------
    # 4. Master 18-feature schema
    #
    # Permanent schema:
    #   1  breed_idx
    #   2  age_years
    #   3  lactation_num
    #   4  vaccination_status
    #   5  prior_mastitis_hist
    #   6  thi_index
    #   7  hygiene_score
    #   8  concentrate_kg
    #   9  water_intake_l
    #   10 body_temp
    #   11 rumination_mins
    #   12 spine_angle
    #   13 milk_yield_l
    #   14 ec_ms_cm
    #   15 milk_ph
    #   16 log_scc
    #   17 tier1_cv_score
    #   18 tier2_cmt_score
    #
    # Only currently available values are populated.
    # --------------------------------------------------------

    features = {
        "breed_idx": int(cow["breed_idx"]),
        "age_years": float(cow["age_years"]),
        "lactation_num": int(cow["lactation_num"]),
        "vaccination_status": int(cow["vaccination_status"]),
        "prior_mastitis_hist": int(cow["prior_mastitis_hist"]),
        "thi_index": None,
        "hygiene_score": float(cow["hygiene_score"]),
        "concentrate_kg": None,
        "water_intake_l": None,
        "body_temp": float(data.temperature),
        "rumination_mins": None,
        "spine_angle": float(cow["spine_angle"]),
        "milk_yield_l": milk_yield,
        "ec_ms_cm": conductivity,
        "milk_ph": ph_value,
        "log_scc": None,
        "tier1_cv_score": None,
        "tier2_cmt_score": None,
    }

    # --------------------------------------------------------
    # 5. Track which features are actually available
    # --------------------------------------------------------
    available_features = {
        "cow_id": active_cow_id,
        "breed_idx": features["breed_idx"],
        "age_years": features["age_years"],
        "lactation_num": features["lactation_num"],
        "vaccination_status": features["vaccination_status"],
        "prior_mastitis_hist": features["prior_mastitis_hist"],
        "hygiene_score": features["hygiene_score"],
        "body_temp": features["body_temp"],
        "spine_angle": features["spine_angle"],
        "milk_yield_l": features["milk_yield_l"],
        "ec_ms_cm": features["ec_ms_cm"],
        "milk_ph": features["milk_ph"],
    }

    # --------------------------------------------------------
    # 6. Prediction profile
    #
    # Do NOT send unavailable values as fake measurements.
    # The existing engine supplies its own defaults where
    # required by its current rule-based implementation.
    # --------------------------------------------------------

    prediction_profile = {
        "cow_id": active_cow_id,
        "breed": "Unknown",

        "breed_idx": features["breed_idx"],
        "age_years": features["age_years"],
        "lactation_num": features["lactation_num"],
        "vaccination_status": features["vaccination_status"],
        "prior_mastitis_hist": features["prior_mastitis_hist"],
        "hygiene_score": features["hygiene_score"],

        "body_temp": features["body_temp"],
        "spine_angle": features["spine_angle"],
        "milk_yield_l": features["milk_yield_l"],
        "ec_ms_cm": features["ec_ms_cm"],
        "milk_ph": features["milk_ph"],
    }

    # --------------------------------------------------------
    # 7. Run mastitis prediction
    # --------------------------------------------------------
    prediction = mastitis_engine.predict_cow_risk(
        prediction_profile
    )

    prediction = make_json_safe(prediction)

    # --------------------------------------------------------
    # 8. Console output
    # --------------------------------------------------------
    print()
    print("==============================")
    print("       SENTRY 2 LIVE DATA")
    print("==============================")
    print(f"Cow ID       : {active_cow_id}")
    print(f"Device ID    : {data.device_id}")
    print(f"Temperature  : {data.temperature:.2f} °C")
    print(f"TDS Raw      : {data.tds_raw}")
    print(f"TDS Voltage  : {data.tds_voltage:.6f} V")
    print(f"Estimated TDS: {estimated_tds:.2f} ppm")
    print(f"Conductivity : {conductivity:.4f} mS/cm")
    print(f"pH           : {ph_value:.2f}")
    print(f"Milk Yield   : {milk_yield:.2f} L")

    print()
    print("AVAILABLE FEATURES")
    print("------------------------------")

    for key, value in available_features.items():
        print(f"{key}: {value}")

    print()
    print("==============================")
    print("       MASTITIS RISK")
    print("==============================")
    print(
        f"Risk     : "
        f"{prediction.get('overall_mastitis_risk_pct')}%"
    )
    print(
        f"Status   : "
        f"{prediction.get('health_status')}"
    )
    print(
        f"Urgency  : "
        f"{prediction.get('action_urgency')}"
    )
    print(
        f"Forecast : "
        f"{prediction.get('forecast_days_to_clinical_onset')}"
    )

    factors = prediction.get("top_contributing_factors", [])

    if factors:
        print()
        print("Risk factors:")

        for factor in factors:
            print(f" - {factor}")

    print("==============================")

    # --------------------------------------------------------
    # 9. Return JSON-safe response
    # --------------------------------------------------------
    response = {
        "success": True,

        "cow_id": active_cow_id,

        "device_id": data.device_id,

        "live_sensor_data": {
            "temperature": data.temperature,
            "tds_raw": data.tds_raw,
            "tds_voltage": data.tds_voltage,
            "estimated_tds_ppm": estimated_tds,
            "conductivity_ms_cm": conductivity,
        },

        "simulated_data": {
            "milk_ph": ph_value,
            "milk_yield_l": milk_yield,
        },

        "cow_database": {
            "cow_id": active_cow_id,
            "breed_idx": cow["breed_idx"],
            "age_years": cow["age_years"],
            "lactation_num": cow["lactation_num"],
            "vaccination_status": cow["vaccination_status"],
            "prior_mastitis_hist": cow["prior_mastitis_hist"],
            "hygiene_score": cow["hygiene_score"],
            "spine_angle": cow["spine_angle"],
        },

        "features": features,

        "available_features": available_features,

        "prediction": prediction,
    }

    latest_sensor_state = response
    return make_json_safe(response)
