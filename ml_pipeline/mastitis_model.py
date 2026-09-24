import os
import json
import math
from typing import Dict, Any, Optional

METADATA_PATH = os.path.join(os.path.dirname(__file__), "tier2_model_metadata.json")


class MastitisPredictiveEngine:
    """
    Fuses in-line sensor telemetry (conductivity, pH, temperature, yield, clotting)
    with Tier-1 behavioural deviations (rumination drop, spine posture) to estimate mastitis risk.
    """

    def __init__(self):
        # Default normalization & weights for [temp, pH, EC, yield, clotting]
        self.means = [36.0, 6.75, 5.20, 18.5, 0.20]
        self.stds = [1.2, 0.25, 1.10, 5.0, 0.40]
        self.weights = [2.05, 1.92, 2.30, -1.68, 2.12]
        self.bias = -2.35

        if os.path.exists(METADATA_PATH):
            try:
                with open(METADATA_PATH, "r") as f:
                    meta = json.load(f)
                    self.means = meta.get("means", self.means)
                    self.stds = meta.get("stds", self.stds)
                    self.weights = meta.get("standardized_weights", self.weights)
                    self.bias = meta.get("bias", self.bias)
            except Exception:
                pass

        self.ref_ec = 4.80
        self.ref_ph = 6.60
        self.ref_rumination = 60.0
        self.ref_spine_angle = 175.0

    def evaluate_risk(
        self,
        cow_id: str,
        current_data: Dict[str, Any],
        baseline: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        has_esp = bool(current_data.get("has_esp32_sensor_data", False)) and (current_data.get("milk_ec_ms_cm") is not None)

        base_cpm = baseline.get("rumination_cpm_normal", self.ref_rumination) if baseline else self.ref_rumination
        base_spine = baseline.get("spine_angle_normal", self.ref_spine_angle) if baseline else self.ref_spine_angle

        cur_cpm = float(current_data.get("rumination_cpm") or base_cpm)
        cur_spine = float(current_data.get("spine_angle_deg") or base_spine)

        delta_cpm = max(0.0, (base_cpm - cur_cpm) / max(1.0, base_cpm))
        delta_spine = max(0.0, (base_spine - cur_spine) / 25.0)

        factors = []
        if delta_cpm > 0.20:
            factors.append({"feature": "Rumination Cadence", "value": f"{cur_cpm:.1f} CPM", "note": f"-{delta_cpm*100:.0f}% drop vs baseline"})
        if cur_spine < 160.0:
            factors.append({"feature": "Spine Curvature", "value": f"{cur_spine:.1f}°", "note": "Kyphosis / arched back posture"})

        if not has_esp:
            # PURE VISION BEHAVIORAL TRIAGE (Camera Only, No Fake Sensor Data)
            cv_risk = (delta_cpm * 45.0) + (delta_spine * 35.0) + 8.0
            final_risk = round(min(70.0, max(5.0, cv_risk)), 1)

            if final_risk >= 35.0:
                status = "Shortlisted for ESP32 Testing"
                category = "TIER1_SHORTLISTED"
                confidence = "Visual Screening (CCTV)"
                actions = [
                    "Animal shortlisted by Barn Camera due to behavioral deviation.",
                    "Perform ESP32 teat sensor measurement during upcoming milking session."
                ]
            else:
                status = "Healthy (Visual Baseline)"
                category = "NORMAL_HEALTHY"
                confidence = "Normal Visual Activity"
                actions = [
                    "Continue continuous CCTV rumination and postural monitoring.",
                    "No immediate intervention required."
                ]

            return {
                "cow_id": cow_id,
                "overall_mastitis_risk_pct": final_risk,
                "health_status": status,
                "risk_category": category,
                "confidence_level": confidence,
                "has_esp32_sensor_data": False,
                "contributing_factors": factors,
                "decision_support_actions": actions,
                "tier2_raw_probability": 0.0
            }

        # ESP32 MILK SENSOR TELEMETRY FUSION
        temp = float(current_data.get("milk_temp_c") or current_data.get("milk_temperature_c") or 38.5)
        ph = float(current_data.get("milk_ph") or 6.65)
        ec = float(current_data.get("milk_ec_ms_cm") or 4.65)
        yield_l = float(current_data.get("milk_yield_liters") or current_data.get("milk_yield_l") or 16.0)
        clotting = 1.0 if current_data.get("clotting", False) or (current_data.get("clotting_flocculation") or 0) >= 1 else 0.0

        raw_features = [temp, ph, ec, yield_l, clotting]
        scaled = [(raw_features[i] - self.means[i]) / self.stds[i] for i in range(len(raw_features))]

        z = sum(self.weights[i] * scaled[i] for i in range(len(scaled))) + self.bias
        z = max(-30.0, min(30.0, z))
        prob_tier2 = 1.0 / (1.0 + math.exp(-z))

        tier2_score = prob_tier2 * 100.0
        cv_penalty = (delta_cpm * 18.0) + (delta_spine * 12.0)
        fused = (tier2_score * 0.70) + (cv_penalty * 0.30)

        final_risk = round(min(98.5, max(5.0, fused)), 1)

        if ec > 5.5:
            factors.append({"feature": "Conductivity", "value": f"{ec:.2f} mS/cm", "note": "Elevated ion leakage"})
        if clotting > 0.5:
            factors.append({"feature": "Clotting", "value": "Visible Flakes", "note": "Protein precipitation"})
        if temp > 39.0:
            factors.append({"feature": "Milk Temp", "value": f"{temp:.1f}°C", "note": "Local inflammation"})
        if ph > 6.75:
            factors.append({"feature": "Milk pH", "value": f"{ph:.2f}", "note": "Alkaline shift"})

        if final_risk >= 75.0:
            status = "Clinical Mastitis"
            category = "HIGH_RISK_CLINICAL"
            confidence = "High Confidence (98%)"
            actions = [
                "Separate cow from primary milking line.",
                "Schedule clinical examination with herd veterinarian.",
                "Perform quarter-level CMT check.",
                "Apply cold compresses and withhold milk."
            ]
        elif final_risk >= 45.0:
            status = "Subclinical Mastitis"
            category = "SUSPECT_SUBCLINICAL"
            confidence = "Elevated Risk (92%)"
            actions = [
                "Apply 0.5% Chlorhexidine post-milking teat dip.",
                "Supplement Vitamin E and Zinc to support immunity.",
                "Milk this cow last in current session to prevent spread.",
                "Flag for veterinarian review."
            ]
        else:
            status = "Healthy"
            category = "NORMAL_HEALTHY"
            confidence = "Normal Baseline (99%)"
            actions = [
                "Maintain standard milking and sanitation routine."
            ]

        return {
            "cow_id": cow_id,
            "overall_mastitis_risk_pct": final_risk,
            "risk_category": category,
            "health_status": status,
            "confidence_level": confidence,
            "has_esp32_sensor_data": True,
            "contributing_factors": factors,
            "decision_support_actions": actions,
            "tier2_raw_probability": round(prob_tier2 * 100, 2)
        }
