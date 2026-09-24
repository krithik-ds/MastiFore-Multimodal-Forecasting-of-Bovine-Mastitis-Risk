import time
import random
import json
from typing import Dict, Any


class ESP32SensorTelemetrySimulator:
    """
    Simulates sensor data collected by an ESP32 microcontroller at the milking parlor.
    Captures conductivity, pH, temperature, and milk yield for shortlisted cows.
    """

    def __init__(self, node_id: str = "ESP32_BAY_01"):
        self.node_id = node_id

    def read_cow_sensors(self, cow_id: str, rfid_tag: str, condition: str = "healthy") -> Dict[str, Any]:
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        if condition == "healthy":
            ec = round(random.uniform(4.30, 4.85), 2)
            ph = round(random.uniform(6.50, 6.68), 2)
            milk_temp = round(random.uniform(38.2, 38.8), 1)
            milk_yield = round(random.uniform(14.0, 18.5), 1)
            clotting = 0
            udder_temp = round(random.uniform(35.5, 36.8), 1)

        elif condition == "subclinical":
            ec = round(random.uniform(5.60, 6.30), 2)
            ph = round(random.uniform(6.78, 6.95), 2)
            milk_temp = round(random.uniform(38.9, 39.4), 1)
            milk_yield = round(random.uniform(10.5, 13.0), 1)
            clotting = 0
            udder_temp = round(random.uniform(37.2, 38.2), 1)

        else:  # clinical
            ec = round(random.uniform(6.60, 7.80), 2)
            ph = round(random.uniform(7.05, 7.45), 2)
            milk_temp = round(random.uniform(39.6, 40.5), 1)
            milk_yield = round(random.uniform(5.0, 8.5), 1)
            clotting = 1
            udder_temp = round(random.uniform(38.8, 40.0), 1)

        return {
            "node_id": self.node_id,
            "cow_id": cow_id,
            "rfid_tag": rfid_tag,
            "timestamp": now,
            "sensors": {
                "milk_temperature_c": milk_temp,
                "milk_ph": ph,
                "milk_ec_ms_cm": ec,
                "milk_yield_liters": milk_yield,
                "clotting_flocculation": clotting,
                "udder_temp_c": udder_temp
            },
            "rssi": -65,
            "battery_level": 95
        }


    def generate_v1_cow_registration(self, cow_id: str = "8492", rfid_uid: str = "A1:B2:C3:D4") -> Dict[str, Any]:
        """Generates payload matching ESP32 firmware POST /api/v1/cows."""
        return {
            "device_id": self.node_id,
            "cow_id": str(cow_id),
            "rfid_uid": rfid_uid,
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def generate_v1_test_payload(
        self,
        cow_id: str = "8492",
        rfid_uid: str = "A1:B2:C3:D4",
        quarter: int = 1,
        condition: str = "healthy",
        sequence_id: int = 1,
        delivery_mode: str = "live"
    ) -> Dict[str, Any]:
        """Generates payload matching ESP32 firmware POST /api/v1/tests."""
        if condition == "healthy":
            ec = round(random.uniform(4.30, 4.85), 2)
            ph = round(random.uniform(6.50, 6.68), 2)
            temp = round(random.uniform(38.2, 38.7), 1)
        elif condition == "subclinical":
            ec = round(random.uniform(5.50, 6.40), 2)
            ph = round(random.uniform(6.75, 6.95), 2)
            temp = round(random.uniform(38.8, 39.4), 1)
        else:  # clinical
            ec = round(random.uniform(6.50, 8.20), 2)
            ph = round(random.uniform(7.00, 7.45), 2)
            temp = round(random.uniform(39.5, 40.5), 1)

        test_id = f"{self.node_id}_{sequence_id:06d}"
        return {
            "test_id": test_id,
            "device_id": self.node_id,
            "cow_id": str(cow_id),
            "rfid_uid": rfid_uid,
            "quarter": quarter,
            "ec": ec,
            "ph": ph,
            "temperature": temp,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sequence_id": sequence_id,
            "time_source": "rtc",
            "delivery_mode": delivery_mode
        }


if __name__ == "__main__":
    sim = ESP32SensorTelemetrySimulator("ESP32_01")
    sample_reg = sim.generate_v1_cow_registration("8492", "KEYPAD_8492")
    sample_test = sim.generate_v1_test_payload("8492", "KEYPAD_8492", quarter=2, condition="subclinical")
    print("Sample Cow Registration:\n", json.dumps(sample_reg, indent=2))
    print("\nSample Test Ingestion:\n", json.dumps(sample_test, indent=2))
