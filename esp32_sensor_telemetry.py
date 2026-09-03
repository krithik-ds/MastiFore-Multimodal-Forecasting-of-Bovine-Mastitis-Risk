"""
Tier-2 ESP32 Multi-Sensor Telemetry Simulator
SIH 26109: Bovine Mastitis Early Forecasting System

Aligns with DPR Section 9 & 10:
- Milk Temperature (°C)
- Milk pH (pH units)
- Milk Electrical Conductivity (EC in mS/cm)
- Milk Yield (Liters/session)
- Udder Temperature (°C, Optional surface thermal probe)
- RFID Tag Linkage
"""

import json
import time
import random
from typing import Dict, Any


class ESP32SensorTelemetrySimulator:
    def __init__(self, node_id: str = "ESP32_MILKING_BAY_01"):
        self.node_id = node_id

    def read_cow_sensors(self, cow_id: str, rfid_tag: str, simulated_condition: str = "healthy") -> Dict[str, Any]:
        """
        Simulates precision sensor acquisition at the milking stall for a shortlisted cow.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if simulated_condition == "healthy":
            # Normal Milk Properties
            ec = round(random.uniform(4.30, 4.85), 2)         # Normal: 4.0 - 5.0 mS/cm
            ph = round(random.uniform(6.50, 6.68), 2)         # Normal: 6.5 - 6.7 pH
            milk_temp = round(random.uniform(38.2, 38.8), 1)  # Normal internal body temp: 38.5°C
            scc_proxy = random.randint(60000, 150000)         # Normal SCC < 200,000 cells/mL
            milk_yield = round(random.uniform(14.0, 18.5), 1) # Normal yield
            udder_temp = round(random.uniform(35.5, 36.8), 1) # Normal surface skin temp

        elif simulated_condition == "subclinical":
            # Subclinical Mastitis (Mild ionic leakage, early alkaline shift)
            ec = round(random.uniform(5.60, 6.30), 2)         # Elevated EC: > 5.5 mS/cm
            ph = round(random.uniform(6.78, 6.95), 2)         # Alkaline shift: > 6.75 pH
            milk_temp = round(random.uniform(38.9, 39.4), 1)  # Mild inflammation temp
            scc_proxy = random.randint(300000, 750000)        # Elevated SCC: 250k - 750k cells/mL
            milk_yield = round(random.uniform(10.5, 13.0), 1) # 10-25% drop in yield
            udder_temp = round(random.uniform(37.2, 38.2), 1) # Localized warmth

        else: # clinical
            # Acute Clinical Mastitis
            ec = round(random.uniform(6.60, 7.80), 2)         # Very High EC: > 6.5 mS/cm
            ph = round(random.uniform(7.05, 7.45), 2)         # Distinct Alkaline: > 7.0 pH
            milk_temp = round(random.uniform(39.6, 40.5), 1)  # High fever/heat
            scc_proxy = random.randint(1200000, 3500000)      # High SCC > 1,000,000 cells/mL
            milk_yield = round(random.uniform(5.0, 8.5), 1)   # Severe yield drop (>40%)
            udder_temp = round(random.uniform(38.8, 40.0), 1) # Hot/Swollen surface

        telemetry_packet = {
            "node_id": self.node_id,
            "rfid_tag": rfid_tag,
            "cow_id": cow_id,
            "timestamp": timestamp,
            "sensors": {
                "milk_temperature_c": milk_temp,
                "milk_ph": ph,
                "milk_ec_ms_cm": ec,
                "milk_yield_liters": milk_yield,
                "udder_surface_temp_c": udder_temp,
                "scc_proxy_cells_ml": scc_proxy
            },
            "firmware_version": "ESP32-Dhenu-v2.4",
            "battery_pct": 96,
            "signal_rssi_dbm": -62
        }

        return telemetry_packet


if __name__ == "__main__":
    sim = ESP32SensorTelemetrySimulator()
    packet = sim.read_cow_sensors("COW_103", "RFID-IN-PB-3390", "subclinical")
    print(json.dumps(packet, indent=2))
