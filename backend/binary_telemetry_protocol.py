import struct
import time
import os
from typing import List, Dict, Any

# 16-Byte Compact Binary Telemetry Packet Specification for Tier 1 (CCTV / Canny Offline Edge):
# < : Little-endian
# I : uint32 (timestamp in seconds, 4 bytes)
# H : uint16 (cow_id as numeric integer, 2 bytes)
# B : uint8  (chews_per_minute: 0-255, 1 byte)
# B : uint8  (spine_angle_deg: 0-255, 1 byte)
# B : uint8  (posture_flags: 0=Lying, 1=Standing, 2=Eating, 3=Arched, 1 byte)
# B : uint8  (optical_flow_amp: 0-255, 1 byte)
# H : uint16 (total_chews: 0-65535, 2 bytes)
# B : uint8  (confidence_pct: 0-100, 1 byte)
# 3s: char[3] (CRC checksum / reserved, 3 bytes)
TIER1_PACKET_FORMAT = "<I H B B B B H B 3s"
TIER1_PACKET_SIZE = struct.calcsize(TIER1_PACKET_FORMAT)  # exactly 16 bytes

# 16-Byte Compact Binary Telemetry Packet Specification for Tier 2 (ESP32 Milking Sensors):
# I : uint32 (timestamp, 4 bytes)
# H : uint16 (cow_id, 2 bytes)
# B : uint8  (quarter: 1=FL, 2=FR, 3=RL, 4=RR, 1 byte)
# H : uint16 (ec_scaled: ec * 100, e.g. 5.42 -> 542, 2 bytes)
# H : uint16 (ph_scaled: ph * 100, e.g. 6.65 -> 665, 2 bytes)
# H : uint16 (temp_scaled: temp * 100, e.g. 38.5 -> 3850, 2 bytes)
# 3s: char[3] (reserved, 3 bytes)
TIER2_PACKET_FORMAT = "<I H B H H H 3s"
TIER2_PACKET_SIZE = struct.calcsize(TIER2_PACKET_FORMAT)  # exactly 16 bytes


def encode_tier1_packet(
    timestamp: int,
    cow_id: int,
    cpm: int,
    spine_angle: int,
    posture_flags: int = 1,
    optical_flow_amp: int = 0,
    total_chews: int = 0,
    confidence_pct: int = 95
) -> bytes:
    """Serializes Tier-1 CV observation into a compact 16-byte binary packet."""
    cpm_b = max(0, min(255, int(cpm)))
    spine_b = max(0, min(255, int(spine_angle)))
    posture_b = max(0, min(255, int(posture_flags)))
    flow_b = max(0, min(255, int(optical_flow_amp)))
    total_b = max(0, min(65535, int(total_chews)))
    conf_b = max(0, min(100, int(confidence_pct)))
    
    return struct.pack(
        TIER1_PACKET_FORMAT,
        int(timestamp),
        int(cow_id),
        cpm_b,
        spine_b,
        posture_b,
        flow_b,
        total_b,
        conf_b,
        b"\x00\x00\x00"
    )


def decode_tier1_packets(binary_data: bytes) -> List[Dict[str, Any]]:
    """Decodes a stream of 16-byte Tier-1 binary packets into structured dictionaries."""
    records = []
    total_len = len(binary_data)
    for offset in range(0, total_len - TIER1_PACKET_SIZE + 1, TIER1_PACKET_SIZE):
        chunk = binary_data[offset:offset + TIER1_PACKET_SIZE]
        ts, cid, cpm, spine, posture, flow, chews, conf, _ = struct.unpack(TIER1_PACKET_FORMAT, chunk)
        
        posture_labels = {0: "Lying", 1: "Standing", 2: "Eating", 3: "Arched Kyphosis"}
        is_arched = (spine < 160) or (posture == 3)
        
        records.append({
            "timestamp": ts,
            "cow_id": f"COW_{cid}",
            "numeric_id": cid,
            "chews_per_min": float(cpm),
            "spine_kyphosis_angle": float(spine),
            "is_spine_arched": is_arched,
            "posture": posture_labels.get(posture, "Standing"),
            "optical_flow_amplitude": flow,
            "total_chews": chews,
            "confidence_pct": conf,
            "is_shortlisted": (cpm < 45) or is_arched
        })
    return records


def encode_tier2_packet(
    timestamp: int,
    cow_id: int,
    quarter: int,
    ec: float,
    ph: float,
    temperature: float
) -> bytes:
    """Serializes Tier-2 ESP32 sensor reading into a 16-byte binary packet."""
    ec_scaled = int(round(ec * 100))
    ph_scaled = int(round(ph * 100))
    temp_scaled = int(round(temperature * 100))
    
    return struct.pack(
        TIER2_PACKET_FORMAT,
        int(timestamp),
        int(cow_id),
        int(quarter),
        ec_scaled,
        ph_scaled,
        temp_scaled,
        b"\x00\x00\x00"
    )


def decode_tier2_packets(binary_data: bytes) -> List[Dict[str, Any]]:
    """Decodes a stream of 16-byte Tier-2 binary packets into structured sensor readings."""
    records = []
    total_len = len(binary_data)
    for offset in range(0, total_len - TIER2_PACKET_SIZE + 1, TIER2_PACKET_SIZE):
        chunk = binary_data[offset:offset + TIER2_PACKET_SIZE]
        ts, cid, qtr, ec_s, ph_s, temp_s, _ = struct.unpack(TIER2_PACKET_FORMAT, chunk)
        
        q_map = {1: "Q1: Front-Left (FL)", 2: "Q2: Front-Right (FR)", 3: "Q3: Rear-Left (RL)", 4: "Q4: Rear-Right (RR)"}
        ec = round(ec_s / 100.0, 2)
        ph = round(ph_s / 100.0, 2)
        temp = round(temp_s / 100.0, 2)
        
        records.append({
            "timestamp": ts,
            "cow_id": f"COW_{cid}",
            "numeric_id": cid,
            "quarter": qtr,
            "quarter_label": q_map.get(qtr, f"Quarter {qtr}"),
            "ec": ec,
            "ph": ph,
            "temperature": temp
        })
    return records
