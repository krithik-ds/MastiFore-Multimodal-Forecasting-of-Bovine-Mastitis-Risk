import time
from typing import Dict, Any, List
from collections import defaultdict


class CrossCameraFusionEngine:
    """
    Fuses observations across multiple barn CCTV camera streams.
    Tracks persistent cow identity and selects the clearest view for rumination and posture.
    """

    def __init__(self):
        self.memory = defaultdict(lambda: {
            "cow_id": "",
            "rumination_cpm": 0.0,
            "chews_per_bolus": 48,
            "inter_bolus_interval_sec": 4.2,
            "jaw_motion_amplitude": 0.0,
            "total_chews": 0,
            "total_rumination_seconds": 0.0,
            "is_eating": False,
            "spine_kyphosis_angle": 175.0,
            "spine_posture_label": "Normal Flat Topline",
            "is_spine_arched": False,
            "posture": "Standing",
            "posture_confidence": 96,
            "active_cam_sources": [],
            "last_updated": 0
        })

    def fuse_n_camera_streams(
        self,
        camera_streams_data: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        now = time.time()
        active_cows = set()

        for cam_name, detections in camera_streams_data.items():
            for d in detections:
                cow_id = d.get("cow_id")
                if not cow_id:
                    continue

                active_cows.add(cow_id)
                record = self.memory[cow_id]
                record["cow_id"] = cow_id
                record["last_updated"] = now

                if cam_name not in record["active_cam_sources"]:
                    record["active_cam_sources"].append(cam_name)

                # Select best jaw / chewing tracking
                if d.get("is_eating", False) or d.get("chews_per_min", 0) > record["rumination_cpm"]:
                    record["rumination_cpm"] = d.get("chews_per_min", record["rumination_cpm"])
                    record["chews_per_bolus"] = d.get("chews_per_bolus", record["chews_per_bolus"])
                    record["inter_bolus_interval_sec"] = d.get("inter_bolus_interval_sec", record["inter_bolus_interval_sec"])
                    record["jaw_motion_amplitude"] = d.get("jaw_motion_amplitude", record["jaw_motion_amplitude"])
                    record["total_chews"] = max(record["total_chews"], d.get("total_chews", 0))
                    record["total_rumination_seconds"] = max(record["total_rumination_seconds"], d.get("total_rumination_seconds", 0.0))
                    record["is_eating"] = True

                # Select best unobstructed lateral spine view
                if d.get("spine_kyphosis_angle") is not None:
                    record["spine_kyphosis_angle"] = d["spine_kyphosis_angle"]
                    record["spine_posture_label"] = d.get("spine_posture_label", "Normal Flat Topline")
                    record["is_spine_arched"] = d.get("is_spine_arched", False)

                # Prioritize resting posture if observed in cubicles
                if d.get("posture") == "Resting":
                    record["posture"] = "Resting"
                    record["posture_confidence"] = d.get("posture_confidence", 94)
                elif record["posture"] != "Resting":
                    record["posture"] = d.get("posture", "Standing")
                    record["posture_confidence"] = d.get("posture_confidence", 96)

        results = []
        for cow_id in sorted(active_cows):
            rec = self.memory[cow_id]
            is_suspect = (rec["rumination_cpm"] < 35.0 and rec["total_chews"] > 0) or rec["is_spine_arched"]

            results.append({
                "cow_id": cow_id,
                "contributing_cameras": rec["active_cam_sources"],
                "posture": rec["posture"],
                "posture_confidence": rec["posture_confidence"],
                "spine_kyphosis_angle": rec["spine_kyphosis_angle"],
                "spine_posture_label": rec["spine_posture_label"],
                "is_spine_arched": rec["is_spine_arched"],
                "rumination_cpm": rec["rumination_cpm"],
                "chews_per_bolus": rec["chews_per_bolus"],
                "inter_bolus_interval_sec": rec["inter_bolus_interval_sec"],
                "jaw_motion_amplitude": rec["jaw_motion_amplitude"],
                "total_chews": rec["total_chews"],
                "total_rumination_seconds": rec["total_rumination_seconds"],
                "tier1_status": "SUSPECT_TIER2" if is_suspect else "NORMAL_TIER1"
            })

        return results

    def fuse_camera_frames(self, cam1_diags, cam2_diags):
        return self.fuse_n_camera_streams({"Cam_1": cam1_diags, "Cam_2": cam2_diags})
