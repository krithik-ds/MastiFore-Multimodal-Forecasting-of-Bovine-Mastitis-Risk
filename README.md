# SIH 26109: AI-Based Early Forecasting of Bovine Mastitis in Indian Dairy Farms

**Theme:** Agriculture, FoodTech & Rural Development  
**Sponsoring Ministry:** Ministry of Fisheries, Animal Husbandry & Dairying  
**Edition:** Hardware / IoT + AI/ML Software Edition  

---

## 🚀 3-Tier Funnel Architecture

Instead of putting expensive IoT hardware on every single cow (which is economically infeasible for Indian farmers), our system implements a **cost-effective 3-tier triage funnel**:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: Zero-Touch Computer Vision Screening (Barn CCTV / RGB Camera)         │
│ • Rhythmic Jaw Optical Flow: Analyzes chews/minute (Rumination drop detection) │
│ • Spine Kyphosis & Arching: Measures dorsal curvature (< 155° indicates pain)  │
│ • Posture Classification: Standing, Resting/Lying, Feeding, Lethargy           │
│ ➔ Output: Filters herd down to "Suspect Cows" (~20-30%)                       │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │ Filtered Suspect Cows
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ TIER 2: Rapid Field / Manual Milk Testing (Spot Check)                         │
│ • California Mastitis Test (CMT Grade 0 - 4), Visual Clot / Discoloration     │
│ • 1-tap mobile logging by farm worker                                          │
│ ➔ Output: Filters suspects down to "High-Risk Cattle" (~5-10%)                 │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │ High-Risk Cows Only
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ TIER 3: Precision IoT Sensor Testing & Multi-Parameter AI Forecasting         │
│ • In-line / Handheld Sensor Probe: EC (mS/cm), pH, Milk Temp, SCC proxy        │
│ • AI/ML Model: Predicts Subclinical Mastitis 3 to 7 Days in Advance           │
│ • Output: Vernacular SMS/WhatsApp alerts + Non-antibiotic proactive care plan  │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
SIH_Project/
├── cv_module/
│   ├── cow_vision_triage.py        # Main CV pipeline (YOLOv8 + Optical Flow + Spine Kyphosis)
│   └── generate_demo_frames.py     # Generates simulated barn video for testing
├── ml_pipeline/
│   └── mastitis_model.py           # Multi-parameter ML predictive model (7 feature vectors)
├── backend/
│   └── server.py                   # FastAPI backend server with triage & vernacular alert endpoints
├── hardware_sim/
│   └── esp32_sensor_telemetry.py   # Simulates ESP32/Arduino IoT milk probe telemetry
├── web_dashboard/
│   └── index.html                  # Interactive 3-Tier Dashboard with charts & vernacular support
├── Cow_Behavior_Detection.ipynb    # Fixed & updated standalone Jupyter Notebook
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🛠️ Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Tier-1 Computer Vision on Simulated Cattle Barn
```bash
# Generate synthetic cattle video
python cv_module/generate_demo_frames.py

# Run CV behavior & spine triage
python cv_module/cow_vision_triage.py demo_barn_feed.mp4 triage_output.mp4
```

### 3. Run AI Predictive Model Test
```bash
python ml_pipeline/mastitis_model.py
```

### 4. Launch the Interactive Web Dashboard
Simply open `web_dashboard/index.html` in any web browser (Chrome, Edge, Safari, Firefox) to explore:
- Live Computer Vision video stream simulation & HUD
- Real-time rumination & spine angle degradation charts
- Herd triage status & individual cow health profiles
- Multilingual support (English, हिंदी, मराठी)
- One-click vernacular WhatsApp/SMS alerts

### 5. (Optional) Run FastAPI Backend & IoT Telemetry Simulator
```bash
# Terminal 1: Launch Backend
python backend/server.py

# Terminal 2: Stream ESP32 sensor telemetry
python hardware_sim/esp32_sensor_telemetry.py subclinical
```
