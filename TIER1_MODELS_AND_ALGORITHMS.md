# Tier-1 Models & Algorithms Reference Guide
**Project:** MastiFore (AI-Based Multimodal Early Forecasting of Bovine Mastitis)  
**Module:** Tier-1 Computer Vision & Behavioral Triage Engine  

---

## 📌 Executive Summary
Tier-1 provides **24/7 continuous non-invasive herd monitoring** using edge computer vision and a **Lightweight 6-Feature XGBoost / Gradient-Boosted Decision Ensemble**. It supports **Dual Execution Paths**:
- **Path 1 (Normal Stream):** Deep Learning with **YOLOv8 + ByteTrack** ($30+\text{ FPS}$ on GPU / Edge AI accelerator).
- **Path 2 (Fallback Stream):** Classical Image Processing with **Gaussian + Sobel/Prewitt + Canny + Aspect Ratio Filter** (CPU-only / Micro-gateways).

---

# 🚀 Path 1: Normal Stream (Deep Learning & ML Pipeline)

| Domain | Model / Algorithm Name | Exact Mathematical Formula / Technique | Primary Function in Path 1 |
| :--- | :--- | :--- | :--- |
| **Deep Learning (Vision)** | **YOLOv8 Deep CNN** (Ultralytics) | Multi-Scale Anchor-Free Bounding Box Regression | Real-time cattle localization (COCO Class 19: `cow`) with bounding boxes in $< 17\text{ ms}$. |
| **Tracking Algorithm** | **ByteTrack + Kalman Filter** | Bipartite Graph Matching + IoU Association Matrix | Retains persistent cattle tracking IDs (`COW_001`, `COW_002`) across occlusions. |
| **Motion Vision** | **Differential Lucas-Kanade Optical Flow** | $\Delta \vec{V} = \vec{V}_{\text{mandible}} - \vec{V}_{\text{body\_centroid}}$<br>$\text{CPM} = \left(\frac{\text{Chews}}{\Delta t}\right) \times 60$ | Cancels camera/body shake to measure **Chews Per Minute (CPM)** and chewing vigor ($\Delta V$). |
| **Sequential AI** | **Inter-Bolus Temporal State Machine** | Deglutition pause threshold: $\Delta t_{\text{pause}} > 3.5\text{s}$ | Tracks **Chews Per Bolus (Cud Count)** and swallow intervals. |
| **Geometric Vision** | **3-Point Anatomical Vector Geometry** | Withers ($P_1$), Mid-Spine ($P_2$), Sacrum ($P_3$):<br>$$\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}\right) \times \frac{180^\circ}{\pi}$$ | Measures **Sprecher Spinal Kyphosis Pain Angle** ($174^\circ$ flat vs. $< 155^\circ$ arched). |
| **Morphological Vision**| **Leg Pillar Vertical Column Density** | Lower 35% ROI projection: $\text{Density}(x) = \frac{1}{255}\sum_{y}\text{Edge}(x, y)$ | Detects $\ge 4$ vertical leg columns for **Standing** vs. **Resting** classification. |
| **Machine Learning (Triage)**| **Lightweight 6-Feature XGBoost / Gradient-Boosted Decision Ensemble** | $\text{Triage Logit} = (3.85\cdot\Delta\text{CPM}) + (3.20\cdot\Delta\text{Spine}) + (2.15\cdot\Delta\text{Cud}) + (1.80\cdot\Delta\text{Pause}) + (1.45\cdot\Delta\text{Amp}) + (1.20\cdot\Delta\text{Rest}) - 1.95$<br>$$P_{\text{Tier1}}(\text{Shortlist}) = \frac{1}{1 + e^{-\text{Triage Logit}}}$$ | **Shortlists at-risk cattle ($\ge 35\%$)** into the Milking Queue for targeted 10s ESP32 probe testing. |

---

# 🛡️ Path 2: Fallback Stream (Edge / CPU Low-Compute Pipeline)

| Domain | Model / Algorithm Name | Exact Mathematical Formula / Technique | Primary Function in Path 2 |
| :--- | :--- | :--- | :--- |
| **Spatial Filtering** | **$5 \times 5$ Gaussian Smoothing Kernel** | $K = \frac{1}{256} \begin{bmatrix} 1 & 4 & 6 & 4 & 1 \\ 4 & 16 & 24 & 16 & 4 \\ 6 & 24 & 36 & 24 & 6 \\ 4 & 16 & 24 & 16 & 4 \\ 1 & 4 & 6 & 4 & 1 \end{bmatrix}$ | Pre-filters camera thermal noise, dust, flies, and straw textures. |
| **Gradient Operators** | **Sobel & Prewitt 1st-Order Derivatives** | $\mathbf{S_x} = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \mathbf{S_y} = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$ | Computes directional image derivatives $\nabla I = [G_x, G_y]^T$ to extract body boundaries. |
| **Edge Detection** | **Canny Edge Detection** | Dual-Threshold Hysteresis ($T_1=25, T_2=85$) + Non-Maximum Suppression (NMS) | Generates clean single-pixel silhouette contours of all objects in the barn. |
| **Geometric Filtering**| **Bovine Aspect Ratio Filter ($\frac{W}{H}$)** | Condition: $1.05 \le \frac{W}{H} \le 3.2$ and $\text{Height} \ge 20\%$ | **Rejects humans** ($\text{Aspect} < 1.05$) and small stray dogs/birds without GPU. |
| **Motion Vision** | **Differential Lucas-Kanade Optical Flow** | $\Delta \vec{V} = \vec{V}_{\text{mandible}} - \vec{V}_{\text{body\_centroid}}$ | Measures **Chews Per Minute (CPM)** and jaw motion amplitude. |
| **Sequential AI** | **Inter-Bolus Temporal State Machine** | Pause threshold: $\Delta t_{\text{pause}} > 3.5\text{s}$ | Tracks chewing bolus cycles and deglutition swallow intervals. |
| **Geometric Vision** | **3-Point Anatomical Vector Geometry** | $\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}\right) \times \frac{180^\circ}{\pi}$ | Measures dorsal kyphosis pain angle (excluding drooping neck/head). |
| **Morphological Vision**| **Leg Pillar Vertical Column Density** | Column edge projection threshold $> 35\%$ | Classifies **Standing (4 legs)** vs. **Resting (folded body)**. |
| **Machine Learning (Triage)**| **Lightweight 6-Feature XGBoost / Gradient-Boosted Decision Ensemble** | $P_{\text{Tier1}}(\text{Shortlist}) = \frac{1}{1 + e^{-\text{Triage Logit}}}$ | **Shortlists suspicious cattle ($\ge 35\%$)** into the Milking Queue for ESP32 sensor testing. |
