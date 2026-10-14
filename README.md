# 🩺 ElderlyCare AI — Medicine Verification & Fall Safety System

A complete, production-grade elderly-care application integrating **AI-based Medicine Identification and Prescription Verification** with **Real-Time Vision Fall Detection Monitoring** into a unified, modular Streamlit platform.

---

## 📌 1. Project Overview & Architectural Objectives

ElderlyCare AI is engineered to address two major safety concerns in geriatric healthcare:
1. **Dispensing & Medication Administration Safety:** Verification of prescribed medications against photographed packaging (strips, bottles, blisters) to prevent adverse drug events caused by dose confusion, look-alike packaging, or illegible prescriptions.
2. **Kinematic Fall Monitoring:** Continuous vision-based human pose estimation and multi-factor biomechanical fall detection to alert caregivers in the event of an accidental fall or sudden collapse.

Both systems are unified under an executive dashboard and shared alert management centre with persistent SQLite event logging, alert deduplication, incident acknowledgment, and comprehensive audit reporting.

---

## 🗂️ 2. Repository Directory Structure

```text
elderly_care_app/
│
├── app.py                          # Main Streamlit unified application entrypoint
│
├── pages/
│   ├── __init__.py                 # Page package marker
│   ├── dashboard.py                # Executive Care Dashboard (KPI metrics, health status)
│   ├── medicine_verification.py    # Prescription verification, packaging OCR, catalogue explorer
│   ├── fall_monitoring.py          # Real-time webcam, video stream, and image fall diagnostics
│   ├── alerts_history.py           # Unified Care Alert Centre (incident logs, ack/resolve actions)
│   └── system_information.py       # Hardware runtime, AI architecture, FPGA edge roadmap
│
├── modules/
│   ├── __init__.py                 # Modules package marker
│   ├── medicine_adapter.py         # Adapter wrapping existing medicine verification engine
│   ├── fall_detector.py            # Geometric & temporal fall detector from Untitled4.ipynb
│   ├── fall_inference.py           # YOLO11 pose inference & 14-point split-view skeleton visualizer
│   ├── camera_manager.py           # DirectShow webcam capture & video streaming engine
│   ├── alert_manager.py            # Central coordinator for alert deduplication & cooldown
│   └── event_store.py              # Persistent SQLite store (cache/elderly_care_events.db)
│
├── src/                            # Preserved core medicine verification source modules
│   ├── __init__.py
│   ├── config.py                   # Paths, thresholds, status constants, hardware device probe
│   ├── image_preprocessing.py      # Laplacian blur variance, CLAHE contrast enhancement
│   ├── cnn_extractor.py            # MobileNetV2 pretrained CNN visual feature extraction
│   ├── ocr_engine.py               # EasyOCR CRAFT text detector & recognition network
│   ├── catalogue_index.py          # SQLite FTS5 full-text search across 253,973 medicines
│   ├── prescription_parser.py      # PyMuPDF digital & scanned prescription parser
│   ├── verification_engine.py      # Rule-based decision engine & explanation generator
│   └── report_generator.py         # CSV & ReportLab PDF clinical audit generation
│
├── tests/                          # Automated unit, regression, and integration test suite
│   ├── __init__.py
│   ├── test_medicine_integration.py # Prescription parsing, OCR verification, orphan detection
│   ├── test_fall_detector.py       # Keypoint rules, temporal drop, camera lifecycle, pose inference
│   └── test_alerts.py              # SQLite storage, alert deduplication, cooldown, independence
│
├── cache/
│   ├── medicine_catalogue_fts.db   # 253,973-record SQLite FTS5 database
│   ├── reference_embeddings.npz    # 1,280-dim MobileNetV2 embeddings for packaging photos
│   └── elderly_care_events.db      # Persistent alert events database
│
├── reports/
│   ├── snapshots/                  # Saved JPEG incident snapshot evidence for fall alerts
│   ├── verification_report.csv     # Exported CSV verification audit report
│   └── verification_report.pdf     # Exported PDF clinical verification report
│
├── Untitled4.ipynb                 # Source notebook for YOLO11 fall detection system
├── evaluation.py                   # Medicine system benchmark & confusion matrix evaluation
├── fpga_compatibility.md           # Edge FPGA deployment roadmap (AMD Xilinx Kria / DPU)
├── requirements.txt                # Consolidated dependencies
├── test_env.py                     # Environment dependency verification utility
└── README.md                       # Complete documentation
```

---

## 🔬 3. Preserved Detection Systems & Decision Rules

### A. Medicine Verification Module (from `app(3).py` & `src/`)
- **Prescription Parsing:** Extracts medicine names and strengths from PDF prescriptions (`src/prescription_parser.py`).
- **Packaging OCR:** Reads brand names, numeric strengths, and active formulations using EasyOCR (`src/ocr_engine.py`).
- **Visual Similarity:** Computes 1,280-dimensional MobileNetV2 embeddings and cosine similarity against cached reference packaging (`src/cnn_extractor.py`).
- **Indian Medicine Catalogue:** Sub-millisecond FTS5 search across 253,973 pharmaceutical items (`cache/medicine_catalogue_fts.db`).
- **Decision Engine:** Evaluates brand name agreement, numeric strength match/conflict, and image blurriness to assign standardized statuses:
  - `MATCH`: Medicine identity and strength agree with high OCR confidence.
  - `MISMATCH`: Explicit conflict detected (e.g., prescribed 500mg, but packaging shows 650mg).
  - `UNREADABLE`: Low Laplacian sharpness or illegible text; requests a clearer photograph.
  - `NOT_FOUND`: Prescribed item has no matching packaging photograph uploaded.
  - `REVIEW_REQUIRED`: Incomplete prescription strength or ambiguous text requiring clinical review.
- **Orphan Packaging Detection:** Flags uploaded packages not present on the prescription.
- **Report Generation:** Generates downloadable CSV and styled PDF clinical audit reports (`src/report_generator.py`).

### B. Fall Detection Module (from `Untitled4.ipynb`)
- **YOLO11 Pose Estimation:** Leverages `yolo11n-pose.pt` to extract 17 COCO body keypoints.
- **Kinematic Wireframe Topology:** 14-point skeleton mapping with bone linkage on dual-canvas split view (`modules/fall_inference.py`).
- **Multi-Feature Decision Rules (`FallDetector`):**
  1. *Center-of-Mass Downward Velocity:* Computes temporal drop velocity across a 15-frame history queue. A velocity $> 280$ px/sec indicates a rapid vertical drop.
  2. *Torso Inclination Angle:* Angle derived from shoulder-hip delta. $\theta < 50^\circ$ indicates a horizontal or inclined torso.
  3. *Aspect Ratio Rules:* Bounding box aspect ratio $> 0.95$ or upper-body aspect ratio (shoulder width / torso height) $> 1.2$ indicates a prone or slumped posture.
  4. *Head-Drop Condition:* Nose position level with or below shoulders ($y_{\text{nose}} \ge y_{\text{sh}} - 15$).
  5. *Combination Logic:* Event triggered if:
     - $(\text{Torso} < 50^\circ \text{ and Aspect} > 0.95)$ OR
     - $(\text{Rapid Drop} \text{ and Torso} < 50^\circ)$ OR
     - $(\text{Head Drop} \text{ and Aspect} > 0.85)$.

---

## ⚡ 4. Hardware Support & FPGA Acceleration Roadmap

- **Automatic Device Probe:** Automatically detects CUDA GPU vs. Host CPU fallback (`src/config.py`).
- **OpenVINO Inference Support:** Supports OpenVINO engine loading if exported models are placed in `yolo11n-pose_openvino_model/`.
- **FPGA Edge Acceleration:** Full deployment blueprint provided in `fpga_compatibility.md` for AMD Xilinx Kria KV260 / KR260 edge boards via ONNX export, INT8 quantization, and DPU compilation.

---

## 🚀 5. Installation & Execution Guide

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows / Linux / macOS

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Environment Dependencies
```bash
python test_env.py
```
Expected output:
```text
Verifying core dependencies for ElderlyCare AI...
  [OK] torch           : 2.13.0+cpu
  [OK] torchvision     : 0.28.0+cpu
  [OK] ultralytics     : 8.4.138
  [OK] cv2             : 5.0.0
  [OK] numpy           : 2.5.1
  [OK] pandas          : 3.0.5
  [OK] streamlit       : 1.60.0
  [OK] easyocr         : 1.7.2
  [OK] pymupdf         : 1.28.2
  [OK] PIL             : 12.3.0
  [OK] reportlab       : 5.0.1

✅ All core dependencies for Medicine Verification and Fall Monitoring are installed!
```

### 4. Launch the Unified Application
```bash
streamlit run app.py
```
Open your browser at: **`http://localhost:8501`**

---

## 🧪 6. Testing & Evaluation Suite

### Run All Integration & Regression Tests
```bash
python -m unittest discover tests
```
The test suite validates:
- SQLite persistent alert recording, cooldown re-arming, deduplication, and module independence (`tests/test_alerts.py`).
- FallDetector geometric rules, rapid velocity drop, camera lifecycle, and YOLO11 pose inference (`tests/test_fall_detector.py`).
- FTS5 catalogue search, prescription parsing, MATCH/MISMATCH/UNREADABLE verification, orphan detection, and CSV/PDF report generation (`tests/test_medicine_integration.py`).

### Run Medicine System Benchmarks & Confusion Matrix
```bash
python evaluation.py
```

---

## 📋 7. Step-by-Step Demonstration Checklist

### Demo A: Care Dashboard
1. Open the application at `http://localhost:8501`.
2. Inspect the **Executive Care Dashboard**: confirm hardware device status (CPU/GPU), active catalogue records (253,973 items), and KPI metric cards.

### Demo B: Medicine Verification Workflow
1. Navigate to **💊 Medicine Verification**.
2. Click **📄 Load Synthetic Test Prescription** to load Paracip, Eciclo-P, and Augmentin.
3. Click **📦 Load Sample Package Photos** to load Paracip 500mg and Paracip 650mg packaging images.
4. Click **🚀 Verify Prescription Against Packages**.
5. Observe:
   - Item 1 (Paracip 500) -> `MATCH`
   - Item 2 (Paracip 650 vs 500 prescription) -> `MISMATCH` (Strength conflict detected)
   - Item 3 (Augmentin) -> `NOT_FOUND` (Missing packaging)
6. Click **📥 Download CSV Verification Report** and **📄 Download Official PDF Clinical Report**.
7. Explore the **Indian Medicine Catalogue Explorer** tab: search for `Augmentin` or `Paracip` to verify sub-millisecond SQLite FTS5 search.

### Demo C: Fall Safety Monitoring
1. Navigate to **🚨 Fall Safety Monitoring**.
2. Choose **🎞️ Video File Demonstration** (upload a sample video) or **🖼️ Single-Image Diagnostic Test** (upload a posture snapshot) or **📹 Live Webcam Monitoring**.
3. Observe real-time 14-point COCO skeleton rendering and posture status (`Standing / Seated` vs. `CRITICAL ALERT: POTENTIAL FALL DETECTED!`).
4. Notice that when a fall event is triggered, a snapshot evidence image is saved to `reports/snapshots/` and an incident is posted to the Care Alert Centre.

### Demo D: Unified Alert Centre
1. Navigate to **🔔 Care Alert Centre**.
2. Filter fall alerts by status (`NEW`, `ACKNOWLEDGED`, `RESOLVED`).
3. Click **👁️ Acknowledge Alert** and **✅ Mark as Resolved**; verify state persistence.
4. Switch to the **💊 Medicine Audit Log** tab to verify that medicine verification audits are preserved alongside fall incidents.

---

## 🛡️ 8. Clinical & Safety Disclaimers

> [!WARNING]
> **🚨 Fall Safety Prototype Disclaimer:**
> The fall monitoring module is an experimental vision-based assistive prototype developed for research and demonstration purposes. It is **not** a certified medical alert device or a life-safety system. It must never substitute for human caregivers, healthcare attendants, or emergency response services. Video occlusions, variable lighting, or clothing loose-fitting may produce false positives or false negatives.

> [!CAUTION]
> **⚠️ Medicine Verification Operational Disclaimer:**
> The medicine identification and prescription verification system provides an auxiliary computer-vision cross-check. Automated outputs must never be used to make independent clinical decisions or administer medications. All pharmaceutical preparations must undergo physical, manual inspection and confirmation by a licensed physician or registered clinical pharmacist prior to dispensing or patient consumption.
