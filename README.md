# AI-Micro-Expression-Analyzer
AI Micro-Expression Analyzer is a real-time facial behavior monitoring system built using OpenCV and MediaPipe. It detects blink rate, eyebrow movement, lip tension, symmetry, and head motion to classify emotional states as CALM, STRESS, or HIGH STRESS. Ideal for interview analysis, presentations, and behavioral studies.
# 📁Project Structure
```
AI-MicroExpression-Analyzer/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── stress_log.csv
│
├── features/
│   ├── blink.py
│   ├── eyebrow.py
│   ├── lip_tension.py
│   ├── head_nod.py
│   └── symmetry.py
│
├── models/
│   └── state_model.py
│
├── utils/
│   ├── landmark_utils.py
│   └── drawing_utils.py
```




### How It Works (System Pipeline)

The system follows a structured real-time processing pipeline:

1️⃣ Video Capture

- Captures live frames using OpenCV webcam interface.

2️⃣ Face Detection

- Uses MediaPipe Face Mesh to detect facial landmarks (468 points).

3️⃣ Feature Extraction

- From the landmarks, the following behavioral signals are computed:

👁 Blink intensity
👄 Lip tension
😠 Eyebrow movement
🔄 Head micro-nod movement
⚖ Facial symmetry shifts

4️⃣ Weighted Scoring

- Each feature is assigned a weight based on its stress relevance.
  A normalized weighted stress score (0.0 – 1.0) is calculated.

5️⃣ Smoothing & Stability

- Exponential Moving Average (EMA) reduces noise.
  Multi-frame stability check prevents false alerts.ing Project Structure…]()

## Installation

**1. Clone the repository**
git clone https://github.com/your-username/AI-Micro-Expression-Analyzer.git
cd AI-Micro-Expression-Analyzer

**2. Install dependencies**
pip install -r requirements.txt
              or
pip install opencv-python mediapipe numpy

**3.Run the project**
python app.py

-----------------------
Press Q to exit.

## ⚠ Disclaimer

This is an academic prototype developed for learning and research purposes.
It is not a medical or psychological diagnostic system.
