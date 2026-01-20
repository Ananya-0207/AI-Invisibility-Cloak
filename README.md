# 🪄 AI Invisibility Cloak using Deep Learning (Real-Time)

A real-time **AI-powered invisibility cloak** system built using **DeepLabV3 semantic segmentation and computer vision techniques**.  
The project detects a cloth worn by a person and dynamically replaces it with the background captured from the webcam, creating a camouflage effect.

---
## 📌 Main Implementation File

👉 **`invisibility.py`**

This file contains the complete implementation of:
- Real-time webcam capture
- DeepLabV3 person segmentation
- Cloth detection & masking
- Background replacement (invisibility effect)
- Face protection
- Video recording and visualization
🔗 [View main implementation → invisibility_cloak.py](./invisibility_cloak.py)


## 🚀 Features

- Real-time webcam processing
- Deep learning–based **person segmentation (DeepLabV3)**
- Cloth detection using HSV color space
- Stable background modeling using median filtering
- Edge feathering for smooth blending
- Temporal smoothing to reduce flickering
- Face protection (face region never camouflaged)
- Output video recording
- Resume & interview ready pipeline

---

## 🧠 Pipeline Overview

1. Capture clean background using webcam
2. Use **DeepLabV3** to segment the person
3. Detect cloth region inside person mask (HSV color)
4. Protect face region
5. Apply morphological cleanup and edge feathering
6. Replace cloth pixels with background
7. Apply temporal smoothing for stability
8. Display and save output video

---

## 🛠 Tech Stack

- Python
- OpenCV
- PyTorch
- TorchVision
- NumPy

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/AI-Invisibility-Cloak.git
cd AI-Invisibility-Cloak
