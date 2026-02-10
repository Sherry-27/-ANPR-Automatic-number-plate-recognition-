# 🚗 Automatic Number Plate Recognition (ANPR)

<div align="center">
![ANPR Demo](https://github.com/Sherry-27/-ANPR-Automatic-number-plate-recognition-/blob/main/demo.png)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv9](https://img.shields.io/badge/YOLOv9-GELAN--C-green.svg)](https://github.com/WongKinYiu/yolov9)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Real-time license plate detection and recognition system powered by YOLOv9 and OCR**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Architecture](#-architecture)
</div>

---

## 🎯 Features
- ⚡ **Real-time Detection**: Achieve 95% mAP@0.5 with YOLOv9 GELAN-C architecture
- 🔍 **OCR Integration**: Accurate alphanumeric extraction from detected plates
- 🌍 **Multi-format Support**: Handles various license plate formats and styles
- 🌤️ **Robust Performance**: Works reliably across different lighting conditions
- 🚀 **Production-Ready**: Optimized inference pipeline for deployment
- 🎯 **GPU Accelerated**: Trained on Tesla T4 GPU for optimal performance
- ⚡ **Inference Optimization**: Exported to ONNX + tested with TensorRT

## 🎬 Demo

### Detection Results
<div align="center">
| Input | Detection | OCR Result |
|:-----:|:---------:|:----------:|
| ![Input](screenshots/input1.jpg) | ![Detection](screenshots/detect1.jpg) | **R-183-JF** |
| Various plate formats | Bounding box + confidence | Extracted text |
</div>

### Sample Output
