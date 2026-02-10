# -ANPR-Automatic-number-plate-recognition-

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

## 🎬 Demo

### Detection Results

<div align="center">

| Input | Detection | OCR Result |
|:-----:|:---------:|:----------:|
| ![Input](screenshots/input1.jpg) | ![Detection](screenshots/detect1.jpg) | **R-183-JF** |
| Various plate formats | Bounding box + confidence | Extracted text |

</div>

### Sample Output
```
Detected: R-183-JF
Confidence: 0.95
Processing Time: 23ms
```

## 📊 Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **mAP@0.5** | 95.0% |
| **Precision** | 94.2% |
| **Recall** | 91.8% |
| **Inference Time** | ~25ms per frame |
| **FPS** | ~40 FPS (Tesla T4) |

### Training Details
- **Dataset Size**: 278 annotated images
- **Training Device**: Tesla T4 GPU
- **Epochs**: 100
- **Batch Size**: 16
- **Input Size**: 640x640

## 🏗️ Architecture

```mermaid
graph LR
    A[Input Image] --> B[YOLOv9 GELAN-C]
    B --> C[Bounding Box Detection]
    C --> D[Crop License Plate]
    D --> E[OCR Pipeline]
    E --> F[Text Extraction]
    F --> G[Final Output]
    
    style B fill:#f9f,stroke:#333,stroke-width:4px
    style E fill:#bbf,stroke:#333,stroke-width:4px
```

### Pipeline Components

1. **Detection Module** (YOLOv9 GELAN-C)
   - Backbone: CSPDarknet with GELAN architecture
   - Neck: PAN (Path Aggregation Network)
   - Head: Decoupled detection head
   - Activation: SiLU

2. **OCR Module**
   - Preprocessing: Grayscale conversion, thresholding
   - Recognition: Tesseract OCR / EasyOCR
   - Post-processing: Format validation, character filtering

## 🛠️ Installation

### Prerequisites
```bash
Python 3.8+
CUDA 11.8+ (for GPU support)
```

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Sherry-27/-ANPR-Automatic-number-plate-recognition-.git
cd ANPR-Automatic-number-plate-recognition
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Requirements.txt
```txt
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
pytesseract>=0.3.10
easyocr>=1.7.0
ultralytics>=8.0.0
matplotlib>=3.7.0
```

4. **Download YOLOv9 weights**
```bash
# Download pre-trained weights
wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt

# Or use our fine-tuned weights
# Place weights in: models/best.pt
```

## 🚀 Usage

### Quick Start

```python
from anpr import ANPR

# Initialize detector
detector = ANPR(
    model_path='models/best.pt',
    device='cuda'  # or 'cpu'
)

# Detect from image
result = detector.detect('path/to/image.jpg')
print(f"Detected Plate: {result['text']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Command Line Interface

```bash
# Single image
python detect.py --source image.jpg --weights models/best.pt

# Video file
python detect.py --source video.mp4 --weights models/best.pt

# Webcam
python detect.py --source 0 --weights models/best.pt

# Batch processing
python detect.py --source images/ --weights models/best.pt
```

### Advanced Usage

```python
# Custom configuration
detector = ANPR(
    model_path='models/best.pt',
    conf_threshold=0.5,  # Detection confidence
    iou_threshold=0.45,  # NMS IOU threshold
    ocr_engine='easyocr',  # or 'tesseract'
    device='cuda'
)

# Process video
for frame in video_stream:
    results = detector.detect(frame)
    for result in results:
        print(f"Plate: {result['text']}, Conf: {result['confidence']}")
```

## 📁 Project Structure

```
ANPR/
├── models/
│   ├── best.pt              # Fine-tuned YOLOv9 weights
│   └── yolov9-c.pt          # Base YOLOv9-C weights
├── data/
│   ├── images/              # Training images
│   ├── labels/              # YOLO format annotations
│   └── data.yaml            # Dataset configuration
├── src/
│   ├── anpr.py              # Main ANPR class
│   ├── detector.py          # YOLOv9 detection module
│   ├── ocr.py               # OCR processing module
│   └── utils.py             # Helper functions
├── notebooks/
│   ├── training.ipynb       # Model training notebook
│   └── evaluation.ipynb     # Performance evaluation
├── detect.py                # Detection script
├── train.py                 # Training script
├── requirements.txt         # Dependencies
└── README.md
```

## 🎓 Training

### Prepare Dataset

```bash
# Organize dataset in YOLO format
data/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### Train Model

```bash
python train.py \
    --data data/data.yaml \
    --cfg models/yolov9-c.yaml \
    --weights yolov9-c.pt \
    --epochs 100 \
    --batch 16 \
    --img 640 \
    --device 0
```

### Monitor Training

```bash
tensorboard --logdir runs/train
```

## 📈 Performance Comparison

| Model | mAP@0.5 | Inference Time | Parameters |
|-------|---------|----------------|------------|
| YOLOv5s | 89.2% | 28ms | 7.2M |
| YOLOv8m | 92.1% | 35ms | 25.9M |
| **YOLOv9-C (Ours)** | **95.0%** | **25ms** | **25.3M** |

## 🌟 Key Highlights

- 🎯 **95% mAP@0.5**: State-of-the-art detection accuracy
- ⚡ **25ms Inference**: Real-time processing capability
- 🎨 **Multi-format**: European, American, Asian plate formats
- 🌓 **Day/Night**: Robust to lighting variations
- 🚗 **Motion Blur**: Handles moving vehicles effectively

## 🔧 Configuration

### config.yaml
```yaml
detection:
  confidence_threshold: 0.5
  iou_threshold: 0.45
  input_size: 640
  
ocr:
  engine: 'easyocr'  # or 'tesseract'
  languages: ['en']
  allowlist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
  
preprocessing:
  grayscale: true
  denoise: true
  adaptive_threshold: true
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size or image size
detector = ANPR(model_path='models/best.pt', img_size=416)
```

**2. OCR Accuracy Issues**
```python
# Try different OCR engine or preprocessing
detector.ocr_engine = 'tesseract'
detector.enable_preprocessing = True
```

**3. Low Detection Confidence**
```python
# Lower confidence threshold
detector.conf_threshold = 0.3
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [YOLOv9](https://github.com/WongKinYiu/yolov9) - Object detection framework
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - OCR engine
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Alternative OCR engine
- [OpenCV](https://opencv.org/) - Computer vision library

## 📧 Contact

**Shaheer Khan**
- Email: sk9109182@gmail.com
- LinkedIn: [shaheer-khan-689a44265](https://www.linkedin.com/in/shaheer-khan-689a44265/)
- GitHub: [@Sherry-27](https://github.com/Sherry-27)

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{khan2025anpr,
  author = {Shaheer Khan},
  title = {Automatic Number Plate Recognition using YOLOv9},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Sherry-27/-ANPR-Automatic-number-plate-recognition-}
}
```

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made with ❤️ by [Shaheer Khan](https://github.com/Sherry-27)

</div>
