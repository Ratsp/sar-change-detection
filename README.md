# 🛰️ **SAR Change Detection using Siamese CNN**

An **upcoming Deep Learning–based project** to perform **change detection on Synthetic Aperture Radar (SAR) images** using a **Siamese Convolutional Neural Network (CNN)** architecture.  
The system aims to **identify meaningful changes** between multi-temporal SAR image pairs for applications such as **urban growth monitoring, disaster assessment, and environmental analysis**.

> ⚠️ **Note:**  
> This repository is currently in the **planning and development phase**.  
> **Code, models, and experiments will be implemented progressively.**

---

## 🎯 **Project Objectives**
- Understand **SAR image characteristics** and speckle noise
- Implement **Siamese CNN architecture** for change detection
- Learn **deep learning workflows for remote sensing data**
- Apply **deep learning to real-world geospatial problems**
- Build an **end-to-end ML pipeline** for SAR change detection

---

## 🧠 **Planned Model Architecture**
- **Siamese Convolutional Neural Network**
  - Two identical CNN branches with **shared weights**
  - Input: **pre-change and post-change SAR image pairs**
  - Output: **change / no-change classification map**
- Possible backbone:
  - Custom CNN
  - Transfer Learning (if applicable)

---

## 📚 **Concepts That Will Be Covered**
- SAR imaging fundamentals
- Speckle noise and preprocessing techniques
- Siamese networks and metric learning
- Patch-based change detection
- Binary classification (Change vs No-Change)
- Model evaluation for imbalanced datasets

---

## 🛠️ **Tech Stack (Planned)**

### **Programming & Frameworks**
- **Python**
- **TensorFlow / Keras** or **PyTorch**

### **Image Processing**
- **OpenCV**
- **NumPy**
- **Rasterio / GDAL** (if required)

### **Visualization**
- **Matplotlib**
- **Seaborn**

### **Experiment Tracking (Optional)**
- **MLflow / TensorBoard**

---

## 📊 **Dataset (Planned)**
- Publicly available **SAR datasets**, such as:
  - **OSCD (Onera Satellite Change Detection)**
  - **SENTINEL-1 SAR imagery**
- Multi-temporal SAR image pairs
- Ground truth change maps (if available)

---

## ⚙️ **Planned Workflow**
1. **Dataset Collection**
2. **SAR Image Preprocessing**
3. **Patch Extraction**
4. **Siamese CNN Model Design**
5. **Training & Validation**
6. **Change Map Generation**
7. **Performance Evaluation**
8. **Visualization of Results**

---

## 🧪 **Planned Implementations**
- SAR image normalization and noise reduction
- Siamese CNN training pipeline
- Change map prediction
- Performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

