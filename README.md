# 💸 Real-Time PKR Currency Detection System

## 📌 Project Overview
This project is a deep learning-based image classification system that detects and identifies Pakistani currency notes in real time using a trained convolutional neural network (MobileNetV2).

The system captures an image using a camera and predicts the denomination of the currency note along with a confidence score.

---

## 🎯 Objectives
- Detect Pakistani currency denominations
- Use transfer learning for efficient training
- Deploy model in a real-time web interface
- Provide prediction confidence and history logging

---

## 🛠 Technologies Used
- Python
- PyTorch
- Torchvision
- Streamlit
- Scikit-learn
- Pandas
- Pillow

---

## 🧠 Model Details
- Architecture: MobileNetV2 (Transfer Learning)
- Input Size: 224x224
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## 📂 Project Structure
data/            # Dataset (train, val, test)
models/          # Saved trained model
train.py         # Model training script
app.py           # Streamlit deployment script
evaluation.py    # Model evaluation
split_data.py    # Dataset splitting
requirements.txt # Project dependencies
README.md        # Project documentation


---

## 🚀 How To Run The Project

1. Install dependencies:
   pip install -r requirements.txt

2. Train the model:
   python train.py

3. Run the application:
   streamlit run app.py

---

## 📊 Features
- Real-time camera input
- Confidence-based prediction
- Rejection of low-confidence predictions
- Capture history
- CSV export functionality

---

## ⚠ Limitations
- Performance depends on lighting conditions
- May struggle with damaged or folded notes
- Requires more dataset for improved accuracy

---

## 🔮 Future Improvements
- Add counterfeit detection
- Convert to mobile application
- Use object detection (YOLO)
- Increase dataset size
