# Age & Gender Detection

A real-time **Age & Gender Detection** application built with **Python, OpenCV, Streamlit, and Streamlit-WebRTC**. The application detects faces from uploaded images or a live webcam stream and predicts the person's **gender** and **age range**.

## 🚀 Features

- 📷 Upload an image and detect faces
- 🎥 Real-time webcam face detection
- 👤 Multiple-face detection in a single frame
- 🧑 Gender prediction
- 🎂 Age-range prediction
- 🖼️ Bounding boxes and prediction labels on detected faces
- ⚡ Streamlit-based interactive web interface
- 🔒 Images are processed locally by the application

## 🧠 How It Works

The application uses OpenCV's DNN module with pretrained deep-learning models:

1. A face detector locates faces in the input image/frame.
2. Each detected face is cropped from the frame.
3. The cropped face is passed to the gender and age networks.
4. The predicted gender and age range are displayed above the face.

```text
Input Image / Webcam
        ↓
   Face Detection
        ↓
   Face Cropping
        ↓
 ┌───────────────┐
 │ Gender Model  │
 │   Age Model   │
 └───────────────┘
        ↓
 Prediction + Bounding Box
```

## 🎂 Age Categories

The current age model predicts one of these age ranges:

```text
(0-2)
(4-6)
(8-12)
(15-20)
(25-32)
(38-43)
(48-53)
(60-100)
```

> **Note:** The application predicts an age range, not an exact age. Predictions can vary depending on image quality, lighting, pose, facial expression, and other factors.

## 🛠️ Technologies Used

- **Python 3.11**
- **OpenCV** – image processing, face detection, and DNN inference
- **Streamlit** – web application interface
- **Streamlit-WebRTC** – real-time webcam streaming
- **NumPy** – numerical operations
- **Pillow** – image handling
- **TensorFlow/OpenCV DNN models** – deep-learning inference

## 📁 Project Structure

```text
Age_Gender_code/
│
├── app.py
├── README.md
├── requirements.txt
│
└── models/
    ├── age_net.caffemodel
    ├── age_deploy.prototxt
    ├── gender_net.caffemodel
    ├── gender_deploy.prototxt
    ├── opencv_face_detector_uint8.pb
    └── opencv_face_detector.pbtxt
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Apex-afk/Age-gender-detection.git
cd Age-gender-detection
```

### 2. Create and activate a Python environment

Using Conda:

```bash
conda create -n agegender python=3.11
conda activate agegender
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

## 📦 Model Files

The project uses pretrained OpenCV DNN model files stored in the `models/` directory. The large model files are managed using **Git LFS**.

Make sure Git LFS is installed before cloning/downloading the repository if you need the model binaries:

```bash
git lfs install
git lfs pull
```

## 📋 Requirements

The main Python dependencies are:

```text
streamlit
opencv-python
numpy
Pillow
streamlit-webrtc
av
```

## 🎥 Usage

### Image Upload

1. Start the Streamlit application.
2. Select the image-upload option.
3. Upload a JPG/PNG image.
4. The application detects faces and displays age and gender predictions.

### Webcam

1. Allow browser camera access when prompted.
2. Start the webcam section.
3. The application processes the live video stream and displays predictions on detected faces.

## ⚠️ Limitations

- Age prediction is an estimate and should not be treated as an exact age.
- Accuracy may decrease with poor lighting, low-resolution images, extreme poses, occlusion, or unusual camera angles.
- Age and gender predictions from facial images can contain demographic and dataset-related biases.
- The model should not be used for high-stakes decisions or identity verification.

## 🔮 Future Improvements

- Improve age prediction accuracy using a model trained specifically on the UTKFace dataset.
- Improve gender prediction accuracy with a trained ensemble model.
- Add confidence scores to predictions.
- Improve performance for multiple faces and low-light conditions.
- Add a cleaner and more customizable user interface.
- Deploy the application online for easier access.

## 📌 Disclaimer

This project is intended for **educational and demonstration purposes**. Age and gender predictions are model estimates and may be inaccurate. Do not use the predictions as definitive information about a person.

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.
