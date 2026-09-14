import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="Age & Gender Detection",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 Age & Gender Detection")
st.write("Detect age range and gender using a photo or your webcam.")


# =========================
# MODEL PATHS
# =========================

MODEL_DIR = Path(__file__).parent / "models"

FACE_MODEL = MODEL_DIR / "opencv_face_detector_uint8.pb"
FACE_PROTO = MODEL_DIR / "opencv_face_detector.pbtxt"

AGE_MODEL = MODEL_DIR / "age_net.caffemodel"
AGE_PROTO = MODEL_DIR / "age_deploy.prototxt"

GENDER_MODEL = MODEL_DIR / "gender_net.caffemodel"
GENDER_PROTO = MODEL_DIR / "gender_deploy.prototxt"


# =========================
# CHECK MODEL FILES
# =========================

required_files = [
    FACE_MODEL,
    FACE_PROTO,
    AGE_MODEL,
    AGE_PROTO,
    GENDER_MODEL,
    GENDER_PROTO
]

missing_files = [
    str(file) for file in required_files
    if not file.exists()
]

if missing_files:

    st.error("❌ Model files are missing:")

    for file in missing_files:
        st.write(file)

    st.stop()


# =========================
# LOAD MODELS
# =========================

@st.cache_resource
def load_models():

    face_net = cv2.dnn.readNetFromTensorflow(
        str(FACE_MODEL),
        str(FACE_PROTO)
    )

    age_net = cv2.dnn.readNetFromCaffe(
        str(AGE_PROTO),
        str(AGE_MODEL)
    )

    gender_net = cv2.dnn.readNetFromCaffe(
        str(GENDER_PROTO),
        str(GENDER_MODEL)
    )

    return face_net, age_net, gender_net


face_net, age_net, gender_net = load_models()


# =========================
# SETTINGS
# =========================

AGE_LIST = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)"
]

GENDER_LIST = [
    "Male",
    "Female"
]

MODEL_MEAN_VALUES = (
    78.4263377603,
    87.7689143744,
    114.895847746
)


# =========================
# DETECTION FUNCTION
# =========================

def detect_faces(frame):

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300, 300),
        [104, 117, 123],
        swapRB=False
    )

    face_net.setInput(blob)

    detections = face_net.forward()

    faces = []

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:

            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)

            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2))

    return faces


# =========================
# PREDICTION FUNCTION
# =========================

def predict_face(face):

    blob = cv2.dnn.blobFromImage(
        face,
        1.0,
        (227, 227),
        MODEL_MEAN_VALUES,
        swapRB=False
    )

    # Gender
    gender_net.setInput(blob)
    gender_predictions = gender_net.forward()

    gender_index = gender_predictions[0].argmax()
    gender = GENDER_LIST[gender_index]

    gender_confidence = (
        float(gender_predictions[0][gender_index]) * 100
    )

    # Age
    age_net.setInput(blob)
    age_predictions = age_net.forward()

    age_index = age_predictions[0].argmax()
    age = AGE_LIST[age_index]

    age_confidence = (
        float(age_predictions[0][age_index]) * 100
    )

    return (
        gender,
        gender_confidence,
        age,
        age_confidence
    )


# =========================
# DRAW DETECTIONS
# =========================

def process_image(image):

    faces = detect_faces(image)

    results = []

    for (x1, y1, x2, y2) in faces:

        face = image[y1:y2, x1:x2]

        if face.size == 0:
            continue

        gender, gender_conf, age, age_conf = predict_face(face)

        label = f"{gender} | Age {age}"

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        results.append({
            "gender": gender,
            "gender_confidence": gender_conf,
            "age": age,
            "age_confidence": age_conf
        })

    return image, results


# =========================
# PHOTO UPLOAD
# =========================

st.header("📷 Option 1 — Upload a Photo")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_np = np.array(image)

    image_cv = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    result_image, results = process_image(image_cv)

    result_image = cv2.cvtColor(
        result_image,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        result_image,
        caption="Detection Result",
        use_container_width=True
    )

    if results:

        for i, result in enumerate(results):

            st.success(
                f"Face {i + 1}: "
                f"{result['gender']} | "
                f"Age {result['age']}"
            )

            st.write(
                f"Gender confidence: "
                f"{result['gender_confidence']:.1f}%"
            )

            st.write(
                f"Age confidence: "
                f"{result['age_confidence']:.1f}%"
            )

    else:

        st.warning("No face detected.")


# =========================
# WEBCAM
# =========================

st.markdown("---")

st.header("🎥 Option 2 — Use Your Webcam")

st.write(
    "Click START and allow your browser to access the camera."
)


class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        result_image, _ = process_image(img)

        return frame.from_ndarray(
            result_image,
            format="bgr24"
        )


RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)


webrtc_streamer(
    key="age-gender-webcam",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)


# =========================
# DISCLAIMER
# =========================

st.markdown("---")

st.warning(
    "⚠️ These are model estimates. "
    "Age is predicted as a range, not an exact age."
)