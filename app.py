import streamlit as st
import numpy as np
import cv2
import tempfile
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

@st.cache_resource
def load_models():
    img_model = tf.keras.models.load_model("deepfake_model.keras")
    vid_model = tf.keras.models.load_model("deepfake_model_videos.keras")
    return img_model, vid_model

img_model, vid_model = load_models()

st.title("🧠 Deepfake Detection System")
st.write("Upload an **Image or Video** to check if it's Real or Fake")

option = st.radio("Choose Input Type", ["Image", "Video"])

# ------------------ IMAGE PREDICTION ------------------
def predict_image(img_file):
    img = image.load_img(img_file, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred = img_model.predict(img_array, verbose=0)[0][0]

    label = "Real" if pred > 0.5 else "Fake"
    return label


# ------------------ VIDEO FUNCTIONS ------------------
def extract_frames(video_path, img_size=224, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (img_size, img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        if len(frames) >= max_frames:
            break

    cap.release()
    return frames


def predict_video(video_file):
    import os

    # ✅ Save uploaded file properly
    temp_path = os.path.join(tempfile.gettempdir(), video_file.name)

    with open(temp_path, "wb") as f:
        f.write(video_file.read())

    # ✅ Now use SAME logic as your working code
    frames = extract_frames(temp_path)

    if len(frames) == 0:
        st.error("❌ No frames extracted")
        return "Error"

    preds = []
    for frame in frames:
        frame = preprocess_input(frame)
        frame = np.expand_dims(frame, axis=0)

        prob = vid_model.predict(frame, verbose=0)[0][0]
        preds.append(prob)

    avg_pred = np.mean(preds)

    st.write("Avg Prediction:", float(avg_pred))  # debug

    # ✅ SAME logic as your working script
    label = "Fake" if avg_pred > 0.5 else "Real"

    return label


# ------------------ IMAGE UI ------------------
if option == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width="stretch")

        label = predict_image(uploaded_file)

        st.subheader(f"Prediction: {label}")


# ------------------ VIDEO UI ------------------
if option == "Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        if st.button("Analyze Video"):
            with st.spinner("Processing video... ⏳"):
                label = predict_video(uploaded_file)

            st.subheader(f"Prediction: {label}")