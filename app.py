import streamlit as st
from streamlit.components.v1 import html
import cv2
import numpy as np
import os
import urllib.request
from PIL import Image
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Image Colorizer",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
    }

    h1 {
        text-align: center;
        color: white;
        font-size: 3rem !important;
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    .glass {
        background: rgba(255,255,255,0.08);
        padding: 20px;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }

    .stButton>button {
        background: linear-gradient(90deg,#7c3aed,#2563eb);
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        transform: scale(1.03);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("🎨 AI Black & White Image Colorizer")

st.markdown(
    "<p class='subtitle'>Transform old grayscale memories into vibrant AI-generated color images</p>",
    unsafe_allow_html=True
)

# Animation
html(
    '''
    <div style="text-align:center; margin-bottom:20px;">
        <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaW92aWQ2b3pjcTg5bW9wNnFhY3NnM2RrN3p0eXUyMzJ2MnBtb3FpYiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/l0HlBO7eyXzSZkJri/giphy.gif" width="250">
    </div>
    ''',
    height=260
)

# Session state for image history
if "history" not in st.session_state:
    st.session_state.history = []

# Model paths
proto_file = "model/colorization_deploy_v2.prototxt"
model_file = "model/colorization_release_v2.caffemodel"
pts_file = "model/pts_in_hull.npy"

# ---------------- AUTO DOWNLOAD MODEL ---------------- #

model_url = "https://github.com/richzhang/colorization/raw/caffe/models/colorization_release_v2.caffemodel"

if not os.path.exists(model_file):
    st.info("Downloading AI model... Please wait.")
    urllib.request.urlretrieve(model_url, model_file)
    st.success("Model downloaded successfully!")
# Load model
net = cv2.dnn.readNetFromCaffe(proto_file, model_file)
pts = np.load(pts_file)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")

pts = pts.transpose().reshape(2, 313, 1, 1)

net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Professional Sidebar
st.sidebar.title("🧠 About Project")

st.sidebar.info(
    '''
    AI-powered image colorization web app
    built using OpenCV, Deep Learning,
    and Streamlit.
    '''
)

st.sidebar.markdown("### 🚀 Features")
st.sidebar.write("✔ Multiple Image Upload")
st.sidebar.write("✔ AI Colorization")
st.sidebar.write("✔ Download Results")
st.sidebar.write("✔ Image History")

st.sidebar.markdown("### 🛠 Tech Stack")
st.sidebar.write("Python")
st.sidebar.write("OpenCV")
st.sidebar.write("Streamlit")
st.sidebar.write("Deep Learning")

# Multiple image upload
uploaded_files = st.file_uploader(
    "Upload Black & White Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Function to colorize image

def colorize_image(image):

    # Normalize image
    scaled = image.astype("float32") / 255.0

    # Convert to LAB
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    # Resize
    resized = cv2.resize(lab, (224, 224))

    # L channel
    L = cv2.split(resized)[0]

    # Mean subtraction
    L -= 50

    # Predict ab channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize predicted channels
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # Original L channel
    L_original = cv2.split(lab)[0]

    # Merge channels
    colorized = np.concatenate(
        (L_original[:, :, np.newaxis], ab),
        axis=2
    )

    # Convert LAB to RGB
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)

    # Clip values
    colorized = np.clip(colorized, 0, 1)

    # Convert to uint8
    colorized = (255 * colorized).astype("uint8")

    return colorized

# Process uploaded files
if uploaded_files:

    for uploaded_file in uploaded_files:

        st.divider()

        st.subheader(f"📸 {uploaded_file.name}")

        # Read image
        image = Image.open(uploaded_file).convert("RGB")
        image = np.array(image)

        # Colorize image
        with st.spinner("Colorizing image using AI..."):
            colorized = colorize_image(image)

        # Create columns
        col1, col2 = st.columns(2)

        # Original image
        with col1:
            st.markdown("### Original Image")
            st.image(image, width="stretch")

        # Colorized image
        with col2:
            st.markdown("### Colorized Image")
            st.image(colorized, width="stretch")

        # Save history
        timestamp = datetime.now().strftime("%H:%M:%S")

        st.session_state.history.append({
            "name": uploaded_file.name,
            "time": timestamp
        })

        # Download button
        st.download_button(
            label=f"⬇ Download {uploaded_file.name}",
            data=cv2.imencode(
                '.png',
                cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)
            )[1].tobytes(),
            file_name=f"colorized_{uploaded_file.name}",
            mime="image/png"
        )

# Image history section
st.divider()
st.header("🕘 Image History")

if st.session_state.history:

    for item in reversed(st.session_state.history):
        st.write(f"📁 {item['name']}  |  ⏰ {item['time']}")

else:
    st.write("No images processed yet.")