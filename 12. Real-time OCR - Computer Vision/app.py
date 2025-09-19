import cv2
import pytesseract
import streamlit as st
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

class OCRProcessor(VideoProcessorBase):
    def __init__(self):
        # Quick knobs we’ll tweak from the UI
        self.conf_threshold = 25
        self.box_color_bgr = (0, 255, 0)
        self.text_color_bgr = (0, 255, 0)
        self.box_thickness = 1
        self.text_thickness = 1
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Step 1: simplify colors → grayscale helps OCR focus on shapes
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Step 2: separate text/background using Otsu threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        try:
            # Step 3: run OCR and get word boxes + confidences
            data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
            
            # Step 4: draw only what we trust
            for i in range(len(data['text'])):
                # Keep boxes with decent confidence and non-empty text
                if int(data['conf'][i]) > self.conf_threshold and data['text'][i].strip():
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    # Box shows where Tesseract thinks the word is
                    cv2.rectangle(img, (x, y), (x + w, y + h), self.box_color_bgr, self.box_thickness)
                    
                    # Label it so we see what was read
                    cv2.putText(img, data['text'][i], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color_bgr, self.text_thickness)
            
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def _hex_to_bgr(hex_color):
    # Streamlit gives hex; OpenCV expects BGR tuples
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)
    return (0, 255, 0)

def run_ocr_on_bgr(image_bgr, conf_threshold=25, box_color_bgr=(0, 255, 0), text_color_bgr=(0, 255, 0), box_thickness=1, text_thickness=1):
    # Same pipeline for still images
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    annotated = image_bgr.copy()
    texts = []
    try:
        data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
        for i in range(len(data['text'])):
            text = data['text'][i]
            conf_str = data['conf'][i]
            try:
                conf = int(conf_str)
            except Exception:
                conf = -1
            # Quick sanity checks: enough confidence and not empty
            if conf > conf_threshold and text.strip():
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color_bgr, box_thickness)
                cv2.putText(annotated, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color_bgr, text_thickness)
                texts.append(text)
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
    return annotated, texts

def read_settings_from_state():
    # Pull current UI settings so everything stays in sync
    conf_threshold = st.session_state["conf_threshold"]
    box_color_bgr = _hex_to_bgr(st.session_state["box_color"])
    text_color_bgr = _hex_to_bgr(st.session_state["text_color"])
    box_thickness = st.session_state["box_thickness"]
    text_thickness = st.session_state["text_thickness"]
    return conf_threshold, box_color_bgr, text_color_bgr, box_thickness, text_thickness

def decode_image_from_uploader(file_like):
    # Turn the uploaded file into an OpenCV image
    file_bytes = np.frombuffer(file_like.getvalue(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def display_annotated_and_text(annotated_bgr, texts, caption):
    # Show what OCR saw and list the tokens found
    st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), caption=caption)
    if texts:
        st.subheader("Detected Text:")
        for t in texts:
            st.write(f"• {t}")
    else:
        st.info("No text detected.")

st.title("Real-time OCR - Computer Vision")
st.write("Pick a mode below and let’s read some text together.")

# Sidebar controls
st.sidebar.header("OCR Settings")

# Defaults
_DEFAULTS = {
    "conf_threshold": 25,
    "box_color": "#00FF00",
    "text_color": "#00FF00",
    "box_thickness": 1,
    "text_thickness": 1,
}

# Initialize session state
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Reset button
if st.sidebar.button("Reset to defaults"):
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

# Widgets bound to session state keys
st.sidebar.slider("Confidence threshold", min_value=0, max_value=100, step=1, key="conf_threshold")
st.sidebar.color_picker("Bounding box color", key="box_color")
st.sidebar.color_picker("Text color", key="text_color")
st.sidebar.slider("Box thickness", min_value=1, max_value=5, key="box_thickness")
st.sidebar.slider("Text thickness", min_value=1, max_value=3, key="text_thickness")

# Read current settings
conf_threshold, box_color_bgr, text_color_bgr, box_thickness, text_thickness = read_settings_from_state()

mode = st.radio("Mode", ("Real-time", "Take Photo", "Upload Photo"), horizontal=True)

if mode == "Real-time":
    st.write("We’ll use your webcam and draw boxes live.")
    ctx = webrtc_streamer(
        key="ocr-camera",
        video_processor_factory=OCRProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
    if ctx and ctx.video_processor:
        # Pass current UI choices into the processor
        ctx.video_processor.conf_threshold = conf_threshold
        ctx.video_processor.box_color_bgr = box_color_bgr
        ctx.video_processor.text_color_bgr = text_color_bgr
        ctx.video_processor.box_thickness = box_thickness
        ctx.video_processor.text_thickness = text_thickness
elif mode == "Take Photo":
    img_file = st.camera_input("Take a photo")
    if img_file is not None:
        image_bgr = decode_image_from_uploader(img_file)
        if image_bgr is not None:
            # One-click OCR on your snapshot
            annotated, texts = run_ocr_on_bgr(image_bgr, conf_threshold, box_color_bgr, text_color_bgr, box_thickness, text_thickness)
            display_annotated_and_text(annotated, texts, caption="Annotated photo")
        else:
            st.error("Could not decode image.")
elif mode == "Upload Photo":
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        image_bgr = decode_image_from_uploader(uploaded)
        if image_bgr is not None:
            # Run the same pipeline for your file
            annotated, texts = run_ocr_on_bgr(image_bgr, conf_threshold, box_color_bgr, text_color_bgr, box_thickness, text_thickness)
            display_annotated_and_text(annotated, texts, caption="Annotated upload")
        else:
            st.error("Could not decode image.")
