import io
import os
from datetime import datetime

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# Torch + GradCAM
import torch
import torch.nn.functional as F
from torchvision import models, transforms


# ===============================
# Page config
# ===============================
st.set_page_config(page_title="Medivora Eye Disease Detection", layout="centered")

# ===============================
# Model / labels
# ===============================
VGG8_CLASSES = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']
ABS_MODEL_PATH = r"C:\Users\Antino\OneDrive\Desktop\Medivora_eye\oct_model\model\ensemble_model.onnx"

@st.cache_resource(show_spinner=True)
def load_session():
    try:
        sess = ort.InferenceSession(ABS_MODEL_PATH, providers=["CPUExecutionProvider"])
    except Exception:
        rel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "ensemble_model.onnx"))
        sess = ort.InferenceSession(rel_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    return sess, input_name

session, INPUT_NAME = load_session()

# ===============================
# Preprocessing
# ===============================
def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def to_probabilities(raw: np.ndarray) -> np.ndarray:
    vec = np.squeeze(raw).astype("float64")
    if (vec < 0).any() or vec.max() > 1.0 or vec.sum() <= 0.0:
        exps = np.exp(vec - np.max(vec))
        vec = exps / np.sum(exps)
    return vec

# ===============================
# Torch Grad-CAM Setup
# ===============================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class):
        output = self.model(input_tensor)
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam / cam.max()
        return cam.cpu().numpy()

# Torch model for GradCAM (EfficientNetB0)
torch_model = models.efficientnet_b0(pretrained=True)
torch_model.eval()
target_layer = torch_model.features[-1]
gradcam = GradCAM(torch_model, target_layer)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def generate_gradcam_with_circle(img, class_idx):
    tensor = transform(img).unsqueeze(0)
    heatmap = gradcam.generate(tensor, class_idx)

    # Find max activation point
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    scale_x = img.size[0] / heatmap.shape[1]
    scale_y = img.size[1] / heatmap.shape[0]
    cx, cy = int(x * scale_x), int(y * scale_y)

    circled_img = img.copy()
    draw = ImageDraw.Draw(circled_img)
    r = 30
    draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline="red", width=5)

    # return PIL image with circle
    return circled_img


# ===============================
# PDF Report Generator
# ===============================
def generate_pdf_report(patient_name, mrn, gender, doctor_name, date_str, notes, left_result, right_result):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(180, height - 50, "Medivora Eye Report")

    # Patient details
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Name: {patient_name}")
    c.drawString(50, height - 120, f"MRN: {mrn}")
    c.drawString(50, height - 140, f"Gender: {gender}")
    c.drawString(50, height - 160, f"Doctor: {doctor_name}")
    c.drawString(50, height - 180, f"Date: {date_str}")
    if notes.strip():
        c.drawString(50, height - 200, f"Notes: {notes}")

    # Left eye results
    if left_result:
        pred_label, confidence, probs, circled_img = left_result
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 240, "Left Eye Result:")
        c.setFont("Helvetica", 12)
        c.drawString(70, height - 260, f"Prediction: {pred_label}")
        c.drawString(70, height - 280, f"Confidence: {confidence*100:.2f}%")
        y_prob = height - 300
        for cls, p in zip(VGG8_CLASSES, probs):
            c.drawString(70, y_prob, f"{cls}: {p*100:.2f}%")
            y_prob -= 12
        try:
            circled_img.thumbnail((150, 150))
            img_reader = ImageReader(circled_img)
            c.drawImage(img_reader, 350, height - 350)
        except Exception:
            pass

    # Right eye results
    if right_result:
        pred_label, confidence, probs, circled_img = right_result
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 480, "Right Eye Result:")
        c.setFont("Helvetica", 12)
        c.drawString(70, height - 500, f"Prediction: {pred_label}")
        c.drawString(70, height - 520, f"Confidence: {confidence*100:.2f}%")
        y_prob = height - 540
        for cls, p in zip(VGG8_CLASSES, probs):
            c.drawString(70, y_prob, f"{cls}: {p*100:.2f}%")
            y_prob -= 12
        try:
            circled_img.thumbnail((150, 150))
            img_reader = ImageReader(circled_img)
            c.drawImage(img_reader, 350, height - 590)
        except Exception:
            pass

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# ===============================
# UI
# ===============================
st.title("Medivora – Eye Disease Detection (OCT)")
st.write("Enter patient details and upload OCT images for left and right eyes to get AI predictions.")

with st.form("medivora_patient_form_unique"):
    patient_name = st.text_input("Patient Name")
    mrn = st.text_input("MRN (Medical Record Number)")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    doctor_name = st.text_input("Doctor Name")
    date_str = st.date_input("Date", value=datetime.today())

    st.markdown("### Upload OCT Images")
    left_eye_file = st.file_uploader("Upload Left Eye Image", type=["jpg", "jpeg", "png"], key="left_eye")
    right_eye_file = st.file_uploader("Upload Right Eye Image", type=["jpg", "jpeg", "png"], key="right_eye")

    notes = st.text_area("Notes / Symptoms", placeholder="Optional")

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    if not patient_name or not mrn:
        st.error("Please fill in at least Patient Name and MRN.")
    elif not left_eye_file and not right_eye_file:
        st.error("Please upload at least one eye image.")
    else:
        left_result = None
        right_result = None

        # Left Eye
        if left_eye_file:
            st.subheader("Left Eye Prediction")
            left_bytes = left_eye_file.read()
            left_img = Image.open(io.BytesIO(left_bytes)).convert("RGB")
            st.image(left_img, caption="Left Eye OCT Image", use_column_width=True)
            with st.spinner("Analyzing Left Eye..."):
                x = preprocess(left_bytes)
                raw = session.run(None, {INPUT_NAME: x})[0]
                probs = to_probabilities(raw)
                pred_idx = int(np.argmax(probs))
                pred_label = VGG8_CLASSES[pred_idx]
                confidence = float(probs[pred_idx])

                # GradCAM with circle
                circled_img = generate_gradcam_with_circle(left_img, pred_idx)

            st.write(f"**Disease Class:** {pred_label}")
            st.write(f"**Confidence:** {confidence*100:.2f}%")
            st.image(circled_img, caption="Left Eye Disease Spot (Grad-CAM)", use_column_width=True)
            left_result = (pred_label, confidence, probs, circled_img)

        # Right Eye
        if right_eye_file:
            st.subheader("Right Eye Prediction")
            right_bytes = right_eye_file.read()
            right_img = Image.open(io.BytesIO(right_bytes)).convert("RGB")
            st.image(right_img, caption="Right Eye OCT Image", use_column_width=True)
            with st.spinner("Analyzing Right Eye..."):
                x = preprocess(right_bytes)
                raw = session.run(None, {INPUT_NAME: x})[0]
                probs = to_probabilities(raw)
                pred_idx = int(np.argmax(probs))
                pred_label = VGG8_CLASSES[pred_idx]
                confidence = float(probs[pred_idx])

                # GradCAM with circle
                circled_img = generate_gradcam_with_circle(right_img, pred_idx)

            st.write(f"**Disease Class:** {pred_label}")
            st.write(f"**Confidence:** {confidence*100:.2f}%")
            st.image(circled_img, caption="Right Eye Disease Spot (Grad-CAM)", use_column_width=True)
            right_result = (pred_label, confidence, probs, circled_img)

        # Generate PDF with both eyes
        pdf_buffer = generate_pdf_report(
            patient_name, mrn, gender, doctor_name, date_str, notes, left_result, right_result
        )
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer,
            file_name=f"{patient_name}_eye_report.pdf",
            mime="application/pdf"
        )
        st.success("Prediction complete. PDF ready to download.")

