import streamlit as st
import os
import io
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import onnxruntime as ort
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


load_dotenv(override=False)

# ---------------- CONFIG ----------------
class Config:
    IMG_SIZE = (224, 224)
    CLASS_NAMES = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']
    MODEL_PATH = r"model/ensemble_model.onnx"
    CLASS_DESCRIPTIONS = {
        'AMD': 'Age-related Macular Degeneration: affects central vision.',
        'CNV': 'Choroidal Neovascularization: abnormal blood vessel growth under retina.',
        'CSR': 'Central Serous Retinopathy: fluid buildup under retina.',
        'DME': 'Diabetic Macular Edema: swelling in the retina due to diabetes.',
        'DR': 'Diabetic Retinopathy: retinal blood vessel damage from diabetes.',
        'DRUSEN': 'Drusen: yellow deposits under retina, often in AMD.',
        'MH': 'Macular Hole: a small break in the central part of the retina.',
        'NORMAL': 'No apparent abnormalities detected.'
    }
    SEVERITY_MAPPING = {
        'NORMAL': 'Low', 'DRUSEN': 'Low', 'CSR': 'Medium', 'MH': 'Medium',
        'AMD': 'High', 'CNV': 'High', 'DME': 'High', 'DR': 'High'
    }

# -------------- MODEL LOADER --------------
class OCTModel:
    def __init__(self, path):  # ✅ Fixed constructor
        self.session = ort.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image):
        out = self.session.run([self.output_name], {self.input_name: image})
        return out[0][0]


@st.cache_resource(show_spinner=False)
def _get_onnx_model():
    return OCTModel(Config.MODEL_PATH)

# -------------- IMAGE PROCESSING --------------
def preprocess_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(Config.IMG_SIZE)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, img


def _to_probabilities(raw: np.ndarray) -> np.ndarray:
    vec = np.asarray(raw).astype("float64")
    if vec.ndim > 1:
        vec = vec.reshape(-1)
    if (vec < 0).any() or vec.max() > 1.0 or vec.sum() <= 0.0:
        exps = np.exp(vec - np.max(vec))
        vec = exps / np.sum(exps)
    return vec.astype("float32")


def _predict_proba(model: OCTModel, image_batch: np.ndarray) -> np.ndarray:
    out = model.session.run([model.output_name], {model.input_name: image_batch})
    raw = out[0][0]
    return _to_probabilities(raw)


def generate_occlusion_heatmap(img: Image.Image, *, model: OCTModel, class_idx: int, grid: int = 8):
    img_rgb = img.convert("RGB")
    base_batch, resized_pil = preprocess_image(img_rgb)
    base_probs = _predict_proba(model, base_batch)
    base_score = float(base_probs[class_idx])

    resized = np.asarray(resized_pil).copy()
    mean_px = resized.mean(axis=(0, 1), keepdims=True).astype(np.uint8)
    h, w = resized.shape[:2]
    ph = max(1, h // grid)
    pw = max(1, w // grid)
    heat = np.zeros((grid, grid), dtype=np.float32)

    total = grid * grid
    p = st.progress(0)
    k = 0
    for gy in range(grid):
        for gx in range(grid):
            y0 = gy * ph
            y1 = h if gy == grid - 1 else (gy + 1) * ph
            x0 = gx * pw
            x1 = w if gx == grid - 1 else (gx + 1) * pw

            occluded = resized.copy()
            occluded[y0:y1, x0:x1] = mean_px
            oc_batch = np.expand_dims(occluded.astype(np.float32) / 255.0, axis=0)
            probs = _predict_proba(model, oc_batch)
            score = float(probs[class_idx])
            heat[gy, gx] = max(0.0, base_score - score)

            k += 1
            p.progress(min(1.0, k / max(1, total)))

    p.empty()

    heat_up = cv2.resize(heat, (Config.IMG_SIZE[0], Config.IMG_SIZE[1]), interpolation=cv2.INTER_CUBIC)
    mn = float(np.min(heat_up))
    mx = float(np.max(heat_up))
    heat_norm = (heat_up - mn) / (mx - mn) if mx > mn else np.zeros_like(heat_up, dtype=np.float32)

    heat_u8 = (heat_norm * 255.0).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(np.asarray(resized_pil), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(base_bgr, 0.60, colored, 0.40, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay_img = Image.fromarray(overlay_rgb)

    y, x = np.unravel_index(int(np.argmax(heat_norm)), heat_norm.shape)
    return overlay_img, heat_norm, (int(x), int(y))


def _get_groq_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError("GROQ_API_KEY is not set")
    model = os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
    return ChatGroq(model=model, temperature=0.2, groq_api_key=api_key)


def generate_llm_report(*, name: str, diagnosis: str, confidence: float, risk: str, prob_dict: dict, hotspot_xy: tuple[int, int] | None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sorted_probs = sorted(prob_dict.items(), key=lambda x: float(x[1]), reverse=True)
    top_probs = "\n".join([f"- {cls}: {float(p):.2%}" for cls, p in sorted_probs[:5]])
    hotspot = f"({hotspot_xy[0]}, {hotspot_xy[1]})" if hotspot_xy else "unknown"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You generate concise, structured OCT scan AI reports for clinical support. You must include a clear disclaimer that this is not medical advice and requires clinician review. Avoid claiming certainty.\n",
            ),
            (
                "user",
                "Patient image: {name}\nTimestamp: {ts}\n\nModel output\n- Predicted diagnosis: {diagnosis}\n- Confidence: {confidence_pct}\n- Risk level: {risk}\n- Top probabilities:\n{top_probs}\n\nLocalization\n- Affected-area heatmap hotspot (x,y on 224x224): {hotspot}\n\nWrite a report with sections:\n1) Summary\n2) Findings (what the model suggests)\n3) Differential considerations (2-4 items)\n4) Recommended next steps\n5) Disclaimer\n",
            ),
        ]
    )

    llm = _get_groq_llm()
    msg = prompt.format_messages(
        name=name,
        ts=ts,
        diagnosis=diagnosis,
        confidence_pct=f"{confidence:.2%}",
        risk=risk,
        top_probs=top_probs,
        hotspot=hotspot,
    )
    resp = llm.invoke(msg)
    return str(resp.content).strip()

# -------------- REPORT GENERATOR --------------
def generate_report(name, diagnosis, confidence, prob_dict):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    risk = Config.SEVERITY_MAPPING[diagnosis]
    desc = Config.CLASS_DESCRIPTIONS[diagnosis]
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

    report = f"""
OCT EYE SCAN REPORT
====================

Date: {ts}
Patient Image: {name}

Diagnosis: {diagnosis}
AI Confidence: {confidence:.2%}
Risk Level: {risk}

Description:
{desc}

Other Probabilities:
---------------------
"""
    for cls, prob in sorted_probs:
        report += f"{cls}: {prob:.2%}\n"
    return report

# -------------- UI SETUP --------------
st.set_page_config(page_title="OCT Scan Diagnosis", layout="centered")

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
    }
    .sub-text {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🩺 OCT Eye Scan Diagnosis</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Upload an OCT image to get a diagnosis with confidence and risk level.</p>', unsafe_allow_html=True)

with st.expander("Localization settings (occlusion heatmap)", expanded=False):
    st.caption("Occlusion heatmaps approximate important regions by masking parts of the image and measuring the confidence drop. This is not Grad-CAM.")
    enable_heatmap = st.checkbox("Generate occlusion heatmap", value=True)
    heatmap_grid = st.slider("Occlusion grid size", min_value=4, max_value=16, value=8, step=1)

uploaded_file = st.file_uploader("📤 Upload OCT Image", type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if "analysis" not in st.session_state:
        st.session_state.analysis = None

    if st.button("🔍 Analyze Scan"):
        with st.spinner("Analyzing scan..."):
            img_batch, processed = preprocess_image(image)
            model = _get_onnx_model()
            preds = _to_probabilities(model.predict(img_batch))

            idx = int(np.argmax(preds))
            diagnosis = Config.CLASS_NAMES[idx]
            confidence = float(preds[idx])
            risk = Config.SEVERITY_MAPPING[diagnosis]

            prob_dict = {Config.CLASS_NAMES[i]: float(preds[i]) for i in range(len(preds))}

            heatmap_png = None
            hotspot_xy = None
            if enable_heatmap:
                with st.spinner("Generating occlusion heatmap..."):
                    heatmap_img, _, hotspot_xy = generate_occlusion_heatmap(
                        processed,
                        model=model,
                        class_idx=idx,
                        grid=int(heatmap_grid),
                    )
                heatmap_buf = io.BytesIO()
                heatmap_img.save(heatmap_buf, format="PNG")
                heatmap_png = heatmap_buf.getvalue()

            report = generate_report(uploaded_file.name, diagnosis, confidence, prob_dict)

            st.session_state.analysis = {
                "name": uploaded_file.name,
                "diagnosis": diagnosis,
                "confidence": confidence,
                "risk": risk,
                "prob_dict": prob_dict,
                "heatmap_png": heatmap_png,
                "hotspot_xy": hotspot_xy,
                "report": report,
            }

    if st.session_state.analysis:
        a = st.session_state.analysis
        st.success(f"✅ Diagnosis: *{a['diagnosis']}*")
        st.info(f"📊 Confidence: *{a['confidence']:.2%}*")
        st.warning(f"⚠ Risk Level: *{a['risk']}*")
        st.markdown(f"*Description:* {Config.CLASS_DESCRIPTIONS[a['diagnosis']]}")

        if a.get("heatmap_png"):
            st.image(a["heatmap_png"], caption="Occlusion heatmap (approx. affected area)", use_column_width=True)
            st.download_button("🖼 Download Heatmap", a["heatmap_png"], file_name="oct_occlusion_heatmap.png", mime="image/png")
        else:
            st.caption("Occlusion heatmap generation is disabled.")

        st.download_button("📄 Download Report", a["report"], file_name="oct_scan_report.txt")

        if st.button("🧠 Generate AI Report (Groq Llama 3)"):
            try:
                llm_report = generate_llm_report(
                    name=a["name"],
                    diagnosis=a["diagnosis"],
                    confidence=float(a["confidence"]),
                    risk=a["risk"],
                    prob_dict=a["prob_dict"],
                    hotspot_xy=a["hotspot_xy"],
                )
                st.text_area("AI Report", llm_report, height=320)
                st.download_button("📄 Download AI Report", llm_report, file_name="oct_ai_report.txt")
            except Exception as e:
                st.error(str(e))

# Footer
st.markdown("""
---
⚠ This is an AI-based tool intended for research and support. Always consult a specialist for medical decisions.
""")
