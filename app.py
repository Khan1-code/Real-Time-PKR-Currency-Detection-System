import streamlit as st
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import io
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd

MODEL_PATH = "models/best_model.pth"

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="PKR Note Detector", page_icon="💸", layout="wide")

# ----------------------------
# Colorful + animated CSS
# ----------------------------
st.markdown("""
<style>
/* Animated colorful background */
.stApp{
  background:
    radial-gradient(circle at 10% 10%, rgba(255, 0, 128, 0.14), transparent 40%),
    radial-gradient(circle at 90% 15%, rgba(0, 200, 255, 0.16), transparent 45%),
    radial-gradient(circle at 80% 90%, rgba(0, 255, 160, 0.14), transparent 40%),
    linear-gradient(120deg, rgba(20,24,38,0.02), rgba(20,24,38,0.02));
}

/* Layout */
.block-container{max-width:1250px; padding-top:1.0rem; padding-bottom:2rem;}

/* Glass card */
.card{
  background: rgba(255,255,255,0.70);
  border: 1px solid rgba(255,255,255,0.45);
  backdrop-filter: blur(12px);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 18px 45px rgba(0,0,0,0.08);
}

/* Hero */
@keyframes floaty {0%{transform:translateY(0)}50%{transform:translateY(-4px)}100%{transform:translateY(0)}}
@keyframes shakeX {0%,100%{transform:translateX(0)}20%{transform:translateX(-2px)}40%{transform:translateX(2px)}60%{transform:translateX(-1px)}80%{transform:translateX(1px)}}
@keyframes shimmer {
  0% { background-position: 0% 50%; }
  100% { background-position: 100% 50%; }
}
.brand{

  font-size: 2.2rem;
  font-weight: 950;
  letter-spacing: -0.03em;
  margin: 0;
  animation: floaty 2.6s ease-in-out infinite;
}
.brand.shake{ animation: shakeX 0.55s ease-in-out 1; }

.tagline{
  font-size: 1.05rem;
  margin-top: 0.2rem;
  opacity: 0.85;
}
.moving{
  display:inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.55);
  background: linear-gradient(90deg, rgba(255,0,128,0.16), rgba(0,200,255,0.18), rgba(0,255,160,0.16));
  background-size: 200% 200%;
  animation: shimmer 2.8s ease-in-out infinite;
  font-weight: 750;
}

/* Smooth entrance */
@keyframes fadeUp { from {opacity:0; transform: translateY(10px);} to {opacity:1; transform: translateY(0);} }
.fade { animation: fadeUp 0.32s ease-out 1; }

/* Confidence badges */
.badgeG{color:#0f7a3a; font-weight:900;}
.badgeO{color:#b45309; font-weight:900;}
.badgeR{color:#b91c1c; font-weight:900;}

/* Round progress bars */
div[data-testid="stProgressBar"] > div > div { border-radius: 999px; }

/* Small helper text */
small{opacity:0.75}

/* Make dataframe feel nicer */
[data-testid="stDataFrame"] { border-radius: 14px; overflow:hidden; }
/* ===========================
   CAMERA SIZE FIX (IMPORTANT)
   =========================== */

/* Make camera container full width */
div[data-testid="stCameraInput"] {
  width: 100% !important;
  max-width: 100% !important;
}

/* Remove inner padding that shrinks camera */
div[data-testid="stCameraInput"] > div {
  padding: 0 !important;
}

/* Make camera preview fill the box */
div[data-testid="stCameraInput"] video,
div[data-testid="stCameraInput"] img {
  width: 100% !important;
  height: 460px !important;   /* 👈 increase size here */
  object-fit: cover !important;
  border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Data model for History
# ----------------------------
@dataclass
class CaptureItem:
    ts: str
    image_bytes: bytes
    label: str
    conf: float
    status: str
    probs: Dict[str, float]
    margin: float

# ----------------------------
# Session State
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history: List[CaptureItem] = []
if "last_hash" not in st.session_state:
    st.session_state.last_hash = None
if "shake" not in st.session_state:
    st.session_state.shake = False

# ----------------------------
# Model
# ----------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(MODEL_PATH, map_location=device)
    classes = ckpt["classes"]

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, classes, device

@st.cache_data
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def predict(image: Image.Image):
    model, classes, device = load_model()
    tfm = get_transform()

    x = tfm(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    probs_dict = {str(c): float(p) for c, p in zip(classes, probs)}
    label = str(classes[int(probs.argmax())])
    conf = float(probs.max())
    return label, conf, probs_dict

def compute_status(conf: float, probs_dict: Dict[str, float]) -> Tuple[str, float, str]:
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    top1 = sorted_probs[0][1]
    top2 = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
    margin = float(top1 - top2)

    if conf < 0.50:
        return "NOT A CURRENCY NOTE", margin, "badgeR"
    elif conf < 0.70:
        return "UNSURE — PLEASE TRY AGAIN", margin, "badgeO"
    else:
        return "CONFIDENT", margin, "badgeG"

def history_df(items: List[CaptureItem]) -> pd.DataFrame:
    rows = []
    for i, it in enumerate(items, start=1):
        rows.append({
            "#": i,
            "time": it.ts,
            "prediction": it.label,
            "confidence": round(it.conf, 3),
            "status": it.status,
            "top1-top2 margin": round(it.margin, 3),
        })
    return pd.DataFrame(rows)

# ----------------------------
# Sidebar (Export + Clear + Toggles)
# ----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    show_probs = st.toggle("Show probabilities in history", value=False)
    show_top3 = st.toggle("Show Top-3 in latest result", value=True)
    st.caption("Tip: Flat note + bright light + fill the frame = best result.")
    st.divider()

    df = history_df(st.session_state.history)
    st.download_button(
        "⬇️ Export CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="pkr_note_detector_history.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=(len(st.session_state.history) == 0),
    )

    if st.button("🧹 Clear all history", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_hash = None
        st.toast("History cleared ✅", icon="🧹")

# ----------------------------
# Hero Header (animated)
# ----------------------------
brand_cls = "brand shake" if st.session_state.shake else "brand"

st.markdown(
    f"""
    <div style="text-align:center; margin-top:40px; margin-bottom:18px;">
        <div class="{brand_cls}" style="
            margin: 0 auto;
            font-size: 2.1rem;
            font-weight: 900;
        ">
            Real-Time PKR Currency Detection
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)




# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([0.52, 0.48], gap="large")

with left:
    with st.container(border=True):
        st.subheader("📷 Capture")
        st.caption("Take a photo. Each capture is saved with prediction + confidence.")
        img_file = st.camera_input("Take a picture")

with right:
    with st.container(border=True):
        st.subheader("🧠 Latest Result")
        




    if img_file is None:
        st.caption("Take a photo to see the result.")

    else:
        image_bytes = img_file.getvalue()
        img_hash = hashlib.md5(image_bytes).hexdigest()
        image = Image.open(io.BytesIO(image_bytes))

        is_new = (img_hash != st.session_state.last_hash)
        if is_new:
            st.session_state.last_hash = img_hash

            with st.spinner("Analyzing…"):
                label, conf, probs_dict = predict(image)
                time.sleep(0.15)

            status, margin, badge_class = compute_status(conf, probs_dict)

            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.history.insert(
                0,
                CaptureItem(ts=ts, image_bytes=image_bytes, label=label, conf=conf,
                            status=status, probs=probs_dict, margin=margin)
            )

            # trigger shake once after new capture
            st.session_state.shake = True
            st.toast("Thank you! Photo captured ✅", icon="📸")

        latest = st.session_state.history[0]
        status, margin, badge_class = compute_status(latest.conf, latest.probs)

        st.markdown("<div class='fade'>", unsafe_allow_html=True)
        st.image(latest.image_bytes, caption="Captured image", use_container_width=True)

        st.markdown(
            f"Status: <span class='{badge_class}'>{status}</span> "
            f"<small>• Top1–Top2 margin: {latest.margin:.2f}</small>",
            unsafe_allow_html=True,
        )

        if status == "CONFIDENT":
            st.success("Prediction is reliable ✅")
            st.metric("Prediction", f"{latest.label}", f"{int(latest.conf*100)}% confidence")
        elif status.startswith("UNSURE"):
            st.warning("Prediction may be wrong — try again ⚠️")
            st.metric("Prediction (maybe)", f"{latest.label}", f"{int(latest.conf*100)}% confidence")
        else:
            st.error("Rejected (not a note) ❌")
            st.metric("Prediction", "—", f"{int(latest.conf*100)}% confidence")

        if show_top3:
            st.write("**Top-3 probabilities**")
            top3 = sorted(latest.probs.items(), key=lambda x: x[1], reverse=True)[:3]
            for k, v in top3:
                st.write(f"{k}: {v:.3f}")
                st.progress(float(v))

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# History
# ----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧾 Capture History")
st.caption("A professional log of every capture. Export CSV from sidebar.")

if len(st.session_state.history) == 0:
    st.info("No captures yet.")
else:
    df = history_df(st.session_state.history)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.write("### Details")
    for i, item in enumerate(st.session_state.history, start=1):
        a, b = st.columns([0.22, 0.78], gap="large")
        with a:
            st.image(item.image_bytes, caption=f"Capture #{i}", use_container_width=True)
        with b:
            status, margin, badge_class = compute_status(item.conf, item.probs)
            st.markdown(f"**Time:** {item.ts}")
            st.markdown(f"**Prediction:** `{item.label}`")
            st.markdown(f"**Confidence:** `{item.conf:.2f}`")
            st.markdown(f"**Status:** <span class='{badge_class}'>{status}</span>", unsafe_allow_html=True)

            if show_probs:
                with st.expander("Show all probabilities"):
                    for k, v in sorted(item.probs.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"{k}: {v:.3f}")
                        st.progress(float(v))
        st.divider()

st.markdown("</div>", unsafe_allow_html=True)
