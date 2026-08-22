import streamlit as st
from PIL import Image
import base64
import io
import requests

st.set_page_config(page_title="Document to Markdown", layout="wide")
st.title("📄 Document to Markdown Generator")

INSTRUCTION = "Convert this document image to Markdown format."

# ─────────────────────────────────────────
# HUGGING FACE INFERENCE API CONFIG
# ─────────────────────────────────────────
# Set this in Streamlit Cloud -> App Settings -> Secrets:
#   HF_TOKEN = "your-hf-token-here"
HF_TOKEN = st.secrets["HF_TOKEN"]

# Qwen2-VL-2B-Instruct via the HF Inference Providers router (chat-completions style).
# Running remotely means we never load the 2B-param model into this app's
# memory, so it works fine within Streamlit Community Cloud's ~1GB RAM limit.
API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def image_to_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a base64 data URL for the API payload."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def query_hf_inference(image: Image.Image, instruction: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                    {"type": "text", "text": instruction},
                ],
            }
        ],
        "max_tokens": 1024,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"HF API error {response.status_code}: {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, width=350)

    with col2:
        with st.spinner("Generating Markdown via Hugging Face Inference API..."):
            try:
                result = query_hf_inference(image, INSTRUCTION)
                st.text_area("Output", result, height=400)
                st.download_button("Download", result, file_name="output.md")
            except Exception as e:
                st.error(f"Generation failed: {e}")
