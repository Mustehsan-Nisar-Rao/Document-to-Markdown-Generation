import subprocess
import sys

# Auto-install torchvision if missing
try:
    import torchvision
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision"])

import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

INSTRUCTION = (
    "Convert the document image into well-structured Markdown. "
    "Preserve all headings, equations, tables, lists, and captions faithfully."
)

@st.cache_resource
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, "Weights/")
    model.eval()
    processor = AutoProcessor.from_pretrained("Weights/", trust_remote_code=True)
    return model, processor

def generate_markdown(model, processor, image):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": INSTRUCTION}
            ]
        }
    ]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True)

st.set_page_config(page_title="Document to Markdown", layout="wide")
st.title("📄 Document to Markdown Generator")
st.write("Upload a document image and get structured Markdown output.")

uploaded = st.file_uploader("Upload Document Image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Input Image")
        st.image(image, width=400)
    with col2:
        st.subheader("📝 Generated Markdown")
        with st.spinner("Generating..."):
            model, processor = load_model()
            result = generate_markdown(model, processor, image)
        st.text_area("Output", result, height=500)
        st.download_button("💾 Download Markdown", result, file_name="output.md")
