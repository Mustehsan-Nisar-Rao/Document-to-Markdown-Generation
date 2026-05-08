import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import gc

st.set_page_config(page_title="Document to Markdown", layout="wide")
st.title("📄 Document to Markdown Generator")

INSTRUCTION = "Convert this document image to Markdown format."

@st.cache_resource
def load_model():
    try:
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Load with minimal memory
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True
        )
        
        return model, processor
    except Exception as e:
        st.error(f"Load failed: {e}")
        return None, None

uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, width=350)
    
    with col2:
        with st.spinner("Loading model (this may take 2-3 minutes)..."):
            model, processor = load_model()
        
        if model:
            with st.spinner("Generating..."):
                try:
                    conversation = [{
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]
                    }]
                    
                    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[image], return_tensors="pt")
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False
                        )
                    
                    result = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    st.text_area("Output", result, height=400)
                    st.download_button("Download", result, file_name="output.md")
                    
                except Exception as e:
                    st.error(f"Generation failed: {e}")
        else:
            st.error("Model failed to load. Try restarting the app.")
