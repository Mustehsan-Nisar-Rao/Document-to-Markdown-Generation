import traceback
import sys

try:
    import streamlit as st
    from PIL import Image
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
    from peft import PeftModel
    
    st.set_page_config(page_title="Document to Markdown", layout="wide")
    st.title("📄 Document to Markdown Generator")
    
    INSTRUCTION = (
        "Convert the document image into well-structured Markdown. "
        "Preserve all headings, equations, tables, lists, and captions faithfully."
    )
    
    @st.cache_resource
    def load_model():
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            st.info("Loading base model...")
            base = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct",
                torch_dtype=torch.float16,
                device_map="cpu",  # CPU pe force karo
                trust_remote_code=True
            )
            
            st.info("Loading PEFT weights...")
            model = PeftModel.from_pretrained(base, "Weights/")
            model.eval()
            
            st.info("Loading processor...")
            processor = AutoProcessor.from_pretrained("Weights/", trust_remote_code=True)
            
            return model, processor
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            st.code(traceback.format_exc())
            return None, None
    
    uploaded = st.file_uploader("Upload Document Image", type=["png", "jpg", "jpeg"])
    
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, width=400)
        with col2:
            with st.spinner("Loading model..."):
                model, processor = load_model()
            
            if model is not None:
                with st.spinner("Generating..."):
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
                    inputs = processor(text=[text], images=[image], return_tensors="pt")
                    
                    if torch.cuda.is_available():
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        model = model.to("cuda")
                    
                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=False,
                        )
                    
                    generated = output_ids[0][inputs["input_ids"].shape[1]:]
                    result = processor.decode(generated, skip_special_tokens=True)
                    
                    st.text_area("Output", result, height=500)
                    st.download_button("Download", result, file_name="output.md")
    
except Exception as e:
    st.error(f"App Error: {str(e)}")
    st.code(traceback.format_exc())
