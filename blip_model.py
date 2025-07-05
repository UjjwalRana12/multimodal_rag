from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in .env")
if not hf_token:
    raise ValueError("HUGGINGFACE_HUB_TOKEN is not set in .env")


os.environ["HF_TOKEN"] = hf_token

# Load BLIP model and processor
print("🔄 Loading BLIP model (CPU)...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

device = torch.device("cpu")
model.to(device)

# Load image
image_path = "Schematic Diagram of CCPP.png"
image = Image.open(image_path).convert("RGB")

# 🔍 Unconditional Captioning
inputs = processor(image, return_tensors="pt").to(device)
out = model.generate(**inputs, max_length=50)
caption = processor.decode(out[0], skip_special_tokens=True)
print(f"\n🧠 Image Caption (auto): {caption}")

# ❓ Question Answering (conditional generation)
question = "What is the process shown in this power plant diagram?"
inputs = processor(image, question, return_tensors="pt").to(device)
out = model.generate(**inputs, max_length=100)
answer = processor.decode(out[0], skip_special_tokens=True)
print(f"\n🧠 Answer to your question: {answer}")
