from byaldi import RAGMultiModalModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

# Load a CPU-compatible model
print("🔄 Loading CPU-compatible model...")
try:
    
    model = RAGMultiModalModel.from_pretrained("sentence-transformers/clip-ViT-B-32")
    # Or: model = RAGMultiModalModel.from_pretrained("openai/clip-vit-base-patch32")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Create an index from your image
print("📄 Indexing image...")
model.index(
    input_path="Schematic Diagram of CCPP.png",  # Single image file
    index_name="ccpp_diagram_index",
    store_collection_with_index=True,
    overwrite=True
)

# Search/query the indexed content
query = "Explain the process shown in the image"
print("🔍 Searching...")
results = model.search(query, k=1)

# Display results
if results:
    print("📑 Results:")
    for i, result in enumerate(results):
        print(f"\nResult #{i+1}")
        print(f"📄 Doc ID: {result['doc_id']}")
        print(f"📊 Score: {result['score']}")
        print(f"🖼️ Base64: {'✅ present' if result.get('base64') else '❌ missing'}")
else:
    print("❌ No results found")

print("🔄 Loading BLIP model for CPU...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Set to CPU mode
device = torch.device("cpu")
model.to(device)

# Load and process image
image_path = "Schematic Diagram of CCPP.png"
image = Image.open(image_path).convert('RGB')

print("📄 Analyzing image...")

# Generate unconditional caption
inputs = processor(image, return_tensors="pt").to(device)
out = model.generate(**inputs, max_length=50)
caption = processor.decode(out[0], skip_special_tokens=True)
print(f"🧠 Image Caption: {caption}")

# Generate conditional caption with question
question = "What is the process shown in this power plant diagram?"
inputs = processor(image, question, return_tensors="pt").to(device) 
out = model.generate(**inputs, max_length=100)
answer = processor.decode(out[0], skip_special_tokens=True)
print(f"🧠 Answer: {answer}")
