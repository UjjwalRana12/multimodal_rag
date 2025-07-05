import os
from dotenv import load_dotenv
from rag_multimodal_test import RAGMultiModalModel


# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in .env")
if not hf_token:
    raise ValueError("HUGGINGFACE_HUB_TOKEN is not set in .env")


os.environ["HF_TOKEN"] = hf_token


MODEL_NAME = "vidore/colpali-v1.2"  
INDEX_NAME = "my_pdf_index"
PDF_DIR = "pdf"  
QUESTION = "Explain the diagram or flow shown in detail."

# Step 1: Load model
print("🔄 Loading model...")
try:
    rag = RAGMultiModalModel.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Step 2: Index the PDF(s)
print("📄 Indexing documents...")
rag.index(
    input_path=PDF_DIR,
    index_name=INDEX_NAME,
    store_collection_with_index=True, 
    overwrite=True
)

# Step 3: Search and get results
print("🔍 Searching with query...")
results = rag.search(QUESTION, k=3)

# Step 4: Display results
print("\n📑 Top Results:")
for i, result in enumerate(results):
    print(f"\nResult #{i+1}")
    print(f"📄 Doc ID: {result['doc_id']}")
    print(f"📃 Page: {result['page_num']}")
    print(f"📊 Score: {result['score']}")
    print(f"📎 Metadata: {result.get('metadata', {})}")
    print(f"🖼️  base64: {'✅ present' if result.get('base64') else '❌ missing'}")

from openai import OpenAI

client = OpenAI()


if results[0].get("base64"):
    print("\n💬 Asking OpenAI Vision model...")
    vision_response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QUESTION},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{results[0]['base64']}"}}
                ]
            }
        ],
        max_tokens=1000
    )
    print("🧠 OpenAI Answer:\n", vision_response.choices[0].message.content)
else:
    print("⚠️ No base64 image found in result to send to Vision model.")
