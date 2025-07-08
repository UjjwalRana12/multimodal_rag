import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import base64
from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import shutil

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in .env")
if not hf_token:
    raise ValueError("HUGGINGFACE_HUB_TOKEN is not set in .env")


os.environ["HF_TOKEN"] = hf_token

client = OpenAI(api_key=api_key)

# Load CLIP model for multimodal embeddings
print("🔄 Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_embedding(image_path):
    """Get CLIP embedding for an image"""
    image = Image.open(image_path).convert('RGB')
    
    with torch.no_grad():
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        image_features = clip_model.get_image_features(**inputs) #embeddings createdd
        # Normalize the embeddings
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu().numpy().flatten()

def get_text_embedding(text):
    """Get CLIP embedding for text"""
    with torch.no_grad():
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
        text_features = clip_model.get_text_features(**inputs)
        # Normalize the embeddings
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy().flatten()

def get_detailed_summary(image_path, page_num):
    """Get detailed summary of a page"""
    print(f"🔄 Creating detailed summary for page {page_num}...")
    base64_image = encode_image_to_base64(image_path)
    
    summary_prompt = """
    Analyze this image/diagram in detail and provide a comprehensive summary including:
    1. What type of diagram/content this is
    2. Main components, elements, or sections visible
    3. Any text, labels, or annotations present
    4. Processes, flows, or relationships shown
    5. Technical details, measurements, or specifications
    6. Overall purpose or function illustrated
    
    Make the summary detailed and searchable so it can be used to match user queries.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": summary_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1500
    )
    
    return response.choices[0].message.content

def create_faiss_index_with_ids(embeddings, page_metadata):
    """Create FAISS index with ID mapping"""
    dimension = embeddings.shape[1]
    
    # Create base index
    base_index = faiss.IndexFlatIP(dimension)
    
    # Wrap with IDMap for custom IDs
    index = faiss.IndexIDMap(base_index)
    
    # Embeddings are already normalized from CLIP
    embeddings_normalized = embeddings.astype('float32')
    
    # Create IDs array (using page numbers as IDs)
    ids = np.array([metadata['page'] for metadata in page_metadata], dtype=np.int64)
    
    # Add embeddings with custom IDs
    index.add_with_ids(embeddings_normalized, ids)
    
    return index

def get_user_dir(user_id):
    """Return the directory for a given user/case."""
    base_dir = os.path.join("data", str(user_id))
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def create_page_summaries(pdf_path, user_id, use_image_embeddings=True):
    """Convert PDF to images and create detailed summaries for a user/case."""
    user_dir = get_user_dir(user_id)
    output_dir = os.path.join(user_dir, "pdf_pages")
    os.makedirs(output_dir, exist_ok=True)

    # Save a copy of the original PDF in the user's folder
    pdf_copy_path = os.path.join(user_dir, os.path.basename(pdf_path))
    if not os.path.exists(pdf_copy_path):
        shutil.copy2(pdf_path, pdf_copy_path)

    print(f"📄 Processing PDF: {pdf_path}")
    
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)
        print(f"📄 Found {len(images)} pages in PDF")
        
        page_metadata = {}  # Dictionary mapping page_id -> metadata
        embeddings_list = []
        
        for i, image in enumerate(images):
            page_num = i + 1
            image_path = os.path.join(output_dir, f"page_{page_num}.png")
            
            # Save page as image
            image.save(image_path, "PNG")
            
            # Get detailed summary
            summary = get_detailed_summary(image_path, page_num)
            
            # Create embedding based on choice
            if use_image_embeddings:
                print(f"🔄 Creating CLIP image embedding for page {page_num}...")
                embedding = get_image_embedding(image_path)
            else:
                print(f"🔄 Creating CLIP text embedding for page {page_num}...")
                embedding = get_text_embedding(summary)
            
            # Store metadata with page number as key
            page_metadata[page_num] = {
                'page': page_num,
                'image_path': image_path,
                'summary': summary,
                'embedding_type': 'image' if use_image_embeddings else 'text'
            }
            
            embeddings_list.append(embedding)
            
            print(f"✅ Page {page_num} summary created")
            print(f"Summary: {summary[:200]}...\n")
            print("-" * 80)
        
        # Convert to numpy array
        embeddings = np.array(embeddings_list)
        
        # Create FAISS index with IDs
        print("🔄 Creating FAISS index with ID mapping...")
        index = create_faiss_index_with_ids(embeddings, list(page_metadata.values()))
        
        # Save metadata dictionary
        metadata_to_save = {k: {**v, 'embedding_type': v['embedding_type']} for k, v in page_metadata.items()}
        with open(os.path.join(user_dir, 'page_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata_to_save, f, indent=2, ensure_ascii=False)
        
        # Save FAISS index
        faiss.write_index(index, os.path.join(user_dir, "page_embeddings.index"))
        
        print(f"💾 Page metadata saved to {os.path.join(user_dir, 'page_metadata.json')}")
        print(f"💾 FAISS index with IDs saved to {os.path.join(user_dir, 'page_embeddings.index')}")
        
        return page_metadata, index
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return {}, None


def search_relevant_pages(query, page_metadata, index, top_k=1):
    """Search for most relevant pages using FAISS with CLIP embeddings"""
    print(f"🔍 Searching for pages relevant to: '{query}'")
    
    # Check what type of embeddings we're using
    embedding_type = list(page_metadata.values())[0].get('embedding_type', 'text')
    
    if embedding_type == 'image':
        print("⚠️ Using image embeddings - text queries might not work as well")
        print("💡 Consider using image+text hybrid or text embeddings for better text search")
    
    # Create embedding for the query (always text)
    query_embedding = get_text_embedding(query)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    
    # Search using FAISS
    scores, ids = index.search(query_embedding, top_k)
    
    print(f"📊 Top {len(ids[0])} relevant pages:")
    relevant_pages = []
    
    for i, page_id in enumerate(ids[0]):
        if page_id != -1 and page_id in page_metadata:  # Valid ID
            page = page_metadata[page_id]
            score = scores[0][i]
            print(f"   Page {page['page']}: {score:.3f} similarity")
            relevant_pages.append(page)
    
    return relevant_pages


def answer_query_with_vision(query, page_metadata, index, top_k=1):
    """Answer user query by finding relevant pages and using OpenAI Vision"""
    
    # Find most relevant pages using FAISS
    relevant_pages = search_relevant_pages(query, page_metadata, index, top_k)
    
    if not relevant_pages:
        return "❌ No relevant pages found for your query."
    
    print(f"\n💬 Analyzing {len(relevant_pages)} most relevant pages with OpenAI Vision...")
    
    results = []
    
    for page in relevant_pages:
        print(f"🔄 Analyzing Page {page['page']}...")
        
        base64_image = encode_image_to_base64(page['image_path'])
        
        enhanced_prompt = f"""
        User Query: "{query}"
        
        Page Summary: {page['summary'][:500]}...
        
        Based on the user's query and this page content, please provide a detailed answer.
        Focus specifically on addressing the user's question using the information visible in this image.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": enhanced_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            result = {
                'page': page['page'],
                'answer': response.choices[0].message.content,
                'image_path': page['image_path']
            }
            
            results.append(result)
            print(f"✅ Page {page['page']} Analysis Complete")
            
        except Exception as e:
            print(f"❌ Error analyzing page {page['page']}: {e}")
    
    return results

def get_page_by_id(page_id, page_metadata):
    """Get page metadata by ID"""
    return page_metadata.get(page_id, None)

def list_all_pages(page_metadata):
    """List all available pages"""
    print("📚 Available pages:")
    for page_id, metadata in sorted(page_metadata.items()):
        embedding_type = metadata.get('embedding_type', 'unknown')
        print(f"   Page {page_id} ({embedding_type}): {metadata['summary'][:100]}...")

# Main execution
def main():
    user_id = input("Enter user/case ID: ").strip()
    pdf_path = input("Enter path to PDF: ").strip()
    
    # Always process the PDF fresh each time
    print("📄 Processing PDF...")
    if os.path.exists(pdf_path):
        print("\n🤔 Choose embedding type:")
        print("1. Image embeddings (better for visual similarity)")
        print("2. Text embeddings (better for text queries)")
        choice = input("Enter choice (1 or 2, default=2): ").strip()
        
        use_image_embeddings = choice == "1"
        page_metadata, index = create_page_summaries(pdf_path, user_id, use_image_embeddings=use_image_embeddings)
    else:
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    if not page_metadata or index is None:
        print("❌ No page data available")
        return
    
    print(f"\n✅ Ready! {len(page_metadata)} pages indexed with FAISS and ready for queries.")
    print(f"📊 FAISS index size: {index.ntotal} vectors")
    
    # Interactive query loop
    while True:
        print("\n" + "="*60)
        print("Commands: 'list' to see all pages, 'page N' to get specific page, or ask a question")
        user_input = input("🤔 Enter your question (or 'quit' to exit): ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input.lower() == 'list':
            list_all_pages(page_metadata)
            continue
        
        if user_input.lower().startswith('page '):
            try:
                page_num = int(user_input.split()[1])
                page = get_page_by_id(page_num, page_metadata)
                if page:
                    print(f"\n📄 Page {page_num}:")
                    print(f"Summary: {page['summary']}")
                else:
                    print(f"❌ Page {page_num} not found")
            except (ValueError, IndexError):
                print("❌ Invalid page command. Use 'page N' where N is a number")
            continue
        
        if user_input.strip():
            results = answer_query_with_vision(user_input, page_metadata, index, top_k=2)
            
            print(f"\n🧠 Answer based on most relevant pages:")
            print("="*60)
            
            if isinstance(results, str):  # Error message
                print(results)
            else:
                for result in results:
                    print(f"\n📄 From Page {result['page']}:")
                    print(result['answer'])
                    print("-" * 40)

if __name__ == "__main__":
    main()