import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import base64
from pdf2image import convert_from_path
from PIL import Image
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=api_key)

# Load sentence transformer for semantic search
print("🔄 Loading sentence transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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
    
    # Normalize embeddings for cosine similarity
    embeddings_normalized = embeddings.astype('float32')
    faiss.normalize_L2(embeddings_normalized)
    
    # Create IDs array (using page numbers as IDs)
    ids = np.array([metadata['page'] for metadata in page_metadata], dtype=np.int64)
    
    # Add embeddings with custom IDs
    index.add_with_ids(embeddings_normalized, ids)
    
    return index

def create_page_summaries(pdf_path, output_dir="pdf_pages"):
    """Convert PDF to images and create detailed summaries"""
    print(f"📄 Processing PDF: {pdf_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
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
            
            # Create embedding for the summary
            embedding = embedding_model.encode(summary)
            
            # Store metadata with page number as key
            page_metadata[page_num] = {
                'page': page_num,
                'image_path': image_path,
                'summary': summary
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
        with open('page_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(page_metadata, f, indent=2, ensure_ascii=False)
        
        # Save FAISS index
        faiss.write_index(index, "page_embeddings.index")
        
        print(f"💾 Page metadata saved to page_metadata.json")
        print(f"💾 FAISS index with IDs saved to page_embeddings.index")
        
        return page_metadata, index
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return {}, None

def load_page_data():
    """Load existing page metadata and FAISS index"""
    try:
        # Load page metadata
        with open('page_metadata.json', 'r', encoding='utf-8') as f:
            page_metadata = json.load(f)
        
        # Convert string keys back to integers
        page_metadata = {int(k): v for k, v in page_metadata.items()}
        
        # Load FAISS index
        index = faiss.read_index("page_embeddings.index")
        
        print(f"📁 Loaded {len(page_metadata)} pages and FAISS index")
        return page_metadata, index
        
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        return {}, None

def search_relevant_pages(query, page_metadata, index, top_k=2):
    """Search for most relevant pages using FAISS with ID mapping"""
    print(f"🔍 Searching for pages relevant to: '{query}'")
    
    # Create embedding for the query
    query_embedding = embedding_model.encode([query]).astype('float32')
    
    # Normalize query embedding for cosine similarity
    faiss.normalize_L2(query_embedding)
    
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

def answer_query_with_vision(query, page_metadata, index, top_k=2):
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
        print(f"   Page {page_id}: {metadata['summary'][:100]}...")

# Main execution
def main():
    pdf_path = r"pdf\Schematic Diagram of CCPP.pdf"
    
    # Check if data already exists
    if os.path.exists('page_metadata.json') and os.path.exists('page_embeddings.index'):
        print("📁 Loading existing page data and FAISS index...")
        page_metadata, index = load_page_data()
    else:
        print("📄 Creating page summaries and FAISS index for the first time...")
        if os.path.exists(pdf_path):
            page_metadata, index = create_page_summaries(pdf_path)
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