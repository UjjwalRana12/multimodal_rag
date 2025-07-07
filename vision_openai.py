import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import base64
from pdf2image import convert_from_path
from PIL import Image

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=api_key)

print("🔄 Using OpenAI text-embedding-3-small model for embeddings...")

def get_user_dir(user_id):
    """Return the directory for a given user/case."""
    base_dir = os.path.join("data", str(user_id))
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def get_openai_embedding(text, model="text-embedding-3-small"):
    """Get embedding from OpenAI's embedding model"""
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"❌ Error getting OpenAI embedding: {e}")
        return None

def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_with_openai(image_path, page_num):
    """Extract text from image using OpenAI Vision API"""
    print(f"🔍 Extracting text using OpenAI OCR for page {page_num}...")
    base64_image = encode_image_to_base64(image_path)
    
    ocr_prompt = """
    Please extract ALL text content from this image. 
    
    Instructions:
    1. Extract every piece of readable text, including:
       - Labels, titles, and headings
       - Numbers, measurements, and values
       - Annotations and captions
       - Any technical terms or abbreviations
       - Text in tables, charts, or diagrams
    
    2. Preserve the structure and context where possible
    3. If there's no readable text, respond with "No readable text found"
    4. Return only the extracted text, nothing else
    
    Extracted text:
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ocr_prompt},
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
        
        extracted_text = response.choices[0].message.content.strip()
        
        if extracted_text and extracted_text.lower() != "no readable text found":
            print(f"✅ OpenAI OCR extracted {len(extracted_text)} characters")
            print(f"📝 Sample text: {extracted_text[:200]}...")
            return extracted_text
        else:
            print("⚠️ No readable text found by OpenAI OCR")
            return ""
            
    except Exception as e:
        print(f"❌ OpenAI OCR extraction failed: {e}")
        return ""

def get_detailed_summary(image_path, page_num):
    """Get detailed visual summary of a page using OpenAI Vision"""
    print(f"🔄 Creating detailed visual summary for page {page_num}...")
    base64_image = encode_image_to_base64(image_path)
    
    summary_prompt = """
    Analyze this image/diagram in detail and provide a comprehensive VISUAL summary including:
    
    1. **Document Type**: What type of diagram, chart, or document this is
    2. **Visual Elements**: Main components, shapes, symbols, and visual elements
    3. **Layout & Structure**: How information is organized and arranged
    4. **Diagrams & Flows**: Any process flows, connections, or relationships shown
    5. **Visual Design**: Colors, styles, formatting that convey meaning
    6. **Technical Details**: Any visual specifications, measurements, or technical elements
    7. **Purpose**: Overall function or purpose illustrated by the visual design
    
    Focus on VISUAL elements rather than text content (text will be extracted separately).
    Make the summary detailed and searchable for visual similarity matching.
    """
    
    try:
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
        
    except Exception as e:
        print(f"❌ Error creating summary: {e}")
        return "Error creating visual summary"

def create_combined_content(summary, ocr_text):
    """Combine visual summary and OCR text for optimal searchability"""
    if ocr_text.strip():
        combined_content = f"""
VISUAL ANALYSIS:
{summary}

EXTRACTED TEXT CONTENT:
{ocr_text}

SEARCHABLE KEYWORDS AND CONTENT:
Visual elements: {summary[:300]}
Text content: {ocr_text[:300]}
Combined context: This page contains both visual diagrams/charts and textual information including {ocr_text[:200].replace(chr(10), ' ')}
        """.strip()
    else:
        combined_content = f"""
VISUAL ANALYSIS:
{summary}

EXTRACTED TEXT CONTENT:
No readable text content found in this image.

SEARCHABLE KEYWORDS AND CONTENT:
Visual elements: {summary[:300]}
Content type: This page contains primarily visual/graphical content including diagrams, charts, or images without readable text.
        """.strip()
    
    return combined_content

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

def create_page_summaries(pdf_path, user_id):
    """Convert PDF to images and create detailed summaries with OpenAI OCR and embeddings for a user/case"""
    user_dir = get_user_dir(user_id)
    output_dir = os.path.join(user_dir, "pdf_pages")
    os.makedirs(output_dir, exist_ok=True)

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
            
            print(f"\n🟦 Processing Page {page_num}...")
            
            # Step 1: Get detailed visual summary from OpenAI Vision
            summary = get_detailed_summary(image_path, page_num)
            
            # Step 2: Extract text using OpenAI Vision (OCR)
            ocr_text = extract_text_with_openai(image_path, page_num)
            
            # Step 3: Combine summary and OCR text
            combined_content = create_combined_content(summary, ocr_text)
            
            # Step 4: Create embedding using OpenAI embedding model
            print(f"🔄 Creating OpenAI embedding for combined content...")
            embedding = get_openai_embedding(combined_content)
            
            if embedding is None:
                print(f"❌ Failed to create embedding for page {page_num}, skipping...")
                continue
            
            # Store metadata with page number as key
            page_metadata[page_num] = {
                'page': page_num,
                'image_path': image_path,
                'visual_summary': summary,
                'extracted_text': ocr_text,
                'combined_content': combined_content,
                'has_text': bool(ocr_text.strip()),
                'text_length': len(ocr_text),
                'summary_length': len(summary),
                'embedding_model': 'text-embedding-3-small'
            }
            
            embeddings_list.append(embedding)
            
            print(f"✅ Page {page_num} processing complete")
            print(f"📊 Visual summary: {len(summary)} chars")
            print(f"📝 Extracted text: {len(ocr_text)} chars")
            print(f"🔗 Combined content: {len(combined_content)} chars")
            print(f"🎯 Embedding dimension: {len(embedding)}")
            print(f"Visual preview: {summary[:150]}...")
            if ocr_text:
                print(f"Text preview: {ocr_text[:150]}...")
            print("-" * 80)
        
        if not embeddings_list:
            print("❌ No valid embeddings created")
            return {}, None
        
        # Convert to numpy array
        embeddings = np.array(embeddings_list)
        
        # Create FAISS index with IDs
        print("🔄 Creating FAISS index with ID mapping...")
        index = create_faiss_index_with_ids(embeddings, list(page_metadata.values()))
        
        # Save metadata dictionary
        with open(os.path.join(user_dir, 'page_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(page_metadata, f, indent=2, ensure_ascii=False)
        
        # Save FAISS index
        faiss.write_index(index, os.path.join(user_dir, "page_embeddings.index"))
        
        print(f"💾 Page metadata saved to {os.path.join(user_dir, 'page_metadata.json')}")
        print(f"💾 FAISS index with IDs saved to {os.path.join(user_dir, 'page_embeddings.index')}")
        print(f"📈 Statistics: {sum(1 for p in page_metadata.values() if p['has_text'])}/{len(page_metadata)} pages contain readable text")
        print(f"📈 Total extracted text: {sum(p['text_length'] for p in page_metadata.values())} characters")
        print(f"🎯 Embedding dimension: {embeddings.shape[1]} (OpenAI text-embedding-3-small)")
        
        return page_metadata, index
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return {}, None

def search_relevant_pages(query, page_metadata, index, top_k=2):
    """Search for most relevant pages using FAISS with OpenAI embeddings"""
    print(f"🔍 Searching for pages relevant to: '{query}'")
    
    # Create embedding for the query using OpenAI
    print(f"🔄 Creating OpenAI embedding for query...")
    query_embedding = get_openai_embedding(query)
    
    if query_embedding is None:
        print("❌ Failed to create query embedding")
        return []
    
    # Reshape and normalize query embedding
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Search using FAISS
    scores, ids = index.search(query_embedding, top_k)
    
    print(f"📊 Top {len(ids[0])} relevant pages:")
    relevant_pages = []
    
    for i, page_id in enumerate(ids[0]):
        if page_id != -1 and page_id in page_metadata:  # Valid ID
            page = page_metadata[page_id]
            score = scores[0][i]
            has_text_indicator = "📝" if page.get('has_text', False) else "🖼️"
            text_chars = page.get('text_length', 0)
            print(f"   Page {page['page']} {has_text_indicator} ({text_chars} chars): {score:.3f} similarity")
            relevant_pages.append(page)
    
    return relevant_pages

def answer_query_with_stored_data(query, page_metadata, index, top_k=2):
    """Answer user query using stored summaries and OCR text (fast, efficient)"""
    
    # Find most relevant pages using FAISS
    relevant_pages = search_relevant_pages(query, page_metadata, index, top_k)
    
    if not relevant_pages:
        return "❌ No relevant pages found for your query."
    
    print(f"\n💬 Answering query using stored analysis (no re-scanning needed)...")
    
    results = []
    
    for page in relevant_pages:
        print(f"🔄 Processing answer for Page {page['page']} using stored data...")
        
        # Use stored visual summary and extracted text instead of re-scanning
        enhanced_prompt = f"""
User Query: "{query}"

STORED VISUAL ANALYSIS:
{page.get('visual_summary', '')}

STORED EXTRACTED TEXT:
{page.get('extracted_text', 'No readable text found')}

INSTRUCTIONS:
Based on the user's query and the stored analysis above, provide a detailed answer using:
1. The visual elements and structure described in the stored analysis
2. The extracted text content from the image
3. The relationships and context between visual and textual elements

Provide a comprehensive response that addresses the user's specific question using only the stored information above.
        """
        
        try:
            # Use text-only completion instead of vision (much faster and cheaper)
            response = client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 text model instead of vision
                messages=[
                    {
                        "role": "user",
                        "content": enhanced_prompt
                    }
                ],
                max_tokens=1000
            )
            
            result = {
                'page': page['page'],
                'answer': response.choices[0].message.content,
                'image_path': page['image_path'],
                'has_text': page.get('has_text', False),
                'text_length': page.get('text_length', 0),
                'extracted_text': page.get('extracted_text', '')[:200] + "..." if page.get('extracted_text', '') else "",
                'used_stored_data': True  # Flag to indicate we used stored data
            }
            
            results.append(result)
            print(f"✅ Page {page['page']} Answer Generated from Stored Data")
            
        except Exception as e:
            print(f"❌ Error generating answer for page {page['page']}: {e}")
    
    return results

def answer_query_with_fresh_scan(query, page_metadata, index, top_k=2):
    """Answer user query by re-scanning images (only use when needed)"""
    
    # Find most relevant pages using FAISS
    relevant_pages = search_relevant_pages(query, page_metadata, index, top_k)
    
    if not relevant_pages:
        return "❌ No relevant pages found for your query."
    
    print(f"\n💬 Re-scanning {len(relevant_pages)} pages with OpenAI Vision...")
    print("⚠️ This will use additional API calls and take longer.")
    
    results = []
    
    for page in relevant_pages:
        print(f"🔄 Re-scanning Page {page['page']}...")
        
        base64_image = encode_image_to_base64(page['image_path'])
        
        enhanced_prompt = f"""
User Query: "{query}"

STORED ANALYSIS (for context):
Visual Summary: {page.get('visual_summary', '')[:200]}...
Extracted Text: {page.get('extracted_text', 'No readable text found')[:200]}...

INSTRUCTIONS:
Please analyze this image fresh and provide a detailed answer to the user's query.
Use both what you can see in the image and any context from the stored analysis above.
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
                'image_path': page['image_path'],
                'has_text': page.get('has_text', False),
                'text_length': page.get('text_length', 0),
                'extracted_text': page.get('extracted_text', '')[:200] + "..." if page.get('extracted_text', '') else "",
                'used_stored_data': False  # Flag to indicate we re-scanned
            }
            
            results.append(result)
            print(f"✅ Page {page['page']} Fresh Analysis Complete")
            
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
        has_text = "📝" if metadata.get('has_text', False) else "🖼️"
        text_length = metadata.get('text_length', 0)
        embedding_model = metadata.get('embedding_model', 'unknown')
        print(f"   Page {page_id} {has_text} ({text_length} chars) [{embedding_model}]: {metadata.get('visual_summary', '')[:100]}...")

# Main execution
def main():
    user_id = input("Enter user/case ID: ").strip()
    pdf_path = input("Enter path to PDF: ").strip()
    
    user_dir = get_user_dir(user_id)
    metadata_path = os.path.join(user_dir, 'page_metadata.json')
    index_path = os.path.join(user_dir, 'page_embeddings.index')

    # Check if data already exists for this user/case
    if os.path.exists(metadata_path) and os.path.exists(index_path):
        print("📁 Loading existing page data and FAISS index for this user/case...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            page_metadata = json.load(f)
        page_metadata = {int(k): v for k, v in page_metadata.items()}
        index = faiss.read_index(index_path)
    else:
        print("📄 Creating page summaries and FAISS index with OpenAI pipeline for the first time...")
        if os.path.exists(pdf_path):
            page_metadata, index = create_page_summaries(pdf_path, user_id)
        else:
            print(f"❌ PDF file not found: {pdf_path}")
            return

    if not page_metadata or index is None:
        print("❌ No page data available")
        return

    print(f"\n✅ Ready! {len(page_metadata)} pages indexed with FAISS and ready for queries.")
    print(f"📊 FAISS index size: {index.ntotal} vectors")

    # Interactive query loop with improved options
    while True:
        print("\n" + "="*60)
        print("Commands:")
        print("  'list' - see all pages")
        print("  'page N' - get specific page details")
        print("  'fresh [question]' - answer using fresh image scan (slower, more API calls)")
        print("  '[question]' - answer using stored data (fast, recommended)")
        print("="*60)
        
        user_input = input("🤔 Enter your command or question (or 'quit' to exit): ")
        
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
                    print(f"Visual Summary: {page.get('visual_summary', '')}")
                    if page.get('extracted_text'):
                        print(f"\n📝 Extracted Text: {page['extracted_text']}")
                    else:
                        print(f"\n📝 Extracted Text: No readable text found")
                    print(f"\n🎯 Embedding Model: {page.get('embedding_model', 'unknown')}")
                else:
                    print(f"❌ Page {page_num} not found")
            except (ValueError, IndexError):
                print("❌ Invalid page command. Use 'page N' where N is a number")
            continue
        
        if user_input.lower().startswith('fresh '):
            # Extract the query after 'fresh '
            query = user_input[6:].strip()
            if query:
                print("🔄 Using fresh image scanning mode...")
                results = answer_query_with_fresh_scan(query, page_metadata, index, top_k=2)
            else:
                print("❌ Please provide a question after 'fresh'. Example: 'fresh explain the diagram'")
                continue
        else:
            # Regular query using stored data
            query = user_input.strip()
            if query:
                print("⚡ Using stored data mode with OpenAI embeddings (fast, recommended)...")
                results = answer_query_with_stored_data(query, page_metadata, index, top_k=2)
            else:
                continue
        
        # Display results
        print(f"\n🧠 Answer based on most relevant pages:")
        print("="*60)
        
        if isinstance(results, str):  # Error message
            print(results)
        else:
            for result in results:
                text_indicator = "📝" if result['has_text'] else "🖼️"
                data_source = "💾 (stored)" if result.get('used_stored_data', True) else "🔄 (fresh scan)"
                print(f"\n📄 From Page {result['page']} {text_indicator} ({result['text_length']} chars) {data_source}:")
                print(result['answer'])
                if result['extracted_text']:
                    print(f"\n📝 Key extracted text: {result['extracted_text']}")
                print("-" * 40)

if __name__ == "__main__":
    main()