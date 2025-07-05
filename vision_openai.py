import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import base64
from pdf2image import convert_from_path
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

def create_page_summaries(pdf_path, output_dir="pdf_pages", summaries_file="page_summaries.json"):
    """Convert PDF to images and create detailed summaries"""
    print(f"📄 Processing PDF: {pdf_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)
        print(f"📄 Found {len(images)} pages in PDF")
        
        page_data = []
        
        for i, image in enumerate(images):
            page_num = i + 1
            image_path = os.path.join(output_dir, f"page_{page_num}.png")
            
            # Save page as image
            image.save(image_path, "PNG")
            
            # Get detailed summary
            summary = get_detailed_summary(image_path, page_num)
            
            # Create embedding for the summary
            embedding = embedding_model.encode(summary).tolist()
            
            page_info = {
                'page': page_num,
                'image_path': image_path,
                'summary': summary,
                'embedding': embedding
            }
            
            page_data.append(page_info)
            
            print(f"✅ Page {page_num} summary created")
            print(f"Summary: {summary[:200]}...\n")
            print("-" * 80)
        
        # Save all summaries and embeddings
        with open(summaries_file, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Page summaries saved to {summaries_file}")
        return page_data
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return []

def load_page_summaries(summaries_file="page_summaries.json"):
    """Load existing page summaries"""
    try:
        with open(summaries_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Summaries file not found: {summaries_file}")
        return []

def search_relevant_pages(query, page_data, top_k=2):
    """Search for most relevant pages based on query"""
    print(f"🔍 Searching for pages relevant to: '{query}'")
    
    # Create embedding for the query
    query_embedding = embedding_model.encode(query)
    
    # Calculate similarities
    similarities = []
    for page in page_data:
        page_embedding = np.array(page['embedding'])
        similarity = cosine_similarity([query_embedding], [page_embedding])[0][0]
        similarities.append((page, similarity))
    
    # Sort by similarity and get top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_pages = similarities[:top_k]
    
    print(f"📊 Top {len(top_pages)} relevant pages:")
    for page, score in top_pages:
        print(f"   Page {page['page']}: {score:.3f} similarity")
    
    return [page for page, score in top_pages]

def answer_query_with_vision(query, page_data, top_k=2):
    """Answer user query by finding relevant pages and using OpenAI Vision"""
    
    # Find most relevant pages
    relevant_pages = search_relevant_pages(query, page_data, top_k)
    
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
    
    return results

# Main execution
def main():
    pdf_path = "pdf\Schematic Diagram of CCPP.pdf"
    summaries_file = "page_summaries.json"
    
    # Check if summaries already exist
    if os.path.exists(summaries_file):
        print("📁 Loading existing page summaries...")
        page_data = load_page_summaries(summaries_file)
    else:
        print("📄 Creating page summaries for the first time...")
        if os.path.exists(pdf_path):
            page_data = create_page_summaries(pdf_path, summaries_file=summaries_file)
        else:
            print(f"❌ PDF file not found: {pdf_path}")
            return
    
    if not page_data:
        print("❌ No page data available")
        return
    
    print(f"\n✅ Ready! {len(page_data)} pages indexed and ready for queries.")
    
    # Interactive query loop
    while True:
        print("\n" + "="*60)
        user_query = input("🤔 Enter your question (or 'quit' to exit): ")
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_query.strip():
            results = answer_query_with_vision(user_query, page_data, top_k=2)
            
            print(f"\n🧠 Answer based on most relevant pages:")
            print("="*60)
            
            for result in results:
                print(f"\n📄 From Page {result['page']}:")
                print(result['answer'])
                print("-" * 40)

if __name__ == "__main__":
    main()