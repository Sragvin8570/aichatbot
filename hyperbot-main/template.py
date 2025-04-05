import os
import openai
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
# Load environment variables
load_dotenv()
# Initialize APIs
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# Configuration
INDEX_NAME = "crop-rag-openai"
CHUNK_SIZE = 1000  # Optimal for OpenAI embeddings
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"
import tensorflow as tf
import numpy as np
from PIL import Image
# Class names (replace with your actual class labels)
CLASS_NAMES = [
#  'Apple___Apple_scab',
#  'Apple___Black_rot',
#  'Apple___Cedar_apple_rust',
#  'Apple___healthy',
#  'Blueberry___healthy',
#  'Cherry_(including_sour)___Powdery_mildew',
#  'Cherry_(including_sour)___healthy',
#  'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
#  'Corn_(maize)___Common_rust_',
#  'Corn_(maize)___Northern_Leaf_Blight',
#  'Corn_(maize)___healthy',
#  'Grape___Black_rot',
#  'Grape___Esca_(Black_Measles)',
#  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#  'Grape___healthy',
#  'Orange___Haunglongbing_(Citrus_greening)',
#  'Peach___Bacterial_spot',
#  'Peach___healthy',
#  'Pepper,_bell___Bacterial_spot',
#  'Pepper,_bell___healthy',
#  'Potato___Early_blight',
#  'Potato___Late_blight',
#  'Potato___healthy',
#  'Raspberry___healthy',
#  'Soybean___healthy',
#  'Squash___Powdery_mildew',
#  'Strawberry___Leaf_scorch',
#  'Strawberry___healthy',
#  'Tomato___Bacterial_spot',
#  'Tomato___Early_blight',
#  'Tomato___Late_blight',
#  'Tomato___Leaf_Mold',
#  'Tomato___Septoria_leaf_spot',
#  'Tomato___Spider_mites Two-spotted_spider_mite',
#  'Tomato___Target_Spot',
#  'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#  'Tomato___Tomato_mosaic_virus',
#  'Tomato___healthy'
 'Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust'
#                 # 'Apple___Apple_scab',
#                 # 'Peach___Bacterial_spot',
#                 # 'Potato___Early_blight',
#                 # 'Potato___Late_blight',
#                 # 'Tomato___Early_blight',
#                 # 'Tomato___Late_blight',
#                 # 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',  # Leaf Curl
#                 # 'Tomato___Septoria_leaf_spot',
#                 # 'Tomato___Tomato_mosaic_virus',  # Mosaic
#                 # 'Corn_(maize)___Common_rust_'   
#                 # ... add all your disease classes in order
]

def predict_disease(image_path):
    model = tf.keras.models.load_model(r'.\trained_model.keras')
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    
    # Resize to match model's expected input
    img = img.resize((224, 224))  # adjust size if your model uses different dimensions
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return CLASS_NAMES[predicted_class], confidence
import random

# def predict_disease2(image_path):
#     disease = random.choice(CLASS_NAMES)
#     # disease = 'Apple___Apple_scab'
#     confidence = round(random.uniform(0.85, 0.91), 2) 
    
#     return disease, confidence

# Example usage  # Output: ('Tomato___Early_blight', 0.87)
    
# Example usage

# Initialize components
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

def initialize_pinecone_index():
    """Create or connect to Pinecone index with ServerlessSpec"""
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # OpenAI embedding dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Created new index: {INDEX_NAME}")
    
    # Wait for index to be ready
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)
    
    return pc.Index(INDEX_NAME)

def scrape_crop_content(url):
    """Scrape crop-related content from websites"""
    try:
        headers = {'User-Agent': 'cropdetection/1.0 (+http://yourdomain.com/bot)'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
            element.decompose()
            
        # Extract main content
        selectors = ['main', 'article', '[role="main"]', 'body']
        main_content = None
        for selector in selectors:
            if soup.select(selector):
                main_content = soup.select(selector)[0]
                break
                
        if not main_content:
            raise ValueError("Could not find main content")
            
        # Extract structured text
        text_parts = []
        for tag in main_content.find_all(['p', 'h1', 'h2', 'h3', 'li']):
            if tag.name.startswith('h'):
                text_parts.append(f"\n{tag.get_text().upper()}\n")
            else:
                text_parts.append(tag.get_text())
                
        return ' '.join(text_parts).replace('\n', ' ').strip()
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return ""

def generate_embeddings(texts):
    """Generate embeddings using OpenAI's API"""
    embeddings = []
    try:
        response = openai.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        return []

def process_and_store_data(urls):
    """Process URLs and store in Pinecone"""
    index = initialize_pinecone_index()
    
    for url in urls:
        print(f"Processing {url}")
        raw_text = scrape_crop_content(url)
        if not raw_text or len(raw_text) < 100:
            print(f"Skipping {url} - insufficient content")
            continue
            
        chunks = text_splitter.split_text(raw_text)
        embeddings = generate_embeddings(chunks)
        
        # Prepare vectors with validation
        vectors = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding and len(embedding) == 1536:  # Validate OpenAI embedding size
                vectors.append({
                    "id": f"doc-{hash(url)}-{idx}",
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "source": url,
                        "chunk_num": idx
                    }
                })
        
        # Batch upsert
        for i in range(0, len(vectors), 100):
            batch = vectors[i:i+100]
            try:
                index.upsert(vectors=batch)
                print(f"Inserted batch {i//100 + 1}")
            except Exception as e:
                print(f"Failed to upsert batch: {str(e)}")
def expand_query(query):
    """Generate alternative queries using OpenAI"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate 1 alternative phrasings of this question for better search results. Return only the variants separated by newlines."},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=150
        )
        variants = [query] + [v.strip() for v in response.choices[0].message.content.split('\n') if v.strip()]
        return list(set(variants))[:4]  # Ensure unique + limit to 4
    except Exception as e:
        print(f"Query expansion failed: {str(e)}")
        return [query]

def crop_chat(query):
    """Enhanced RAG query with OpenAI GPT"""
    index = pc.Index(INDEX_NAME)
    
    try:
        # Generate query embedding
        # expanded_queries = expand_query(query)
        response = openai.embeddings.create(
            input=[query],
            model=EMBEDDING_MODEL
        )
        query_embedding = response.data[0].embedding
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter={"source": {"$exists": True}}
        )
        # embeddings = []
        # for q in expanded_queries:
        #     response = openai.embeddings.create(
        #         input=[q],
        #         model=EMBEDDING_MODEL
        #     )
        #     embeddings.append(response.data[0].embedding)
        
        # # Average the embeddings
        # avg_embedding = [sum(values)/len(values) for values in zip(*embeddings)]
        
        # Search with averaged embedding
        # results = index.query(
        #     vector=avg_embedding,
        #     top_k=5,
        #     include_metadata=True,
        #     filter={
        #         "source": {"$exists": True}
        #     }
        # )
        if not results.matches:
            return "No relevant information found. Please ask another question about crop."
        
        # Build context
        context = []
        for match in results.matches:
            context.append(
                f"SOURCE: {match.metadata['source']}\n"
                f"CONTENT: {match.metadata['text']}\n"
                f"RELEVANCE: {match.score:.2f}"
            )
        context_str = "\n\n".join(context)
        
        # Generate response
        response = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a knowledgeable crop expert assistant. Use the context to answer questions in a friendly, engaging manner."},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing your request: {str(e)}"

# if __name__ == "__main__":
#     # crop information sources
#     # info_sources = [
#     #     "https://agritech.tnau.ac.in/crop_protection/crop_diseases_postharvest_apple_4.html",
#     #     "https://agritech.tnau.ac.in/crop_protection/peach_diseases1.html",
#     #     "https://agritech.tnau.ac.in/crop_protection/crop_prot_crop%20diseases_veg_potato_1.html",
#     #     "https://agritech.tnau.ac.in/crop_protection/crop_prot_crop%20diseases_veg_potato_2.html",
#     #     "https://agritech.tnau.ac.in/crop_protection/tomato_diseases_2.html",
#     #     "https://agritech.tnau.ac.in/crop_protection/tomato_diseases_8.html",
#     #     " https://agritech.tnau.ac.in/crop_protection/tomato_diseases_6.html",
#     #     "https://agritech.tnau.ac.in/crop_protection/tomato_diseases_4.html",
#     #     "https://agritech.tnau.ac.in/crop_protection/tomato_diseases_3.html",
#     #     "https://extension.umn.edu/corn-pest-management/common-rust-corn"
#     # ]
    
#     # # Process data (run once)
#     # process_and_store_data(info_sources)
#     # Chat interface
#     print("crop Explorer Chatbot (OpenAI)")
#     print("Type 'exit' to quit\n")
#     while True:
#         try:
#             query = input("User: ")
#             if query.lower() in ['exit', 'quit']:
#                 break
#             response = crop_chat(query)
#             print(f"\nAssistant: {response}\n")
#         except KeyboardInterrupt:
#             break
#         except Exception as e:
#             print(f"Error: {str(e)}")
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os

app = Flask(__name__)
CORS(app)  # Enable CORS if needed
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        # Handle form data
        text_input = request.form.get('text', '')
        image_file = request.files.get('image')
        
        # Initialize query
        final_query = text_input
        
        # Process image if provided
        disease = None
        if image_file:
            # Validate image
            if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                return jsonify({
                    "status": "error",
                    "message": "Invalid image format. Only PNG, JPG, JPEG allowed."
                }), 400
            
            # Save temporary image
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(temp_path)
            
            try:
                # Get disease prediction
                disease, confidence = predict_disease2(temp_path)
                final_query += f" | Detected Disease: {disease} (Confidence: {confidence*100:.1f}%)"
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": f"Image processing failed: {str(e)}"
                }), 500
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Process query through RAG pipeline
        if not final_query.strip():
            return jsonify({
                "status": "error",
                "message": "No input provided. Please send text or image."
            }), 400
            
        try:
            response = crop_chat(final_query)
            return jsonify({
                "status": "success",
                "message": response,
                "detected_disease": disease if disease else None
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Query processing failed: {str(e)}"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)