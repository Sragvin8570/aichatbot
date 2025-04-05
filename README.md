# Crop Disease Detection & Advisory System

A comprehensive system for crop disease detection using deep learning and providing AI-powered agricultural advice through a Retrieval-Augmented Generation (RAG) chatbot.

## Features

- 🌱 **Image-based Disease Detection**: 
  - 38+ plant disease classes supported
  - CNN model with 85-90% accuracy
  - Confidence scoring for predictions
- 🤖 **AI-Powered Advisory**:
  - RAG system with web-scraped knowledge
  - Context-aware responses
  - Multi-source agricultural information
- 🔍 **Content Retrieval**:
  - Web scraping of agricultural resources
  - Pinecone vector database integration
  - OpenAI embeddings for semantic search
- 🚀 **API Endpoints**:
  - Unified chat endpoint handling text + images
  - JSON responses with structured data
  - Error handling and validation

## Technologies Used

- **Machine Learning**: 
  - TensorFlow/Keras (CNN model)
  - NumPy/PIL (image processing)
- **Natural Language Processing**:
  - OpenAI Embeddings & GPT-3.5
  - LangChain text processing
- **Backend**:
  - Flask REST API
  - Pinecone vector database
  - BeautifulSoup (web scraping)
- **Environment**:
  - Python 3.9+
  - dotenv configuration
  - Serverless Pinecone index

## Installation

1. **Clone Repository**:
   ```bash
   git clone https://github.com/yourusername/crop-advisory-system.git
   cd crop-advisory-system

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Environment Setup**:
   - Create `.env` file:
     ```ini
     OPENAI_API_KEY=your_openai_key
     PINECONE_API_KEY=your_pinecone_key
     ```
   - Place trained model `trained_model.keras` in project root

## Configuration

1. **API Keys**:
   - Get [OpenAI API Key](https://platform.openai.com/api-keys)
   - Get [Pinecone API Key](https://app.pinecone.io/)

2. **Pinecone Index**:
   - Ensure index `crop-rag-openai` exists in `us-east-1` AWS region
   - Dimension: 1536 (matches OpenAI embeddings)

## Usage

1. **Run Flask Application**:
   ```bash
   python app.py
   ```

2. **API Endpoints**:
   - **POST** `/chat` - Main advisory endpoint
     - Accepts both text and image inputs
     - Form-data parameters:
       - `text`: Query text (optional)
       - `image`: Image file (optional)

   **Example Request**:
   ```bash
   curl -X POST -F "text=How treat apple scab?" -F "image=@disease_leaf.jpg" http://localhost:5000/chat
   ```

3. **Example Response**:
   ```json
   {
     "status": "success",
     "message": "Apple scab can be treated with...",
     "detected_disease": "Apple___Apple_scab"
   }
   ```

## Data Processing

To update agricultural knowledge base:
```python
# Add your information sources
info_sources = [
    "https://agritech.tnau.ac.in/crop_protection/crop_diseases_postharvest_apple_4.html",
    # Add other agricultural URLs
]

# Run once to populate Pinecone
process_and_store_data(info_sources)
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## License

MIT License - See [LICENSE](LICENSE) for details

## Acknowledgments

- PlantVillage dataset for disease classification
- OpenAI for NLP models
- Pinecone for vector database infrastructure
- TNAU Agritech Portal for agricultural content

---

**Disclaimer**: This system provides educational suggestions only. Always consult with agricultural professionals for critical decisions.
```

This README provides:
- Clear setup and configuration instructions
- API usage examples
- System architecture overview
- Contribution guidelines
- Proper acknowledgments and licensing

Adjust the repository URLs and license file as needed for your specific implementation.
