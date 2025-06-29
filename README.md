# 🧠 BiztelAI — Chat Transcript Analysis API

BiztelAI is a sophisticated FastAPI application designed for analyzing chat transcripts between agents discussing Washington Post articles. The system provides comprehensive analysis including sentiment evaluation, text summarization, and detailed agent-level insights through a well-structured REST API.

## 🌟 Core Features

### Data Processing
- **Advanced Text Preprocessing**
  - Tokenization and lemmatization using spaCy
  - Configurable stopword removal
  - Custom token preservation
  - URL and email cleaning
  - Special character handling

### Analysis Capabilities
- **Sentiment Analysis**
  - Agent-specific sentiment scoring
  - Configurable sentiment thresholds
  - TextBlob-based polarity analysis

### Summarization
- **LLM-powered Summarization**
  - BART-based text summarization (facebook/bart-large-cnn)
  - GPU acceleration with CUDA support
  - Fallback to lighter models for resource constraints
  - Configurable summary length

### Technical Features
- **Robust Data Handling**
  - JSON dataset processing
  - Pandas DataFrame operations
  - Null value management
  - Type conversion and validation

- **Performance Optimization**
  - Async endpoint support
  - Performance monitoring decorators
  - Efficient batch processing

- **Developer Experience**
  - Comprehensive logging system
  - Detailed error handling
  - Performance metrics tracking
  - Interactive API documentation

## 📁 Project Structure

```
biztelai/
├── app/
│   ├── api/             # FastAPI routes and request/response schemas
│   │   ├── endpoints.py  # API route definitions
│   │   └── schemas.py    # Pydantic models
│   ├── core/            # Core utilities
│   │   └── performance.py # Performance monitoring
│   ├── services/        # Business logic implementation
│   │   ├── analysis.py   # Chat analysis service
│   │   ├── data_cleaner.py # Data cleaning utilities
│   │   ├── data_loader.py  # Dataset loading service
│   │   ├── summarizer.py   # Text summarization service
│   │   └── transformer.py  # Text preprocessing service
│   └── utils/           # Helper utilities
│       ├── helper.py     # General helper functions
│       └── logger.py     # Logging configuration
├── data/               # Dataset storage
├── notebooks/         # Jupyter notebooks for analysis
├── main.py            # Application entry point
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- CUDA-compatible GPU (optional, for accelerated summarization)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Sayantang8/biztelai.git
cd biztelai
```

2. Install dependencies:
```bash
# CPU-only installation
pip install -r requirements.txt

# GPU-accelerated installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

3. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

### Running the API
```bash
uvicorn app.main:app --reload
```

Access the interactive API documentation at: http://localhost:8000/docs

## 📚 API Documentation

### GET /health
- Health check endpoint
- Returns API status and configuration

### POST /preprocess
- Text preprocessing endpoint
- Handles tokenization, cleaning, and normalization
- Request body:
```json
{
  "text": "Raw text to process"
}
```

### POST /analyze-chat
- Chat analysis endpoint
- Provides sentiment analysis and message statistics
- Request body:
```json
{
  "messages": [
    {"agent_id": "agent1", "message": "Message content"},
    {"agent_id": "agent2", "message": "Response content"}
  ]
}
```

### GET /summary
- Dataset summary endpoint
- Returns dataset statistics and structure

## 🐳 Docker Support

```bash
# Build the container
docker build -t biztelai-api .

# Run the container
docker run -p 8000:8000 biztelai-api
```

## 🔄 Development Workflow

1. Data Loading (`DataLoader`)
   - Handles JSON dataset ingestion
   - Provides dataset structure analysis
   - Implements error handling and validation

2. Data Cleaning (`DataCleaner`)
   - Manages data type conversion
   - Handles missing values
   - Removes duplicates

3. Text Processing (`TextPreprocessor`)
   - Implements NLP pipeline
   - Configurable preprocessing steps
   - Logging of transformation steps

4. Analysis (`ChatAnalyzer`)
   - Performs sentiment analysis
   - Computes message statistics
   - Generates insights

5. Summarization (`SimpleChatSummarizer`)
   - Generates concise summaries
   - Handles long text truncation
   - Manages model loading and inference

## 🛠️ Future Improvements

- [ ] Implement OAuth2 authentication
- [ ] Add article classification features
- [ ] Create frontend dashboard
- [ ] Add batch processing capabilities
- [ ] Implement caching layer
- [ ] Add comprehensive test suite
- [ ] Implement rate limiting
- [ ] Add monitoring and analytics

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

