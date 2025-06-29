from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import logging

# Import services (these would be the actual service modules)
from ..services.data_loader import DataLoader
from ..services.data_cleaner import DataCleaner
from ..services.transformer import TextPreprocessor
from ..services.analysis import ChatAnalyzer
from ..services.summarizer import SimpleChatSummarizer

# Import Pydantic models from a separate schemas module
# Note: Remove the circular import - this should import from a separate schemas.py file
from pydantic import BaseModel
from enum import Enum

# Define the schemas directly here to avoid circular imports
class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class PreprocessRequest(BaseModel):
    text: str

class PreprocessResponse(BaseModel):
    status: str
    original_text: str
    cleaned_text: str
    character_count_original: str
    character_count_cleaned: str

class ChatMessage(BaseModel):
    agent_id: str
    message: str

class ChatTranscript(BaseModel):
    messages: List[ChatMessage]

class ProcessedChatResponse(BaseModel):
    agent1_msg_count: int
    agent2_msg_count: int
    agent1_sentiment: SentimentType
    agent2_sentiment: SentimentType
    summary: str
    article_link: str

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter(
    prefix="/api",
    tags=["chat-analysis"],
    responses={404: {"description": "Not found"}},
)

# Dataset path constant
DATASET_PATH = "data/BiztelAI_DS_Dataset_V1.json"


@router.get("/summary", response_model=Dict[str, Any])
async def get_dataset_summary():
    try:
        logger.info(f"Loading dataset from {DATASET_PATH}")
        loader = DataLoader(DATASET_PATH)
        df = loader.load_data()
        if df.empty:
            raise HTTPException(status_code=404, detail="Dataset not found or empty")

        logger.info("Cleaning dataset")
        cleaner = DataCleaner(df)
        cleaner.config['remove_duplicates'] = False  # Temporarily disable due to unhashable lists
        cleaned_df = cleaner.clean()

        # Ensure all column names are strings
        cleaned_df.columns = [str(col) for col in cleaned_df.columns]

        # Safely convert null_counts keys to strings
        null_counts = {}
        for col, val in cleaned_df.isnull().sum().items():
            try:
                null_counts[str(col)] = int(val)
            except Exception as e:
                logger.warning(f"Could not convert column '{col}' null count due to: {e}")

        dataset_info = {
            "shape": cleaned_df.shape,
            "columns": [str(col) for col in cleaned_df.columns],
            "null_counts": null_counts,
            "total_entries": len(cleaned_df),
            "sample_keys": [str(col) for col in cleaned_df.columns[:5]]
        }

        logger.info("Dataset summary generated successfully")
        return {
            "status": "success",
            "dataset_path": DATASET_PATH,
            "shape": dataset_info.get("shape", {}),
            "column_names": dataset_info.get("columns", []),
            "null_counts": dataset_info.get("null_counts", {}),
            "total_entries": dataset_info.get("total_entries", 0),
            "sample_keys": dataset_info.get("sample_keys", [])
        }

    except FileNotFoundError:
        logger.error(f"Dataset file not found: {DATASET_PATH}")
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {DATASET_PATH}")
    except Exception as e:
        logger.error(f"Error processing dataset summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/preprocess", response_model=PreprocessResponse)
def preprocess_text(request: PreprocessRequest):
    try:
        logger.info("Starting text preprocessing")

        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text field cannot be empty or whitespace only")

        preprocessor = TextPreprocessor()
        cleaned_text = preprocessor.preprocess_text(request.text)

        logger.info("Text preprocessing completed successfully")
        # Explicitly convert to strings to avoid validation errors
        original_count = str(len(request.text))
        cleaned_count = str(len(cleaned_text))
        
        logger.info(f"Character counts - Original: {original_count}, Cleaned: {cleaned_count}")
        
        return PreprocessResponse(
            status="success",
            original_text=request.text,
            cleaned_text=cleaned_text,
            character_count_original=original_count,
            character_count_cleaned=cleaned_count
        )

    except Exception as e:
        logger.error(f"Error during text preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text preprocessing failed: {str(e)}")


@router.post("/analyze-chat", response_model=ProcessedChatResponse)
async def analyze_chat_transcript(transcript: ChatTranscript):
    try:
        logger.info(f"Starting chat analysis for {len(transcript.messages)} messages")
        if not transcript.messages:
            raise HTTPException(status_code=400, detail="Chat transcript cannot be empty")

        agent_messages = {}
        for message in transcript.messages:
            if message.agent_id not in agent_messages:
                agent_messages[message.agent_id] = []
            agent_messages[message.agent_id].append(message.message)

        agent_ids = list(agent_messages.keys())
        if len(agent_ids) < 2:
            agent1_id = agent_ids[0] if agent_ids else "agent_1"
            agent2_id = "agent_2" if agent1_id != "agent_2" else "agent_1"
        else:
            agent1_id, agent2_id = agent_ids[0], agent_ids[1]

        agent1_messages = agent_messages.get(agent1_id, [])
        agent2_messages = agent_messages.get(agent2_id, [])

        agent1_msg_count = len(agent1_messages)
        agent2_msg_count = len(agent2_messages)

        logger.info(f"Message counts - Agent1: {agent1_msg_count}, Agent2: {agent2_msg_count}")

        analyzer = ChatAnalyzer()
        agent1_scores = [analyzer.analyze_sentiment(msg) for msg in agent1_messages] if agent1_messages else [0]
        agent2_scores = [analyzer.analyze_sentiment(msg) for msg in agent2_messages] if agent2_messages else [0]

        agent1_sentiment = analyzer.classify_sentiment(sum(agent1_scores) / len(agent1_scores))
        agent2_sentiment = analyzer.classify_sentiment(sum(agent2_scores) / len(agent2_scores))

        logger.info(f"Sentiment analysis completed - Agent1: {agent1_sentiment}, Agent2: {agent2_sentiment}")

        all_messages = [msg.message for msg in transcript.messages]
        conversation_text = " ".join(all_messages)
        summarizer = SimpleChatSummarizer()
        summaries = summarizer.batch_summarize([conversation_text])
        summary = summaries[0] if summaries else "No summary available."
        article_link = summarizer.generate_article_link(summary, str(hash(conversation_text) % 10000))

        return ProcessedChatResponse(
            agent1_msg_count=agent1_msg_count,
            agent2_msg_count=agent2_msg_count,
            agent1_sentiment=agent1_sentiment,
            agent2_sentiment=agent2_sentiment,
            summary=summary,
            article_link=article_link
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during chat analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat analysis failed: {str(e)}")


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Chat analysis API is running",
        "dataset_path": DATASET_PATH
    }


@router.get("/sentiment-types")
def get_sentiment_types():
    return {
        "status": "success",
        "supported_sentiments": [sentiment.value for sentiment in SentimentType]
    }