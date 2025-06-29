from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import logging

# Import services (these would be the actual service modules)
from ..services.data_loader import DataLoader
from ..services.data_cleaner import DataCleaner
from ..services.transformer import TextPreprocessor
from ..services.analysis import ChatAnalyzer
from ..services.summarizer import SimpleChatSummarizer

# Import Pydantic models
from .schemas import (
    PreprocessRequest, 
    ChatTranscript, 
    ProcessedChatResponse,
    ChatMessage
)

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
        cleaned_df = cleaner.clean()

        # Flatten all columns to strings for safety
        def stringify(col):
            if isinstance(col, (list, tuple)):
                return ".".join(map(str, col))
            return str(col)

        cleaned_df.columns = [stringify(col) for col in cleaned_df.columns]
        logger.debug(f"Flattened column names: {cleaned_df.columns.tolist()[:10]}")

        # Compute null counts with string keys only
        null_counts_series = cleaned_df.isnull().sum()
        null_counts = {}
        for k, v in null_counts_series.items():
            try:
                skey = stringify(k)
                null_counts[skey] = int(v)
            except Exception as err:
                logger.warning(f"Skipping column {k} due to error: {err}")

        logger.info("Dataset summary generated successfully")

        return {
            "status": "success",
            "dataset_path": DATASET_PATH,
            "shape": cleaned_df.shape,
            "column_names": list(cleaned_df.columns),
            "null_counts": null_counts,
            "total_entries": len(cleaned_df),
            "sample_keys": list(cleaned_df.columns[:5]),
        }

    except FileNotFoundError:
        logger.error(f"Dataset file not found: {DATASET_PATH}")
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {DATASET_PATH}")

    except Exception as e:
        logger.exception(f"Error processing dataset summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/preprocess", response_model=Dict[str, str])
def preprocess_text(request: PreprocessRequest):
    """
    Preprocess raw text using the transformer service.
    """
    try:
        logger.info("Starting text preprocessing")
        
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text field cannot be empty or whitespace only")
        
        # ✅ Instantiate preprocessor and use method
        preprocessor = TextPreprocessor()
        cleaned_text = preprocessor.preprocess_text(request.text)
        
        logger.info("Text preprocessing completed successfully")
        
        # ✅ Convert character counts to strings to match response model
        return {
            "status": "success",
            "original_text": request.text,
            "cleaned_text": cleaned_text,
            "character_count_original": str(len(request.text)),  # Convert to string
            "character_count_cleaned": str(len(cleaned_text))    # Convert to string
        }
        
    except Exception as e:
        logger.error(f"Error during text preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text preprocessing failed: {str(e)}")


@router.post("/", response_model=ProcessedChatResponse)
async def analyze_chat_transcript(transcript: ChatTranscript):
    """
    Analyze a chat transcript to extract insights and generate summary.
    
    Args:
        transcript: ChatTranscript containing list of chat messages
        
    Returns:
        ProcessedChatResponse with analysis results
    """
    try:
        logger.info(f"Starting chat analysis for {len(transcript.messages)} messages")
        
        if not transcript.messages:
            raise HTTPException(status_code=400, detail="Chat transcript cannot be empty")
        
        # Extract messages by agent
        agent_messages = {}
        for message in transcript.messages:
            if message.agent_id not in agent_messages:
                agent_messages[message.agent_id] = []
            agent_messages[message.agent_id].append(message.message)
        
        # Ensure we have at least 2 agents (agent1 and agent2)
        agent_ids = list(agent_messages.keys())
        if len(agent_ids) < 2:
            # If only one agent, pad with empty data for the second agent
            agent1_id = agent_ids[0] if agent_ids else "agent_1"
            agent2_id = "agent_2" if agent1_id != "agent_2" else "agent_1"
        else:
            agent1_id, agent2_id = agent_ids[0], agent_ids[1]
        
        # Get message counts
        agent1_messages = agent_messages.get(agent1_id, [])
        agent2_messages = agent_messages.get(agent2_id, [])
        
        agent1_msg_count = len(agent1_messages)
        agent2_msg_count = len(agent2_messages)
        
        logger.info(f"Message counts - Agent1: {agent1_msg_count}, Agent2: {agent2_msg_count}")
        
        # Analyze sentiments using the analysis service
        analyzer = ChatAnalyzer()
        agent1_sentiment_scores = [analyzer.analyze_sentiment(msg) for msg in agent1_messages] if agent1_messages else [0]
        agent2_sentiment_scores = [analyzer.analyze_sentiment(msg) for msg in agent2_messages] if agent2_messages else [0]
        
        # Calculate average sentiment and classify
        agent1_avg_sentiment = sum(agent1_sentiment_scores) / len(agent1_sentiment_scores) if agent1_sentiment_scores else 0
        agent2_avg_sentiment = sum(agent2_sentiment_scores) / len(agent2_sentiment_scores) if agent2_sentiment_scores else 0
        
        agent1_sentiment = analyzer.classify_sentiment(agent1_avg_sentiment).lower()
        agent2_sentiment = analyzer.classify_sentiment(agent2_avg_sentiment).lower()

        
        logger.info(f"Sentiment analysis completed - Agent1: {agent1_sentiment}, Agent2: {agent2_sentiment}")
        
        # Generate summary and guess article link using summarizer service
        all_messages = [msg.message for msg in transcript.messages]
        conversation_text = " ".join(all_messages)
        
        # Use the SimpleChatSummarizer for summarization
        summarizer_instance = SimpleChatSummarizer()
        
        # Generate summary using batch_summarize method
        summaries = summarizer_instance.batch_summarize([conversation_text])
        summary = summaries[0] if summaries else "No summary available."
        
        # Generate article link
        conversation_id = hash(conversation_text) % 10000
        guessed_article_link = summarizer_instance.generate_article_link(summary, str(conversation_id))
        
        logger.info("Chat analysis completed successfully")
        
        return ProcessedChatResponse(
            agent1_msg_count=agent1_msg_count,
            agent2_msg_count=agent2_msg_count,
            agent1_sentiment=agent1_sentiment,
            agent2_sentiment=agent2_sentiment,
            summary=summary,
            article_link=guessed_article_link
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during chat analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat analysis failed: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "message": "Chat analysis API is running",
        "dataset_path": DATASET_PATH
    }


# Additional utility endpoint to get supported sentiment types
@router.get("/sentiment-types")
def get_sentiment_types():
    return {
        "status": "success",
        "supported_sentiments": [
            "Curious to dive deeper", "Neutral", "Positive", "Negative",
            "Engaged", "Analytical", "Surprised", "Fearful",
            "Angry", "Sad", "Happy", "Disgusted"
        ]
    }
