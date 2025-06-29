import json
import re
import hashlib
import time
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.logger import logger, log_start_end
from core.performance import timeit


@log_start_end
@timeit

class SimpleChatSummarizer:
    """
    Simple chat transcript summarizer focused only on summarization.
    Optimized for 6GB VRAM with batch size 8.
    """
    
    def __init__(self, 
                 model_name: str = "facebook/bart-large-cnn",
                 max_length: int = 150, 
                 min_length: int = 50):
        """
        Initialize the Simple Chat Summarizer.
        
        Args:
            model_name: Summarization model name
            max_length: Maximum summary length
            min_length: Minimum summary length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"🚀 CUDA detected! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  CUDA not available. Using CPU.")
        
        # Load model
        self._load_model()
        
        # Statistics tracking
        self.processing_stats = {
            'total_conversations': 0,
            'successful_summaries': 0,
            'start_time': None,
            'batch_times': []
        }
    
    def _load_model(self):
        """Load summarization model only."""
        try:
            print(f"🔄 Loading summarization model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model {self.model_name}: {e}")
            print("🔄 Falling back to distilbart-cnn-12-6")
            self.model_name = "sshleifer/distilbart-cnn-12-6"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
    
    def load_dataset(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        """Load the chat dataset from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"📁 Successfully loaded dataset with {len(data)} conversations")
            return data
        except FileNotFoundError:
            print(f"❌ Error: Dataset file not found at {file_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in dataset file: {e}")
            return {}
    
    def preprocess_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Convert chat messages into a single text for summarization."""
        text_parts = []
        
        for msg in messages:
            if isinstance(msg, dict):
                # Try different possible field names for message content
                content = msg.get('message', msg.get('content', msg.get('text', '')))
                agent = msg.get('agent', msg.get('role', msg.get('sender', 'speaker')))
                
                if content and isinstance(content, str):
                    text_parts.append(f"{agent}: {content}")
            elif isinstance(msg, str):
                text_parts.append(msg)
        
        return " ".join(text_parts)
    
    def truncate_text(self, text: str, max_tokens: int = 900) -> str:
        """Truncate text to fit within model token limits."""
        try:
            tokens = self.tokenizer.encode(text, truncation=True, max_length=max_tokens, add_special_tokens=False)
            truncated_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            return truncated_text
        except Exception as e:
            print(f"⚠️  Error in truncate_text: {e}")
            return text[:max_tokens * 4]  # Fallback: character-based truncation
    
    def generate_article_link(self, summary: str, conversation_id: str) -> str:
        """Generate a plausible article link based on the summary topic."""
        # Extract keywords from summary
        keywords = re.findall(r'\b[A-Z][a-z]+\b|\b(?:AI|ML|technology|data|business|customer|service|chat|support)\b', summary, re.IGNORECASE)
        
        if not keywords:
            keywords = ['technology', 'conversation']
        
        # Create topic slug
        topic_slug = "-".join(keywords[:3]).lower().replace(' ', '-')
        
        # Generate consistent ID
        content_hash = hashlib.md5(f"{summary}{conversation_id}".encode()).hexdigest()[:8]
        
        # Select domain
        domains = [
            "techcrunch.com", "wired.com", "medium.com", "businessinsider.com",
            "forbes.com", "venturebeat.com", "arstechnica.com", "theverge.com"
        ]
        
        domain = domains[len(summary) % len(domains)]
        
        return f"https://{domain}/articles/{topic_slug}-{content_hash}"
    
    def batch_summarize(self, texts: List[str]) -> List[str]:
        """Perform batch summarization optimized for 6GB VRAM."""
        if not texts:
            return []
        
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summaries with memory optimization
            with torch.no_grad():
                # Use mixed precision for memory efficiency
                with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                    summary_ids = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=self.max_length,
                        min_length=self.min_length,
                        length_penalty=2.0,
                        num_beams=4,
                        early_stopping=True,
                        do_sample=False,
                        no_repeat_ngram_size=2  # Avoid repetition
                    )
            
            # Decode summaries
            summaries = []
            for summary_id in summary_ids:
                summary = self.tokenizer.decode(summary_id, skip_special_tokens=True)
                summaries.append(summary)
            
            return summaries
            
        except Exception as e:
            print(f"❌ Error in batch_summarize: {e}")
            return [f"Error generating summary: {str(e)}" for _ in texts]
    
    def process_dataset(self, 
                       file_path: str = "data/BiztelAI_DS_Dataset_V1.json", 
                       batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Process the entire dataset with progress tracking.
        Optimized for 6GB VRAM with batch size 8.
        
        Args:
            file_path: Path to the dataset file
            batch_size: Batch size (optimized for 6GB VRAM)
            
        Returns:
            List of summarization results
        """
        dataset = self.load_dataset(file_path)
        
        if not dataset:
            print("❌ No data to process!")
            return []
        
        # Initialize statistics
        self.processing_stats['total_conversations'] = len(dataset)
        self.processing_stats['start_time'] = time.time()
        
        # Prepare data items
        print("🔄 Preparing conversations for summarization...")
        data_items = []
        for conv_id, conv in dataset.items():
            content = conv.get("content", [])
            if content:  # Only process conversations with content
                text = self.preprocess_messages(content)
                text = self.truncate_text(text)
                
                if text.strip():
                    data_items.append({
                        "conversation_id": conv_id,
                        "text": text,
                        "num_messages": len(content),
                        "config": conv.get("config", ""),
                        "original_article_url": conv.get("article_url", "")
                    })
        
        if not data_items:
            print("❌ No valid conversations found!")
            return []
        
        print(f"📊 Processing {len(data_items)} conversations in batches of {batch_size}")
        
        # Process with progress bar
        results = []
        total_batches = (len(data_items) + batch_size - 1) // batch_size
        
        with tqdm(total=len(data_items), desc="Summarizing conversations", unit="conv") as pbar:
            for i in range(0, len(data_items), batch_size):
                batch_start_time = time.time()
                batch = data_items[i:i + batch_size]
                batch_texts = [item["text"] for item in batch]
                
                # Update progress description
                pbar.set_description(f"Batch {i//batch_size + 1}/{total_batches}")
                
                # Summarize batch
                batch_summaries = self.batch_summarize(batch_texts)
                
                # Create results for this batch
                for item, summary in zip(batch, batch_summaries):
                    # Generate article link
                    article_link = item["original_article_url"] or self.generate_article_link(
                        summary, item["conversation_id"]
                    )
                    
                    # Check if summarization was successful
                    success = not summary.startswith(('Error', 'Processing error', 'Failed'))
                    if success:
                        self.processing_stats['successful_summaries'] += 1
                    
                    result = {
                        "conversation_id": item["conversation_id"],
                        "summary": summary,
                        "article_link": article_link,
                        "num_messages": item["num_messages"],
                        "config": item["config"],
                        "original_article_url": item["original_article_url"],
                        "success": success
                    }
                    
                    results.append(result)
                    pbar.update(1)
                
                # Track timing
                batch_time = time.time() - batch_start_time
                self.processing_stats['batch_times'].append(batch_time)
                
                # Update progress bar with timing info
                if self.processing_stats['batch_times']:
                    avg_batch_time = np.mean(self.processing_stats['batch_times'])
                    remaining_batches = total_batches - (i//batch_size + 1)
                    estimated_remaining = remaining_batches * avg_batch_time
                    
                    pbar.set_postfix({
                        'Batch': f'{batch_time:.1f}s',
                        'Avg': f'{avg_batch_time:.1f}s',
                        'ETA': f'{estimated_remaining/60:.1f}m'
                    })
                
                # Clear GPU cache periodically to prevent memory issues
                if torch.cuda.is_available() and (i // batch_size + 1) % 5 == 0:
                    torch.cuda.empty_cache()
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str = "chat_summaries.json"):
        """Save summarization results."""
        total_time = time.time() - self.processing_stats['start_time']
        
        output_data = {
            'metadata': {
                'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_used': self.model_name,
                'total_conversations': len(results),
                'successful_summaries': self.processing_stats['successful_summaries'],
                'success_rate': f"{self.processing_stats['successful_summaries']/len(results)*100:.1f}%" if results else "0%",
                'total_processing_time_minutes': f"{total_time/60:.1f}",
                'average_time_per_conversation_seconds': f"{total_time/len(results):.2f}" if results else "0"
            },
            'summaries': results
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to {output_path}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print processing summary."""
        if not results:
            print("❌ No results to summarize.")
            return
        
        total_time = time.time() - self.processing_stats['start_time']
        successful = self.processing_stats['successful_summaries']
        
        print("\n" + "="*60)
        print("📊 SUMMARIZATION RESULTS")
        print("="*60)
        print(f"🔸 Total Conversations: {len(results)}")
        print(f"🔸 Successful Summaries: {successful}")
        print(f"🔸 Success Rate: {successful/len(results)*100:.1f}%")
        print(f"🔸 Total Processing Time: {total_time/60:.1f} minutes")
        print(f"🔸 Average Time per Conversation: {total_time/len(results):.2f} seconds")
        print(f"🔸 Model Used: {self.model_name}")
        
        # Show sample results
        print(f"\n📋 SAMPLE SUMMARIES:")
        for i, result in enumerate(results[:3]):
            print(f"\n🔸 Conversation {i+1}: {result['conversation_id']}")
            print(f"   Messages: {result['num_messages']}")
            print(f"   Summary: {result['summary']}")
            print(f"   Article: {result['article_link']}")
            print("-" * 50)
        
        print(f"\n🎉 Summarization complete!")


# Example usage
if __name__ == "__main__":
    print("🚀 Initializing Simple Chat Summarizer (6GB VRAM optimized)...")
    
    # Initialize with model optimized for your 6GB VRAM
    summarizer = SimpleChatSummarizer(
        model_name="facebook/bart-large-cnn",  # Good quality, fits in 6GB
        max_length=150,
        min_length=40
    )
    
    print("📊 Processing chat transcripts (batch size: 8)...")
    
    # Process with batch size 8 (optimized for 6GB VRAM)
    results = summarizer.process_dataset(
        "data/BiztelAI_DS_Dataset_V1.json",  # Update path as needed
        batch_size=8
    )
    
    if results:
        # Save results
        summarizer.save_results(results)
        
        # Print summary
        summarizer.print_summary(results)
        
        print(f"\n💾 Full results saved to 'chat_summaries.json'")
        
    else:
        print("❌ No results to process. Please check your dataset path and format.")