import json
from textblob import TextBlob
from collections import defaultdict
from typing import List, Dict, Union, Any

class ChatAnalyzer:
    """
    A class to analyze chat transcripts and compute message counts and sentiment analysis per agent.
    """
    
    def __init__(self):
        """Initialize the ChatAnalyzer."""
        self.sentiment_thresholds = {
            'positive': 0.1,
            'negative': -0.1
        }
    
    def load_chat_data(self, file_path: str) -> List[Dict]:
        """
        Load chat data from the BiztelAI dataset JSON file.
        
        Args:
            file_path (str): Path to the JSON file containing chat data
            
        Returns:
            List[Dict]: List of chat messages from all conversations
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            all_messages = []
            
            # Handle BiztelAI dataset structure
            if isinstance(data, dict):
                # Iterate through each conversation (e.g., "t_d004c097-424d-45d4-8f91-833d85c2da31")
                for conversation_id, conversation_data in data.items():
                    if isinstance(conversation_data, dict) and 'content' in conversation_data:
                        # Extract messages from the 'content' array
                        messages = conversation_data['content']
                        if isinstance(messages, list):
                            # Add conversation metadata to each message
                            for message in messages:
                                message_copy = message.copy()
                                message_copy['conversation_id'] = conversation_id
                                message_copy['article_url'] = conversation_data.get('article_url', '')
                                message_copy['config'] = conversation_data.get('config', '')
                                all_messages.append(message_copy)
                        
            return all_messages
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Chat data file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ValueError(f"Error processing dataset structure: {e}")
    
    def extract_agent_and_message(self, message_obj: Dict) -> tuple:
        """
        Extract agent ID and message text from the BiztelAI dataset message object.
        
        Args:
            message_obj (Dict): Single message object from BiztelAI dataset
            
        Returns:
            tuple: (agent_id, message_text)
        """
        # For BiztelAI dataset, the structure is well-defined
        agent_id = message_obj.get('agent')
        message_text = message_obj.get('message')
        
        if agent_id is None:
            raise ValueError(f"Could not find 'agent' field in message: {message_obj}")
        if message_text is None:
            raise ValueError(f"Could not find 'message' field in message: {message_obj}")
            
        return str(agent_id), str(message_text)
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Sentiment polarity score (-1 to 1)
        """
        if not text or text.strip() == "":
            return 0.0
        
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception:
            return 0.0
    
    def classify_sentiment(self, polarity: float) -> str:
        """
        Classify sentiment based on polarity score.
        
        Args:
            polarity (float): Sentiment polarity score
            
        Returns:
            str: Sentiment classification ('Positive', 'Negative', 'Neutral')
        """
        if polarity > self.sentiment_thresholds['positive']:
            return "Positive"
        elif polarity < self.sentiment_thresholds['negative']:
            return "Negative"
        else:
            return "Neutral"
    
    def analyze_chat(self, chat_data: Union[str, List[Dict]]) -> Dict[str, Union[int, str]]:
        """
        Analyze chat transcript to count messages and compute sentiment per agent.
        
        Args:
            chat_data: Either a file path (str) or list of message dictionaries
            
        Returns:
            Dict: Analysis results with message counts and sentiment per agent
        """
        # Load data if file path is provided
        if isinstance(chat_data, str):
            messages = self.load_chat_data(chat_data)
        else:
            messages = chat_data
        
        # Initialize counters
        agent_message_counts = defaultdict(int)
        agent_sentiments = defaultdict(list)
        
        # Process each message
        for message_obj in messages:
            try:
                agent_id, message_text = self.extract_agent_and_message(message_obj)
                
                # Count messages
                agent_message_counts[agent_id] += 1
                
                # Analyze sentiment
                sentiment_score = self.analyze_sentiment(message_text)
                agent_sentiments[agent_id].append(sentiment_score)
                
            except ValueError as e:
                print(f"Warning: Skipping message due to error: {e}")
                continue
        
        # Compute average sentiment per agent
        results = {}
        
        for agent_id in agent_message_counts:
            # Add message count
            results[f"{agent_id}_msg_count"] = agent_message_counts[agent_id]
            
            # Compute and classify average sentiment
            if agent_sentiments[agent_id]:
                avg_sentiment = sum(agent_sentiments[agent_id]) / len(agent_sentiments[agent_id])
                sentiment_class = self.classify_sentiment(avg_sentiment)
                results[f"{agent_id}_sentiment"] = sentiment_class
            else:
                results[f"{agent_id}_sentiment"] = "Neutral"
        
        return results
    
    def get_dataset_summary(self, chat_data: Union[str, List[Dict]]) -> Dict[str, Any]:
        """
        Get a summary of the BiztelAI dataset including conversation counts and structure.
        
        Args:
            chat_data: Either a file path (str) or list of message dictionaries
            
        Returns:
            Dict: Dataset summary statistics
        """
        if isinstance(chat_data, str):
            with open(chat_data, 'r', encoding='utf-8') as file:
                raw_data = json.load(file)
        else:
            # If already processed messages, we need to reconstruct some info
            messages = chat_data
            conversation_ids = set()
            for msg in messages:
                if 'conversation_id' in msg:
                    conversation_ids.add(msg['conversation_id'])
            
            return {
                'total_conversations': len(conversation_ids),
                'total_messages': len(messages),
                'unique_agents': len(set(msg.get('agent', 'unknown') for msg in messages))
            }
        
        # Analyze raw dataset structure
        total_conversations = len(raw_data)
        total_messages = 0
        agents = set()
        configs = set()
        sentiment_types = set()
        
        for conv_id, conv_data in raw_data.items():
            if isinstance(conv_data, dict) and 'content' in conv_data:
                messages = conv_data['content']
                total_messages += len(messages)
                
                # Collect unique values
                if 'config' in conv_data:
                    configs.add(conv_data['config'])
                
                for msg in messages:
                    if 'agent' in msg:
                        agents.add(msg['agent'])
                    if 'sentiment' in msg:
                        sentiment_types.add(msg['sentiment'])
        
        return {
            'total_conversations': total_conversations,
            'total_messages': total_messages,
            'unique_agents': list(agents),
            'configurations': list(configs),
            'sentiment_types_in_dataset': list(sentiment_types),
            'avg_messages_per_conversation': round(total_messages / total_conversations, 2) if total_conversations > 0 else 0
        }
    
    def get_detailed_analysis(self, chat_data: Union[str, List[Dict]]) -> Dict[str, Any]:
        """
        Get detailed analysis including raw sentiment scores and statistics.
        
        Args:
            chat_data: Either a file path (str) or list of message dictionaries
            
        Returns:
            Dict: Detailed analysis results
        """
        # Load data if file path is provided
        if isinstance(chat_data, str):
            messages = self.load_chat_data(chat_data)
        else:
            messages = chat_data
        
        agent_stats = defaultdict(lambda: {
            'message_count': 0,
            'sentiment_scores': [],
            'messages': []
        })
        
        # Process each message
        for message_obj in messages:
            try:
                agent_id, message_text = self.extract_agent_and_message(message_obj)
                
                # Count messages and store text
                agent_stats[agent_id]['message_count'] += 1
                agent_stats[agent_id]['messages'].append(message_text)
                
                # Analyze sentiment
                sentiment_score = self.analyze_sentiment(message_text)
                agent_stats[agent_id]['sentiment_scores'].append(sentiment_score)
                
            except ValueError as e:
                print(f"Warning: Skipping message due to error: {e}")
                continue
        
        # Compute detailed statistics
        detailed_results = {}
        
        for agent_id, stats in agent_stats.items():
            if stats['sentiment_scores']:
                avg_sentiment = sum(stats['sentiment_scores']) / len(stats['sentiment_scores'])
                sentiment_class = self.classify_sentiment(avg_sentiment)
                
                detailed_results[agent_id] = {
                    'message_count': stats['message_count'],
                    'average_sentiment_score': round(avg_sentiment, 3),
                    'sentiment_classification': sentiment_class,
                    'min_sentiment': round(min(stats['sentiment_scores']), 3),
                    'max_sentiment': round(max(stats['sentiment_scores']), 3),
                    'total_messages_analyzed': len(stats['sentiment_scores'])
                }
        
        return detailed_results


# Implementation with actual BiztelAI dataset
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = ChatAnalyzer()
    
    # Dataset file path
    dataset_path = "data/BiztelAI_DS_Dataset_V1.json"
    
    try:
        print("=== Loading BiztelAI Dataset ===")
        print(f"Loading dataset from: {dataset_path}")
        
        # Get dataset summary first
        print("\n=== Dataset Summary ===")
        summary = analyzer.get_dataset_summary(dataset_path)
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        print("\n=== Analyzing Chat Data ===")
        print("Processing all conversations for message counts and sentiment analysis...")
        
        # Main analysis - this is what you requested
        results = analyzer.analyze_chat(dataset_path)
        
        print("\n=== RESULTS (Requested Format) ===")
        for key, value in results.items():
            print(f'"{key}": {value}')
        
        print(f"\n=== Results as Dictionary ===")
        print(results)
        
        # Detailed analysis for additional insights
        print("\n=== Detailed Analysis ===")
        detailed_results = analyzer.get_detailed_analysis(dataset_path)
        
        for agent_id, stats in detailed_results.items():
            print(f"\n{agent_id.upper()} STATISTICS:")
            print(f"  Message Count: {stats['message_count']}")
            print(f"  Average Sentiment Score: {stats['average_sentiment_score']}")
            print(f"  Sentiment Classification: {stats['sentiment_classification']}")
            print(f"  Sentiment Range: {stats['min_sentiment']} to {stats['max_sentiment']}")
            print(f"  Total Messages Analyzed: {stats['total_messages_analyzed']}")
        
        # Additional analysis: Compare original sentiment labels vs computed sentiment
        print("\n=== Comparing Dataset Sentiment Labels vs Computed Sentiment ===")
        messages = analyzer.load_chat_data(dataset_path)
        
        # Count original sentiment labels per agent
        agent_original_sentiments = defaultdict(lambda: defaultdict(int))
        for msg in messages:
            agent = msg.get('agent', 'unknown')
            original_sentiment = msg.get('sentiment', 'unknown')
            agent_original_sentiments[agent][original_sentiment] += 1
        
        for agent, sentiment_counts in agent_original_sentiments.items():
            print(f"\n{agent.upper()} - Original Sentiment Labels:")
            for sentiment, count in sentiment_counts.items():
                print(f"  {sentiment}: {count} messages")
        
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at '{dataset_path}'")
        print("Please ensure the file exists at the specified path.")
        print("Current working directory files should include:")
        print("- data/BiztelAI_DS_Dataset_V1.json")
        
    except Exception as e:
        print(f"ERROR: An error occurred while processing the dataset: {e}")
        print("Please check that the file format matches the expected BiztelAI structure.")
        
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS:")
    print("1. Ensure your dataset file is at: data/BiztelAI_DS_Dataset_V1.json")
    print("2. Run this script to get message counts and sentiment analysis")
    print("3. The main results will be in the format you requested:")
    print('   {"agent_1_msg_count": X, "agent_2_msg_count": Y, ...}')
    print("="*60)