import pandas as pd
import spacy
import re
import json
from typing import List, Union, Optional, Dict, Any
import logging
from collections import Counter
import warnings

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.logger import logger, log_start_end
from core.performance import timeit


@log_start_end
@timeit

class TextPreprocessor:
    """
    A comprehensive text preprocessing class using spaCy for NLP tasks.
    
    Features:
    - Tokenization using spaCy
    - Lemmatization
    - Stopword removal
    - Lowercasing
    - Punctuation and special character handling
    - Configurable preprocessing pipeline
    """
    
    def __init__(self, model_name: str = "en_core_web_sm", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TextPreprocessor with spaCy model and configuration.
        
        Args:
            model_name (str): spaCy model name to use
            config (dict, optional): Configuration dictionary for preprocessing parameters
        """
        self.model_name = model_name
        self.nlp = self._load_spacy_model(model_name)
        self.config = config or {}
        self.preprocessing_log = []
        
        # Default configuration
        self.default_config = {
            'lowercase': True,
            'remove_stopwords': True,
            'lemmatize': True,
            'remove_punctuation': True,
            'remove_numbers': False,
            'remove_whitespace': True,
            'min_token_length': 2,
            'max_token_length': 50,
            'remove_urls': True,
            'remove_emails': True,
            'remove_special_chars': True,
            'keep_alpha_only': False,  # Keep only alphabetic tokens
            'custom_stopwords': [],    # Additional stopwords
            'preserve_tokens': [],     # Tokens to never remove
            'return_as': 'string',     # 'string' or 'list'
            'log_intermediate': True,  # Log intermediate preprocessing steps
            'parse_json_content': True, # Parse JSON content if detected
            'handle_message_fields': True, # Special handling for message fields
            'log_token_counts': True   # Log token counts per row
        }
        
        # Merge user config with defaults
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Add custom stopwords to spaCy's stopwords
        self._setup_custom_stopwords()
    
    def _load_spacy_model(self, model_name: str):
        """Load spaCy model with error handling."""
        try:
            nlp = spacy.load(model_name)
            logging.info(f"Loaded spaCy model: {model_name}")
            return nlp
        except OSError:
            error_msg = f"""
            spaCy model '{model_name}' not found. Please install it using:
            python -m spacy download {model_name}
            
            For basic English processing, you can also try:
            python -m spacy download en_core_web_sm
            """
            logging.error(error_msg)
            raise OSError(error_msg)
    
    def _setup_custom_stopwords(self):
        """Add custom stopwords to the spaCy model."""
        custom_stopwords = self.config.get('custom_stopwords', [])
        if custom_stopwords:
            for word in custom_stopwords:
                self.nlp.vocab[word.lower()].is_stop = True
            self._log_action(f"Added {len(custom_stopwords)} custom stopwords")
    
    def _log_action(self, action: str):
        """Log preprocessing actions."""
        self.preprocessing_log.append(action)
        logging.info(f"TextPreprocessor: {action}")
    
    def _clean_urls_emails(self, text: str) -> str:
        """Remove URLs and email addresses from text."""
        if self.config['remove_urls']:
            # Remove URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            text = re.sub(url_pattern, '', text)
            # Remove www.domain patterns
            www_pattern = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            text = re.sub(www_pattern, '', text)
        
        if self.config['remove_emails']:
            # Remove email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            text = re.sub(email_pattern, '', text)
        
        return text
        
    def _clean_special_characters(self, text: str) -> str:
        """
        Remove special characters based on configuration.
        """
        if self.config['remove_special_chars']:
            # Remove special characters but keep spaces and basic punctuation
            text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
        
        if self.config['remove_numbers']:
            # Remove standalone numbers
            text = re.sub(r'\b\d+\b', '', text)
        
        return text
    
    def _parse_json_content(self, text: str) -> str:
        """
        Parse JSON content if detected and extract text fields.
        
        Args:
            text (str): Input text that might contain JSON
            
        Returns:
            str: Extracted text content
        """
        if not self.config.get('parse_json_content', True):
            return text
        
        # Check if text looks like JSON
        text_stripped = text.strip()
        if (text_stripped.startswith('{') and text_stripped.endswith('}')) or \
           (text_stripped.startswith('[') and text_stripped.endswith(']')):
            try:
                parsed_json = json.loads(text_stripped)
                extracted_text = self._extract_text_from_json(parsed_json)
                if self.config.get('log_intermediate', False):
                    self._log_action(f"Parsed JSON content: {len(extracted_text)} characters extracted")
                return extracted_text
            except json.JSONDecodeError:
                if self.config.get('log_intermediate', False):
                    self._log_action("Failed to parse JSON content, using original text")
                return text
        
        return text
    
    def _extract_text_from_json(self, json_obj: Union[dict, list, str, int, float]) -> str:
        """
        Recursively extract text content from JSON object.
        
        Args:
            json_obj: JSON object to extract text from
            
        Returns:
            str: Extracted text content
        """
        if isinstance(json_obj, str):
            return json_obj
        elif isinstance(json_obj, (int, float)):
            return str(json_obj)
        elif isinstance(json_obj, dict):
            text_parts = []
            # Special handling for message fields
            if self.config.get('handle_message_fields', True) and 'message' in json_obj:
                if isinstance(json_obj['message'], list):
                    # Concatenate all message entries
                    messages = []
                    for msg in json_obj['message']:
                        if isinstance(msg, str):
                            messages.append(msg)
                        elif isinstance(msg, dict):
                            messages.append(self._extract_text_from_json(msg))
                    text_parts.append(' '.join(messages))
                    if self.config.get('log_intermediate', False):
                        self._log_action(f"Concatenated {len(json_obj['message'])} message fields")
                else:
                    text_parts.append(str(json_obj['message']))
            
            # Extract other text fields
            for key, value in json_obj.items():
                if key != 'message':  # Skip message if already handled
                    if isinstance(value, str):
                        text_parts.append(value)
                    elif isinstance(value, (dict, list)):
                        text_parts.append(self._extract_text_from_json(value))
            
            return ' '.join(text_parts)
        elif isinstance(json_obj, list):
            text_parts = []
            for item in json_obj:
                text_parts.append(self._extract_text_from_json(item))
            return ' '.join(text_parts)
        else:
            return str(json_obj)
    
    def _log_intermediate_step(self, step_name: str, text: str, row_index: Optional[int] = None):
        """Log intermediate preprocessing steps if enabled."""
        if self.config.get('log_intermediate', False):
            prefix = f"Row {row_index}: " if row_index is not None else ""
            text_preview = text[:100] + "..." if len(text) > 100 else text
            self._log_action(f"{prefix}{step_name} - Length: {len(text)}, Preview: '{text_preview}'")
    
    def _count_tokens_before_filtering(self, doc) -> Dict[str, int]:
        """Count different types of tokens before filtering."""
        counts = {
            'total_tokens': len(doc),
            'alpha_tokens': sum(1 for token in doc if token.is_alpha),
            'stop_words': sum(1 for token in doc if token.is_stop),
            'punctuation': sum(1 for token in doc if token.is_punct),
            'spaces': sum(1 for token in doc if token.is_space),
            'numbers': sum(1 for token in doc if token.like_num)
        }
        return counts
    
    def preprocess_text(self, text: str, row_index: Optional[int] = None) -> Union[str, List[str]]:
        """
        Preprocess a single text string with detailed logging.
        
        Args:
            text (str): Input text to preprocess
            row_index (int, optional): Row index for logging purposes
            
        Returns:
            Union[str, List[str]]: Preprocessed text as string or list of tokens
        """
        if not isinstance(text, str) or not text.strip():
            if self.config.get('log_intermediate', False):
                self._log_action(f"Row {row_index}: Empty or invalid input")
            return "" if self.config['return_as'] == 'string' else []
        
        # Log original text
        if self.config.get('log_intermediate', False):
            self._log_intermediate_step("Original text", text, row_index)
        
        # Step 1: Parse JSON content if needed
        text = self._parse_json_content(text)
        if self.config.get('log_intermediate', False):
            self._log_intermediate_step("After JSON parsing", text, row_index)
        
        # Step 2: Lowercase
        if self.config['lowercase']:
            text = text.lower()
            if self.config.get('log_intermediate', False):
                self._log_intermediate_step("After lowercasing", text, row_index)
        
        # Step 3: Clean URLs and emails
        text = self._clean_urls_emails(text)
        if self.config.get('log_intermediate', False):
            self._log_intermediate_step("After URL/email removal", text, row_index)
        
        # Step 4: Clean special characters
        text = self._clean_special_characters(text)
        if self.config.get('log_intermediate', False):
            self._log_intermediate_step("After special char removal", text, row_index)
        
        # Step 5: Remove extra whitespace
        if self.config['remove_whitespace']:
            text = re.sub(r'\s+', ' ', text).strip()
            if self.config.get('log_intermediate', False):
                self._log_intermediate_step("After whitespace cleanup", text, row_index)
        
        # Step 6: Process with spaCy
        doc = self.nlp(text)
        
        # Log token counts before filtering
        if self.config.get('log_token_counts', False):
            token_counts = self._count_tokens_before_filtering(doc)
            if row_index is not None:
                self._log_action(f"Row {row_index} token counts before filtering: {token_counts}")
            else:
                self._log_action(f"Token counts before filtering: {token_counts}")
        
        # Step 7: Extract and filter tokens
        tokens = []
        preserve_tokens = [token.lower() for token in self.config.get('preserve_tokens', [])]
        
        for token in doc:
            # Skip tokens based on configuration
            if self._should_skip_token(token, preserve_tokens):
                continue
            
            # Apply lemmatization if requested
            if self.config['lemmatize']:
                token_text = token.lemma_
            else:
                token_text = token.text
            
            # Final token cleaning
            token_text = token_text.strip()
            
            # Length filtering
            if (len(token_text) >= self.config['min_token_length'] and 
                len(token_text) <= self.config['max_token_length']):
                tokens.append(token_text)
        
        # Log final token count
        if self.config.get('log_token_counts', False):
            if row_index is not None:
                self._log_action(f"Row {row_index}: Final token count: {len(tokens)}")
            else:
                self._log_action(f"Final token count: {len(tokens)}")
        
        # Log final result
        if self.config.get('log_intermediate', False):
            result_preview = ' '.join(tokens[:10]) + ("..." if len(tokens) > 10 else "")
            self._log_intermediate_step("Final result", result_preview, row_index)
        
        # Return based on configuration
        if self.config['return_as'] == 'string':
            return ' '.join(tokens)
        else:
            return tokens
    
    def _should_skip_token(self, token, preserve_tokens: List[str]) -> bool:
        """Determine if a token should be skipped based on configuration."""
        token_lower = token.text.lower()
        
        # Never skip preserved tokens
        if token_lower in preserve_tokens:
            return False
        
        # Skip stopwords
        if self.config['remove_stopwords'] and token.is_stop:
            return True
        
        # Skip punctuation
        if self.config['remove_punctuation'] and token.is_punct:
            return True
        
        # Skip spaces
        if token.is_space:
            return True
        
        # Skip non-alphabetic tokens if configured
        if self.config['keep_alpha_only'] and not token.is_alpha:
            return True
        
        # Skip tokens that are just numbers
        if self.config['remove_numbers'] and token.like_num:
            return True
        
        return False
    
    def preprocess_column(self, df: pd.DataFrame, column_name: str, 
                         new_column_name: Optional[str] = None, 
                         inplace: bool = False) -> pd.DataFrame:
        """
        Apply preprocessing to a specific column in a DataFrame with detailed logging.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            column_name (str): Name of the column to preprocess
            new_column_name (str, optional): Name for the new preprocessed column
            inplace (bool): Whether to modify the original DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with preprocessed column
        """
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")
        
        # Create a copy if not inplace
        if not inplace:
            df = df.copy()
        
        # Determine output column name
        output_col = new_column_name or f"{column_name}_preprocessed"
        
        # Apply preprocessing with detailed logging
        total_rows = len(df)
        self._log_action(f"Starting preprocessing of column '{column_name}' ({total_rows} rows)")
        
        # Handle missing values
        mask = df[column_name].notna()
        df[output_col] = ""
        
        if mask.sum() > 0:  # If there are non-null values
            # Apply preprocessing to non-null values with row-by-row logging
            preprocessed_results = []
            
            for idx, (row_idx, text) in enumerate(df.loc[mask, column_name].items()):
                # Special handling for Row 3 (index 2)
                if row_idx == 2 and self.config.get('handle_message_fields', True):
                    self._log_action(f"Special processing for Row 3 (index {row_idx})")
                    # Check if this is JSON content with message fields
                    if isinstance(text, str):
                        try:
                            parsed = json.loads(text)
                            if isinstance(parsed, dict) and 'message' in parsed:
                                if isinstance(parsed['message'], list):
                                    # Extract and concatenate message fields
                                    messages = []
                                    for msg in parsed['message']:
                                        if isinstance(msg, str):
                                            messages.append(msg)
                                        elif isinstance(msg, dict):
                                            messages.append(self._extract_text_from_json(msg))
                                    concatenated_messages = ' '.join(messages)
                                    self._log_action(f"Row 3: Concatenated {len(parsed['message'])} message fields into: '{concatenated_messages[:100]}{'...' if len(concatenated_messages) > 100 else ''}'")
                                    text = concatenated_messages
                        except json.JSONDecodeError:
                            pass
                
                # Preprocess with row index for logging
                preprocessed_text = self.preprocess_text(text, row_index=row_idx)
                preprocessed_results.append(preprocessed_text)
                
                # Log progress every 100 rows
                if (idx + 1) % 100 == 0:
                    self._log_action(f"Processed {idx + 1}/{mask.sum()} rows")
            
            # Assign results
            df.loc[mask, output_col] = preprocessed_results
        
        # Log final statistics
        non_empty_after = (df[output_col] != "").sum()
        empty_after = (df[output_col] == "").sum()
        self._log_action(f"Preprocessing completed: {mask.sum()} non-null inputs → {non_empty_after} non-empty results, {empty_after} empty results")
        
        # Log detailed statistics for first few rows
        if self.config.get('log_intermediate', False):
            self._log_action("Sample results (first 3 rows):")
            for i in range(min(3, len(df))):
                original = str(df[column_name].iloc[i])[:100] + ("..." if len(str(df[column_name].iloc[i])) > 100 else "")
                processed = str(df[output_col].iloc[i])[:100] + ("..." if len(str(df[output_col].iloc[i])) > 100 else "")
                self._log_action(f"  Row {i}: '{original}' → '{processed}'")
        
        return df
    
    def preprocess_multiple_columns(self, df: pd.DataFrame, 
                                  columns: List[str], 
                                  suffix: str = "_preprocessed",
                                  inplace: bool = False) -> pd.DataFrame:
        """
        Apply preprocessing to multiple columns in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): List of column names to preprocess
            suffix (str): Suffix to add to new column names
            inplace (bool): Whether to modify the original DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with preprocessed columns
        """
        if not inplace:
            df = df.copy()
        
        for col in columns:
            if col in df.columns:
                new_col_name = f"{col}{suffix}"
                df = self.preprocess_column(df, col, new_col_name, inplace=True)
            else:
                warnings.warn(f"Column '{col}' not found in DataFrame, skipping...")
        
        return df
    
    def get_preprocessing_stats(self, df: pd.DataFrame, 
                              original_column: str, 
                              preprocessed_column: str) -> Dict[str, Any]:
        """
        Get statistics about the preprocessing results.
        
        Args:
            df (pd.DataFrame): DataFrame containing both columns
            original_column (str): Name of original column
            preprocessed_column (str): Name of preprocessed column
            
        Returns:
            dict: Statistics about preprocessing
        """
        original_texts = df[original_column].dropna()
        preprocessed_texts = df[preprocessed_column].dropna()
        
        # Calculate word counts
        original_word_counts = original_texts.str.split().str.len()
        if self.config['return_as'] == 'string':
            preprocessed_word_counts = preprocessed_texts.str.split().str.len()
        else:
            preprocessed_word_counts = preprocessed_texts.str.len()
        
        # Get vocabulary
        if self.config['return_as'] == 'string':
            all_preprocessed_words = ' '.join(preprocessed_texts).split()
        else:
            all_preprocessed_words = []
            for text_list in preprocessed_texts:
                if isinstance(text_list, list):
                    all_preprocessed_words.extend(text_list)
        
        vocab_counter = Counter(all_preprocessed_words)
        
        stats = {
            'total_documents': len(df),
            'non_null_original': len(original_texts),
            'non_empty_preprocessed': (preprocessed_texts != "").sum(),
            'avg_words_original': original_word_counts.mean() if len(original_word_counts) > 0 else 0,
            'avg_words_preprocessed': preprocessed_word_counts.mean() if len(preprocessed_word_counts) > 0 else 0,
            'vocabulary_size': len(vocab_counter),
            'most_common_words': vocab_counter.most_common(10),
            'preprocessing_log': self.preprocessing_log
        }
        
        return stats
    
    def save_vocabulary(self, df: pd.DataFrame, column: str, 
                       output_file: str, min_frequency: int = 1):
        """
        Save vocabulary from preprocessed text to a file.
        
        Args:
            df (pd.DataFrame): DataFrame containing preprocessed text
            column (str): Column name with preprocessed text
            output_file (str): Path to save vocabulary file
            min_frequency (int): Minimum frequency to include word in vocabulary
        """
        # Extract all words
        if self.config['return_as'] == 'string':
            all_words = ' '.join(df[column].dropna()).split()
        else:
            all_words = []
            for text_list in df[column].dropna():
                if isinstance(text_list, list):
                    all_words.extend(text_list)
        
        # Count frequencies
        vocab_counter = Counter(all_words)
        
        # Filter by minimum frequency
        filtered_vocab = {word: count for word, count in vocab_counter.items() 
                         if count >= min_frequency}
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("word,frequency\n")
            for word, count in sorted(filtered_vocab.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{word},{count}\n")
        
        self._log_action(f"Saved vocabulary ({len(filtered_vocab)} words) to {output_file}")


# Utility functions
def load_and_preprocess_dataset(file_path: str, 
                               text_columns: List[str],
                               config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Load dataset and preprocess specified text columns.
    
    Args:
        file_path (str): Path to the dataset file
        text_columns (List[str]): List of text columns to preprocess
        config (dict, optional): Preprocessing configuration
        
    Returns:
        pd.DataFrame: DataFrame with preprocessed text columns
    """
    try:
        # Load the dataset
        df = pd.read_json(file_path)
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor(config=config)
        
        # Preprocess specified columns
        df = preprocessor.preprocess_multiple_columns(df, text_columns)
        
        # Print statistics
        for col in text_columns:
            if col in df.columns:
                preprocessed_col = f"{col}_preprocessed"
                if preprocessed_col in df.columns:
                    stats = preprocessor.get_preprocessing_stats(df, col, preprocessed_col)
                    print(f"\nPreprocessing stats for '{col}':")
                    print(f"Original avg words: {stats['avg_words_original']:.2f}")
                    print(f"Preprocessed avg words: {stats['avg_words_preprocessed']:.2f}")
                    print(f"Vocabulary size: {stats['vocabulary_size']}")
                    print(f"Most common words: {[word for word, count in stats['most_common_words'][:5]]}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None


if __name__ == "__main__":
    # Example usage with the specified dataset
    dataset_path = "data/BiztelAI_DS_Dataset_V1.json"
    
    # Custom configuration example
    preprocessing_config = {
        'lowercase': True,
        'remove_stopwords': True,
        'lemmatize': True,
        'remove_punctuation': True,
        'min_token_length': 2,
        'remove_urls': True,
        'remove_emails': True,
        'return_as': 'string',  # or 'list' for token lists
        'custom_stopwords': ['example', 'custom'],  # Add custom stopwords
        'log_intermediate': True,  # Enable detailed logging
        'parse_json_content': True,  # Parse JSON content
        'handle_message_fields': True,  # Special handling for message fields
        'log_token_counts': True  # Log token counts per row
    }
    
    try:
        # Load dataset
        df = pd.read_json(dataset_path)
        print(f"Loaded dataset with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Identify text columns (you may need to adjust these based on your actual dataset)
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        print(f"Detected text columns: {text_columns}")
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor(config=preprocessing_config)
        
        # Example: Preprocess a specific column with detailed logging
        if text_columns:
            sample_column = text_columns[0]  # Use first text column as example
            print(f"\nPreprocessing column '{sample_column}' with detailed logging...")
            
            # Enable logging to see intermediate steps
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            
            df_processed = preprocessor.preprocess_column(df, sample_column)
            
            # Show results
            print(f"\nExample preprocessing results for column '{sample_column}':")
            print("\nOriginal vs Preprocessed (first 3 rows):")
            for i in range(min(3, len(df_processed))):
                original = df_processed[sample_column].iloc[i]
                preprocessed = df_processed[f"{sample_column}_preprocessed"].iloc[i]
                print(f"\nRow {i+1} (Index {i}):")
                print(f"Original: {str(original)[:200]}{'...' if len(str(original)) > 200 else ''}")
                print(f"Preprocessed: {str(preprocessed)[:200]}{'...' if len(str(preprocessed)) > 200 else ''}")
                
                # Special attention to Row 3 (index 2)
                if i == 2:
                    print(f"*** SPECIAL PROCESSING FOR ROW 3 ***")
                    # Check if original contains JSON with message fields
                    if isinstance(original, str):
                        try:
                            parsed = json.loads(original)
                            if isinstance(parsed, dict) and 'message' in parsed:
                                print(f"Row 3 contains message field with {len(parsed['message']) if isinstance(parsed['message'], list) else 1} entries")
                                if isinstance(parsed['message'], list):
                                    print(f"Message entries: {[str(msg)[:50] + '...' if len(str(msg)) > 50 else str(msg) for msg in parsed['message'][:3]]}")
                        except json.JSONDecodeError:
                            print("Row 3 is not valid JSON")
            
            # Get and display statistics
            stats = preprocessor.get_preprocessing_stats(
                df_processed, sample_column, f"{sample_column}_preprocessed"
            )
            print(f"\nPreprocessing Statistics:")
            print(f"Total documents: {stats['total_documents']}")
            print(f"Average words (original): {stats['avg_words_original']:.2f}")
            print(f"Average words (preprocessed): {stats['avg_words_preprocessed']:.2f}")
            print(f"Vocabulary size: {stats['vocabulary_size']}")
            print(f"Top 5 words: {[word for word, count in stats['most_common_words'][:5]]}")
            
            # Show preprocessing log
            print(f"\nPreprocessing Log (last 10 entries):")
            for log_entry in preprocessor.preprocessing_log[-10:]:
                print(f"  {log_entry}")
        
    except Exception as e:
        print(f"Error in example execution: {str(e)}")
        print("Make sure you have spaCy installed and the English model downloaded:")
        print("pip install spacy")
        print("python -m spacy download en_core_web_sm")