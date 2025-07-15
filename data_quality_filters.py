import re
import logging
import hashlib
import pickle
import numpy as np
from typing import Dict, Any, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import secrets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_quality_filters")

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Error downloading NLTK resources: {e}")

# Spam and low-quality content patterns
SPAM_PATTERNS = [
    r'get rich quick',
    r'100%\s*guaranteed',
    r'click\s*here',
    r'limited\s*time\s*offer',
    r'make\s*money\s*fast',
    r'no\s*risk',
    r'investment\s*secret',
    r'\$\$\$',
    r'rocket\s*emoji',
    r'to\s*the\s*moon',
    r'diamond\s*hands'
]

# Financial content quality keywords
FINANCIAL_KEYWORDS = [
    'stock', 'market', 'trading', 'investment', 'earnings', 
    'revenue', 'profit', 'analysis', 'trend', 'financial', 
    'price', 'dividend', 'portfolio', 'fund', 'equity'
]

class ContentQualityFilter:
    def __init__(self):
        """
        Initialize content quality filter with various assessment mechanisms.
        """
        try:
            # Stopwords for cleanup
            self.stopwords = set(stopwords.words('english'))
            
            # Compile spam patterns
            self.spam_regex = [re.compile(pattern, re.IGNORECASE) for pattern in SPAM_PATTERNS]
            
            logger.info("Content Quality Filter initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Content Quality Filter: {e}")
            self.stopwords = set()
            self.spam_regex = []
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for quality assessment.
        
        Args:
            text (str): Input text
        
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def detect_spam(self, text: str) -> bool:
        """
        Detect spam content based on predefined patterns.
        
        Args:
            text (str): Text to analyze
        
        Returns:
            bool: True if spam is detected, False otherwise
        """
        # Check against spam patterns
        for pattern in self.spam_regex:
            if pattern.search(text):
                return True
        
        return False
    
    def calculate_content_score(self, text: str) -> float:
        """
        Calculate a comprehensive quality score for the content.
        
        Args:
            text (str): Text to evaluate
        
        Returns:
            float: Content quality score (0-1)
        """
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Very short text is considered low quality
            if len(processed_text) < 20:
                return 0.0
            
            # Tokenize and remove stopwords
            tokens = [token for token in word_tokenize(processed_text) if token not in self.stopwords]
            
            # Calculate various quality metrics
            metrics = {
                'length_score': min(1.0, len(processed_text) / 500),  # Longer texts score higher
                'token_diversity': len(set(tokens)) / len(tokens) if tokens else 0,
                'financial_keyword_ratio': sum(
                    1 for token in tokens if token in FINANCIAL_KEYWORDS
                ) / len(tokens) if tokens else 0,
                'spam_penalty': 0.5 if self.detect_spam(text) else 1.0
            }
            
            # Weighted score calculation
            quality_score = (
                (metrics['length_score'] * 0.3) +
                (metrics['token_diversity'] * 0.3) +
                (metrics['financial_keyword_ratio'] * 0.3) +
                (metrics['spam_penalty'] * 0.1)
            )
            
            return np.clip(quality_score, 0.0, 1.0)
        
        except Exception as e:
            logger.error(f"Error calculating content score: {e}")
            return 0.5  # Default neutral score
    
    def detect_duplicates(self, documents: list, threshold: float = 0.9) -> list:
        """
        Detect duplicate documents using content hashing.
        
        Args:
            documents (list): List of documents to check
            threshold (float): Similarity threshold for duplicates
        
        Returns:
            list: Indices of duplicate documents
        """
        def generate_hash(text: str) -> str:
            """Generate a hash for a given text."""
            processed = self.preprocess_text(text)
            return hashlib.md5(processed.encode()).hexdigest()
        
        hashes = [generate_hash(doc) for doc in documents]
        
        duplicates = []
        for i in range(len(hashes)):
            if i in duplicates:
                continue
            
            for j in range(i + 1, len(hashes)):
                if hashes[i] == hashes[j]:
                    duplicates.append(j)
        
        return duplicates

def filter_quality_content(
    text: str, 
    source: str = "Unknown", 
    min_quality_score: float = 0.3
) -> Dict[str, Any]:
    """
    Public function for content quality filtering.
    
    Args:
        text (str): Text to evaluate
        source (str): Source of the content
        min_quality_score (float): Minimum acceptable quality score
    
    Returns:
        Dict[str, Any]: Quality assessment results
    """
    try:
        # Initialize content filter
        content_filter = ContentQualityFilter()
        
        # Calculate quality score
        quality_score = content_filter.calculate_content_score(text)
        
        # Detect spam
        is_spam = content_filter.detect_spam(text)
        
        # Determine if content passes quality threshold
        is_quality_content = (
            quality_score >= min_quality_score and 
            not is_spam and 
            len(text.strip()) >= 20
        )
        
        # Determine filter reason if not quality content
        filter_reason = (
            "Low quality score" if quality_score < min_quality_score else
            "Detected as spam" if is_spam else
            "Text too short" if len(text.strip()) < 20 else
            None
        )
        
        return {
            "is_quality_content": is_quality_content,
            "quality_score": round(quality_score, 2),
            "source": source,
            "is_spam": is_spam,
            "reason": filter_reason
        }
    
    except Exception as e:
        logger.error(f"Error in content quality filtering: {e}")
        return {
            "is_quality_content": False,
            "quality_score": 0.0,
            "source": source,
            "is_spam": False,
            "reason": "Unexpected error in quality assessment"
        }

# Create a global content filter instance
CONTENT_FILTER = ContentQualityFilter()