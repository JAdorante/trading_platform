import re
import logging
import secrets
import spacy
import numpy as np
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_entity_recognition")

# Stock keywords for advanced detection
STOCK_KEYWORDS = {
    "NVDA": ["Nvidia", "AI chips", "graphics processing", "semiconductor", "data center"],
    "AAPL": ["Apple", "iPhone", "Mac", "iOS", "Tim Cook", "Cupertino"],
    "MSFT": ["Microsoft", "Windows", "Azure", "cloud computing", "Satya Nadella"],
    "GOOGL": ["Google", "Alphabet", "search engine", "Android", "Sundar Pichai"],
    "AMZN": ["Amazon", "AWS", "e-commerce", "cloud services", "Jeff Bezos"],
    "META": ["Meta", "Facebook", "Instagram", "social media", "Mark Zuckerberg"],
    "TSLA": ["Tesla", "electric vehicle", "Elon Musk", "autonomous driving"],
    "AMD": ["Advanced Micro Devices", "processor", "chip", "Lisa Su"],
    "INTC": ["Intel", "processor", "semiconductor", "chip manufacturing"],
    "NFLX": ["Netflix", "streaming", "entertainment", "content production"]
}

class FinancialEntityRecognizer:
    def __init__(self):
        """
        Initialize the financial entity recognition system with multiple NLP models.
        """
        try:
            # Load SpaCy model for base NER
            self.nlp = spacy.load("en_core_web_sm")
            
            # Optional: Add custom entity ruler for financial entities
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            
            # Add stock ticker patterns
            patterns = []
            for ticker, keywords in STOCK_KEYWORDS.items():
                # Cashtag pattern
                patterns.append({"label": "STOCK", "pattern": [{"ORTH": f"${ticker}"}]})
                
                # Keyword patterns
                for keyword in keywords:
                    patterns.append({"label": "STOCK", "pattern": [{"LOWER": keyword.lower()}]})
            
            ruler.add_patterns(patterns)
            
            logger.info("Financial Entity Recognizer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Financial Entity Recognizer: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better entity recognition.
        
        Args:
            text (str): Input text to preprocess
        
        Returns:
            str: Preprocessed text
        """
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase for pattern matching
        return text.lower()
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract financial entities from the given text.
        
        Args:
            text (str): Text to analyze
        
        Returns:
            Dict[str, List[str]]: Extracted entities by type
        """
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Process with SpaCy
            doc = self.nlp(processed_text)
            
            # Initialize entities dictionary
            entities: Dict[str, List[str]] = {
                "ORG": [],
                "PRODUCT": [],
                "PERSON": [],
                "GPE": [],
                "STOCK": [],
                "MONEY": []
            }
            
            # Extract standard NER entities
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)
            
            # Extract stock-related entities
            stock_matches = self.detect_stocks(processed_text)
            entities["STOCK"].extend(stock_matches)
            
            # Remove duplicates while preserving order
            for key in entities:
                entities[key] = list(dict.fromkeys(entities[key]))
            
            return entities
        
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {
                "ORG": [],
                "PRODUCT": [],
                "PERSON": [],
                "GPE": [],
                "STOCK": [],
                "MONEY": []
            }
    
    def detect_stocks(self, text: str) -> List[str]:
        """
        Detect stock tickers and related keywords in text.
        
        Args:
            text (str): Preprocessed text to analyze
        
        Returns:
            List[str]: Detected stock tickers
        """
        # Initialize detected stocks
        detected_stocks = set()
        
        # Cashtag detection
        cashtag_pattern = r'\$([A-Z]{1,5})\b'
        cashtag_matches = re.findall(cashtag_pattern, text.upper())
        detected_stocks.update(cashtag_matches)
        
        # Keyword-based detection
        for ticker, keywords in STOCK_KEYWORDS.items():
            # Check if any keyword is in the text
            if any(keyword.lower() in text for keyword in keywords):
                detected_stocks.add(ticker)
        
        return list(detected_stocks)

def enhanced_stock_detection(text: str) -> Optional[List[str]]:
    """
    Public function for stock detection, handles text preprocessing and stock identification.
    
    Args:
        text (str): Text to analyze for stock tickers
    
    Returns:
        Optional[List[str]]: List of detected stock tickers, or None if no stocks found
    """
    try:
        recognizer = FinancialEntityRecognizer()
        detected_stocks = recognizer.detect_stocks(recognizer.preprocess_text(text))
        return detected_stocks if detected_stocks else None
    except Exception as e:
        logger.error(f"Error in enhanced stock detection: {e}")
        return None

# Create a global instance for easy import
FINANCIAL_NER = FinancialEntityRecognizer()