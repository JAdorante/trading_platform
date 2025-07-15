import re
import logging
import numpy as np
from typing import Dict, Any, Optional
from transformers import pipeline
import secrets
import pickle
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sentiment_analysis_refinement")

# Financial sentiment lexicons and context modifiers
FINANCIAL_SENTIMENT_LEXICON = {
    # Positive financial terms with their sentiment boost
    "positive": {
        "bullish": 0.3, 
        "growth": 0.2, 
        "profit": 0.25, 
        "revenue": 0.2, 
        "earnings": 0.25, 
        "innovation": 0.2, 
        "breakthrough": 0.3,
        "expansion": 0.2,
        "investment": 0.15,
        "merger": 0.15,
        "acquisition": 0.15,
        "partnership": 0.15
    },
    # Negative financial terms with their sentiment reduction
    "negative": {
        "bearish": -0.3,
        "loss": -0.25, 
        "decline": -0.2, 
        "recession": -0.3, 
        "debt": -0.2, 
        "layoff": -0.25, 
        "bankruptcy": -0.4,
        "downgrade": -0.25,
        "selloff": -0.3,
        "crash": -0.4,
        "crisis": -0.3,
        "lawsuit": -0.2,
        "investigation": -0.2
    }
}

# Negation words that can invert sentiment
NEGATION_WORDS = {
    "not", "no", "never", "neither", "hardly", "scarcely"
}

# Context modifiers for financial text
CONTEXT_MODIFIERS = {
    "potential": 0.1,   # Slightly reduces extreme sentiments
    "possibly": 0.1,
    "might": 0.1,
    "could": 0.1,
    "unlikely": -0.1,   # Slightly reduces positive sentiments
    "uncertain": -0.1
}

class FinancialSentimentAnalyzer:
    def __init__(self):
        """
        Initialize multiple sentiment analysis models for robust analysis.
        """
        try:
            # Financial sentiment model (FinBERT)
            self.finbert = pipeline(
                "sentiment-analysis", 
                model="ProsusAI/finbert",
                top_k=None,
                truncation=True,
                max_length=512
            )
            
            # General sentiment model (RoBERTa)
            self.roberta = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment",
                top_k=None,
                truncation=True,
                max_length=512
            )
            
            logger.info("Financial Sentiment Analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {e}")
            self.finbert = None
            self.roberta = None
    
    def preprocess_text(self, text: str, max_length: int = 512) -> str:
        """
        Preprocess text for sentiment analysis with length normalization.
        
        Args:
            text (str): Input text
            max_length (int): Maximum token length for models
        
        Returns:
            str: Preprocessed text
        """
        # Handle None or empty input
        if not text:
            return ""
        
        # Convert to string (in case of non-string input)
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters, keep apostrophes for contractions
        text = re.sub(r'[^a-zA-Z\'\s]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip().lower()
        
        # Truncate to maximum length
        tokens = text.split()
        tokens = tokens[:max_length]
        
        return ' '.join(tokens)
    
    def extract_lexicon_sentiment(self, text: str) -> float:
        """
        Extract sentiment based on financial lexicon.
        
        Args:
            text (str): Preprocessed text
        
        Returns:
            float: Lexicon-based sentiment score
        """
        words = text.split()
        sentiment_score = 0.0
        is_negated = False
        
        for word in words:
            # Check for negation
            if word in NEGATION_WORDS:
                is_negated = not is_negated

            # Check positive lexicon
            if word in FINANCIAL_SENTIMENT_LEXICON["positive"]:
                sentiment_score += (
                    -FINANCIAL_SENTIMENT_LEXICON["positive"][word] 
                    if is_negated 
                    else FINANCIAL_SENTIMENT_LEXICON["positive"][word]
                )

            # Check negative lexicon
            if word in FINANCIAL_SENTIMENT_LEXICON["negative"]:
                sentiment_score += (
                    -FINANCIAL_SENTIMENT_LEXICON["negative"][word] 
                    if is_negated 
                    else FINANCIAL_SENTIMENT_LEXICON["negative"][word]
                )

            # Check context modifiers
            if word in CONTEXT_MODIFIERS:
                sentiment_score *= (1 + CONTEXT_MODIFIERS[word])
        
        return np.clip(sentiment_score, -1, 1)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Perform multi-model sentiment analysis with improved preprocessing.
        
        Args:
            text (str): Text to analyze
        
        Returns:
            Dict[str, Any]: Comprehensive sentiment analysis results
        """
        try:
            # Preprocess text with specific model constraints
            processed_text = self.preprocess_text(text)
            
            # If text is too short after preprocessing, return neutral sentiment
            if len(processed_text.split()) < 5:
                return {
                    "score": 0.0,
                    "label": "neutral",
                    "models": {
                        "finbert": {"score": 0.0, "label": "neutral"},
                        "roberta": {"score": 0.0, "label": "neutral"},
                        "lexicon": {"score": 0.0}
                    }
                }
            
            # Initialize results dictionary
            sentiment_results = {
                "score": 0.0,
                "label": "neutral",
                "models": {}
            }
            
            # Safely handle potential model processing errors
            model_results = []
            
            # Analyze with FinBERT
            if self.finbert:
                try:
                    finbert_result = self.finbert(processed_text)[0]
                    finbert_sentiment = {
                        r['label']: r['score'] for r in finbert_result
                    }
                    finbert_score = (
                        finbert_sentiment.get('positive', 0) - 
                        finbert_sentiment.get('negative', 0)
                    )
                    sentiment_results["models"]["finbert"] = {
                        "score": finbert_score,
                        "label": (
                            "positive" if finbert_score > 0.1 else 
                            "negative" if finbert_score < -0.1 else "neutral"
                        )
                    }
                    model_results.append(finbert_score)
                except Exception as finbert_err:
                    logger.warning(f"FinBERT sentiment error: {finbert_err}")
            
            # Analyze with RoBERTa
            if self.roberta:
                try:
                    roberta_result = self.roberta(processed_text)[0]
                    roberta_sentiment = {
                        r['label']: r['score'] for r in roberta_result
                    }
                    roberta_score = (
                        roberta_sentiment.get('POSITIVE', 0) - 
                        roberta_sentiment.get('NEGATIVE', 0)
                    )
                    sentiment_results["models"]["roberta"] = {
                        "score": roberta_score,
                        "label": (
                            "positive" if roberta_score > 0.1 else 
                            "negative" if roberta_score < -0.1 else "neutral"
                        )
                    }
                    model_results.append(roberta_score)
                except Exception as roberta_err:
                    logger.warning(f"RoBERTa sentiment error: {roberta_err}")
            
            # Lexicon-based sentiment analysis
            try:
                lexicon_score = self.extract_lexicon_sentiment(processed_text)
                sentiment_results["models"]["lexicon"] = {
                    "score": lexicon_score
                }
                model_results.append(lexicon_score)
            except Exception as lexicon_err:
                logger.warning(f"Lexicon sentiment error: {lexicon_err}")
            
            # Combine model scores
            if model_results:
                # Weighted average with more emphasis on financial models
                weights = {
                    "finbert": 0.4,  # Most importance to financial model
                    "roberta": 0.3,
                    "lexicon": 0.3
                }
                
                # Dynamically adjust weights based on available models
                total_weight = sum(weights.get(name, 0.33) for name in sentiment_results["models"])
                
                weighted_score = sum(
                    model_results[i] * weights.get(list(sentiment_results["models"].keys())[i], 0.33) / total_weight
                    for i in range(len(model_results))
                )
                
                # Normalize and clip the final score
                sentiment_results["score"] = np.clip(weighted_score, -1, 1)
                
                # Determine overall label
                if sentiment_results["score"] > 0.2:
                    sentiment_results["label"] = "positive"
                elif sentiment_results["score"] < -0.2:
                    sentiment_results["label"] = "negative"
                else:
                    sentiment_results["label"] = "neutral"
            
            return sentiment_results
        
        except Exception as e:
            logger.error(f"Comprehensive sentiment analysis error: {e}")
            return {
                "score": 0.0,
                "label": "neutral",
                "models": {}
            }

def get_enhanced_sentiment(text: str) -> Optional[float]:
    """
    Public function for enhanced sentiment analysis.
    
    Args:
        text (str): Text to analyze
    
    Returns:
        Optional[float]: Sentiment score, or None if analysis fails
    """
    try:
        analyzer = FinancialSentimentAnalyzer()
        result = analyzer.analyze_sentiment(text)
        return result.get('score')
    except Exception as e:
        logger.error(f"Error in enhanced sentiment analysis: {e}")
        return None

# Create a global instance for easy import
FINANCIAL_SENTIMENT = FinancialSentimentAnalyzer()    
def preprocess_text(self, text: str, max_length: int = 512, min_length: int = 10) -> str:
        """
        Preprocess text for sentiment analysis with length normalization.
        
        Args:
            text (str): Input text
            max_length (int): Maximum token length for models
            min_length (int): Minimum length for valid text
        
        Returns:
            str: Preprocessed text
        """
        # Handle None or empty input
        if not text:
            return ""
        
        # Convert to string (in case of non-string input)
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters, keep apostrophes for contractions
        text = re.sub(r'[^a-zA-Z\'\s]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip().lower()
        
        # Truncate or pad to ensure consistent length
        tokens = text.split()
        
        # Check minimum length
        if len(tokens) < min_length:
            # Pad with neutral words if too short
            padding = ['the', 'and', 'is', 'in', 'of'] * ((min_length // 5) + 1)
            tokens.extend(padding[:min_length - len(tokens)])
        
        # Truncate if too long
        tokens = tokens[:max_length]
        
        return ' '.join(tokens)
    
def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Perform multi-model sentiment analysis with improved preprocessing.
        
        Args:
            text (str): Text to analyze
        
        Returns:
            Dict[str, Any]: Comprehensive sentiment analysis results
        """
        try:
            # Preprocess text with specific model constraints
            processed_text = self.preprocess_text(text)
            
            # If text is too short after preprocessing, return neutral sentiment
            if len(processed_text.split()) < 5:
                return {
                    "score": 0.0,
                    "label": "neutral",
                    "models": {
                        "finbert": {"score": 0.0, "label": "neutral"},
                        "roberta": {"score": 0.0, "label": "neutral"},
                        "lexicon": {"score": 0.0}
                    }
                }
            
            # Initialize results dictionary
            sentiment_results = {
                "score": 0.0,
                "label": "neutral",
                "models": {}
            }
            
            # Safely handle potential model processing errors
            model_results = []
            
            # Analyze with FinBERT
            if self.finbert:
                try:
                    finbert_result = self.finbert(processed_text)[0]
                    finbert_sentiment = {
                        r['label']: r['score'] for r in finbert_result
                    }
                    finbert_score = (
                        finbert_sentiment.get('positive', 0) - 
                        finbert_sentiment.get('negative', 0)
                    )
                    sentiment_results["models"]["finbert"] = {
                        "score": finbert_score,
                        "label": (
                            "positive" if finbert_score > 0.1 else 
                            "negative" if finbert_score < -0.1 else "neutral"
                        )
                    }
                    model_results.append(finbert_score)
                except Exception as finbert_err:
                    logger.warning(f"FinBERT sentiment error: {finbert_err}")
            
            # Analyze with RoBERTa
            if self.roberta:
                try:
                    roberta_result = self.roberta(processed_text)[0]
                    roberta_sentiment = {
                        r['label']: r['score'] for r in roberta_result
                    }
                    roberta_score = (
                        roberta_sentiment.get('POSITIVE', 0) - 
                        roberta_sentiment.get('NEGATIVE', 0)
                    )
                    sentiment_results["models"]["roberta"] = {
                        "score": roberta_score,
                        "label": (
                            "positive" if roberta_score > 0.1 else 
                            "negative" if roberta_score < -0.1 else "neutral"
                        )
                    }
                    model_results.append(roberta_score)
                except Exception as roberta_err:
                    logger.warning(f"RoBERTa sentiment error: {roberta_err}")
            
            # Lexicon-based sentiment analysis
            try:
                lexicon_score = self.extract_lexicon_sentiment(processed_text)
                sentiment_results["models"]["lexicon"] = {
                    "score": lexicon_score
                }
                model_results.append(lexicon_score)
            except Exception as lexicon_err:
                logger.warning(f"Lexicon sentiment error: {lexicon_err}")
            
            # Combine model scores
            if model_results:
                # Weighted average with more emphasis on financial models
                weights = {
                    "finbert": 0.4,  # Most importance to financial model
                    "roberta": 0.3,
                    "lexicon": 0.3
                }
                
                # Dynamically adjust weights based on available models
                total_weight = sum(weights.get(name, 0.33) for name in sentiment_results["models"])
                
                weighted_score = sum(
                    model_results[i] * weights.get(list(sentiment_results["models"].keys())[i], 0.33) / total_weight
                    for i in range(len(model_results))
                )
                
                # Normalize and clip the final score
                sentiment_results["score"] = np.clip(weighted_score, -1, 1)
                
                # Determine overall label
                if sentiment_results["score"] > 0.2:
                    sentiment_results["label"] = "positive"
                elif sentiment_results["score"] < -0.2:
                    sentiment_results["label"] = "negative"
                else:
                    sentiment_results["label"] = "neutral"
            
            return sentiment_results
        
        except Exception as e:
            logger.error(f"Comprehensive sentiment analysis error: {e}")
            return {
                "score": 0.0,
                "label": "neutral",
                "models": {}
            }