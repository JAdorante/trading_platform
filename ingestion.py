import praw
import requests
import logging
import os
import json
import concurrent.futures
import spacy
import numpy as np
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from transformers import pipeline
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from bertopic import BERTopic
from gensim import corpora
from gensim.models import LdaModel
import xgboost as xgb
import joblib
import pickle
import time
import re
import threading
from dotenv import load_dotenv
from database import save_raw_data, save_score, save_price_prediction, save_technical_indicators, save_topic, save_entities
from filtering import keyword_pre_scan, STOCKS

# Import NLTK and download required resources
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Import enhanced components (new)
try:
    from enhanced_entity_recognition import FINANCIAL_NER, enhanced_stock_detection
    entity_recognition_available = True
except ImportError:
    entity_recognition_available = False

try:
    from sentiment_analysis_refinement import FINANCIAL_SENTIMENT, get_enhanced_sentiment
    sentiment_analysis_available = True
except ImportError:
    sentiment_analysis_available = False

try:
    from data_quality_filters import CONTENT_FILTER, filter_quality_content
    content_filter_available = True
except ImportError:
    content_filter_available = False

try:
    from enhanced_topic_modeling import TOPIC_MODELER, analyze_document_topics
    topic_modeling_available = True
except ImportError:
    topic_modeling_available = False

try:
    from integration import ENHANCED_INTEGRATION
    integration_available = True
except ImportError:
    integration_available = False

# IMPORTANT: Enable integration component - previously disabled
integration_available = True  # Change from False to True

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Detailed logging for troubleshooting
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log", mode='a'),  # 'w' mode to overwrite log each run
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ingestion")

# Configuration options for enhancements
USE_ENHANCED_NER = os.environ.get("USE_ENHANCED_NER", "true").lower() == "true"
USE_ENHANCED_SENTIMENT = os.environ.get("USE_ENHANCED_SENTIMENT", "true").lower() == "true"
USE_CONTENT_FILTERING = os.environ.get("USE_CONTENT_FILTERING", "true").lower() == "true"
USE_TOPIC_MODELING = os.environ.get("USE_TOPIC_MODELING", "true").lower() == "true"
MIN_QUALITY_SCORE = float(os.environ.get("MIN_QUALITY_SCORE", "0.3"))

# Add enhanced_ingestion_processor function directly to this file
def enhanced_ingestion_processor(submission, source):
    """
    Enhanced processing with additional quality checks.
    
    Args:
        submission: Reddit submission object
        source: Source of the submission
    
    Returns:
        int: Number of saved documents
    """
    try:
        # Preprocess text
        text = (submission.title + " " + (submission.selftext or "")).strip()
        
        # Content quality filtering
        if content_filter_available and USE_CONTENT_FILTERING:
            quality_result = filter_quality_content(text, source, MIN_QUALITY_SCORE)
            if not quality_result.get('is_quality_content', False):
                logger.debug(f"Filtered low-quality content from {source}")
                return 0
        
        # Enhanced entity recognition
        tickers = []
        if entity_recognition_available and USE_ENHANCED_NER:
            tickers = enhanced_stock_detection(text) or []
        else:
            tickers = keyword_pre_scan(text) or []
        
        # Sentiment analysis
        sentiment_score = 0
        if sentiment_analysis_available and USE_ENHANCED_SENTIMENT:
            sentiment_score = get_enhanced_sentiment(text) or 0
        else:
            sentiment_score = get_sentiment(text)
        
        # Save and process each detected ticker
        saved_count = 0
        if tickers:
            for ticker in tickers:
                try:
                    # Save raw data
                    document_id = save_raw_data(source, text, "keyword_match", ticker, submission.url)
                    
                    # Extract entities if enabled
                    if entity_recognition_available and USE_ENHANCED_NER and document_id:
                        entities = FINANCIAL_NER.extract_entities(text)
                        save_entities(document_id, entities)
                    
                    # Save sentiment
                    save_score(ticker, sentiment_score, f"{source} submission {submission.id}")
                    
                    # Extract topics if enabled
                    if topic_modeling_available and USE_TOPIC_MODELING and document_id:
                        analyze_document_topics(document_id, text)
                    
                    saved_count += 1
                    logger.debug(f"Processed submission {submission.id} for ticker {ticker}")
                except Exception as ticker_error:
                    logger.error(f"Error processing ticker {ticker}: {ticker_error}")
        
        return saved_count
    
    except Exception as e:
        logger.error(f"Error in enhanced processing: {e}")
        return 0

def load_robust_nlp_models():
    """
    Load NLP models with comprehensive error handling and fallback.
    
    Returns:
        Tuple of (FinBERT model, RoBERTa model, SpaCy model)
    """
    try:
        logger.info("Starting comprehensive NLP model loading...")
        
        # FinBERT Sentiment Model
        try:
            finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            logger.info("FinBERT model loaded successfully")
        except Exception as finbert_err:
            logger.error(f"FinBERT model loading failed: {finbert_err}")
            finbert = None
        
        # RoBERTa Sentiment Model
        try:
            roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
            logger.info("RoBERTa model loaded successfully")
        except Exception as roberta_err:
            logger.error(f"RoBERTa model loading failed: {roberta_err}")
            roberta = None
        
        # Fallback Sentiment Analysis
        def fallback_sentiment(text):
            """Simple rule-based sentiment analysis as fallback"""
            positive_words = ['good', 'great', 'positive', 'buy', 'growth', 'increase']
            negative_words = ['bad', 'poor', 'negative', 'sell', 'decline', 'loss']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return [{'label': 'positive', 'score': pos_count / (pos_count + neg_count)}]
            elif neg_count > pos_count:
                return [{'label': 'negative', 'score': neg_count / (pos_count + neg_count)}]
            else:
                return [{'label': 'neutral', 'score': 0.5}]
        
        # SpaCy NER Model
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully")
        except Exception as spacy_err:
            logger.error(f"SpaCy model loading failed: {spacy_err}")
            nlp = None
        
        # Ensure fallback for any failed models
        finbert = finbert or fallback_sentiment
        roberta = roberta or fallback_sentiment
        
        return finbert, roberta, nlp
    
    except Exception as e:
        logger.critical(f"Comprehensive NLP model loading error: {e}")
        return None, None, None

# Load NLP models with robust loading
FINBERT, ROBERTA, NLP = load_robust_nlp_models()

# Ensure models are loaded before proceeding
if not (FINBERT and ROBERTA and NLP):
    logger.critical("Critical: One or more NLP models failed to load. Ingestion may be impaired.")

# Named Entity Recognition function
def extract_entities(text):
    """Extract organizations, products, and other entities from text."""
    # Use enhanced entity recognition if available
    if entity_recognition_available and USE_ENHANCED_NER:
        return FINANCIAL_NER.extract_entities(text)
    
    # Fall back to original implementation
    try:
        doc = NLP(text)
        entities = {
            "ORG": [],
            "PRODUCT": [],
            "PERSON": [],
            "GPE": [],
            "MONEY": []
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        return entities
    except Exception as e:
        logger.error(f"Error in entity extraction: {e}")
        return {
            "ORG": [],
            "PRODUCT": [],
            "PERSON": [],
            "GPE": [],
            "MONEY": []
        }

# Sentiment Analysis function
def get_sentiment(text):
    """Enhanced sentiment analysis using multiple models."""
    # Use enhanced sentiment analysis if available
    if sentiment_analysis_available and USE_ENHANCED_SENTIMENT:
        return get_enhanced_sentiment(text)
    
    # Fall back to original implementation
    try:
        # Preprocessing
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Early return for very short texts
        if len(text) < 5:
            return 0
        
        # Sentiment calculation
        sentiments = []
        
        # FinBERT Sentiment
        try:
            finbert_result = FINBERT(text)[0]
            finbert_score = finbert_result['score'] if finbert_result['label'] == 'positive' else -finbert_result['score']
            sentiments.append(finbert_score)
        except Exception as finbert_err:
            logger.error(f"FinBERT sentiment error: {finbert_err}")
        
        # RoBERTa Sentiment
        try:
            roberta_result = ROBERTA(text)[0]
            roberta_score = (roberta_result['score'] if roberta_result['label'] == 'POSITIVE'
                             else -roberta_result['score'] if roberta_result['label'] == 'NEGATIVE'
                             else 0)
            sentiments.append(roberta_score * 0.8)
        except Exception as roberta_err:
            logger.error(f"RoBERTa sentiment error: {roberta_err}")
        
        # If no sentiments calculated, return neutral
        if not sentiments:
            return 0
        
        # Context modifiers
        words = set(text.lower().split())
        context_multiplier = 1.0
        
        NEGATION_WORDS = {"not", "no", "never", "sell", "dump", "short", "bearish", "fail"}
        STRONG_POSITIVE = {"buying", "bullish", "strong", "soar", "surge", "rally", "breakout"}
        STRONG_NEGATIVE = {"crash", "plummet", "bankruptcy", "selloff", "recession", "collapse"}
        
        if words & NEGATION_WORDS:
            context_multiplier *= 0.7
        if words & STRONG_POSITIVE:
            context_multiplier *= 1.3
        if words & STRONG_NEGATIVE:
            context_multiplier *= 0.7
        
        # Final sentiment calculation
        return np.mean(sentiments) * context_multiplier
    
    except Exception as e:
        logger.error(f"Comprehensive sentiment analysis error: {e}")
        return 0

# Reddit API Credentials
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "JjRoAQp_Y9AjKEoFeWDD2w")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "7voo4yB7D-zo3yeYwBxG_A8NKiyPIA")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "python:Sentiment Scraper 2.0:v1.0 (by u/Substantial_Staff199)")

# Initialize Reddit API client
def initialize_reddit_client():
    """Initialize Reddit API client with detailed error handling."""
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        # Verify connection by accessing a simple attribute
        reddit.read_only = True
        reddit.subreddits.popular(limit=1)
        logger.info("Reddit API client initialized successfully")
        return reddit
    except Exception as e:
        logger.error(f"Failed to initialize Reddit API client: {e}")
        return None

# Global Reddit client
REDDIT_CLIENT = initialize_reddit_client()

# Subreddits and collection parameters
SUBREDDITS = [
   "stocks", "investing", "wallstreetbets", "finance", "personalfinance",
    "pennystocks", "StockMarket", "options", "dividends", "securityanalysis",
    "algotrading", "cryptocurrency", "financialindependence", "RobinHood",
    "Daytrading", "ValueInvesting", "Forex", "Economics", "Trading", "ETFs",
    "Stock_Picks", "InvestmentClub", "Wealth", "fidelity", "Thetagang",
    "SmallStreetBets", "CanadianInvestor", "UKInvesting", "RealEstateInvesting",
    "Investing101", "CryptoMarkets", "StockAnalysis", "PassiveIncome",
    "QuantitativeFinance", "FuturesTrading", "SwingTrading", "IndiaInvestments",
    "AusFinance", "EuroStocks", "Commodities", "SPACs", "Retirement",
    "Investors", "FinancialPlanning", "WallStForMainSt", "StockTrading",
    "Portfolio", "Blockchain", "MarketAnalysis", "Money"

]
POST_LIMIT = 100
SEMAPHORE = threading.Semaphore(3)

# Process a single submission
def process_submission(submission, subreddit_name):
    """Separate function to process a single submission."""
    try:
        text = (submission.title + " " + (submission.selftext or "")).strip()
        if len(text) < 50:
            return 0
        
        # Apply content quality filter if available
        if content_filter_available and USE_CONTENT_FILTERING:
            filter_result = filter_quality_content(text, "Reddit", MIN_QUALITY_SCORE)
            if not filter_result["is_quality_content"]:
                logger.debug(f"Filtered content: {filter_result['reason']}")
                return 0
        
        # Detect tickers - use enhanced if available
        if entity_recognition_available and USE_ENHANCED_NER:
            tickers = enhanced_stock_detection(text)
        else:
            tickers = keyword_pre_scan(text)
        
        saved_count = 0
        if tickers:
            for ticker in tickers:
                try:
                    # Save raw data
                    document_id = save_raw_data("Reddit", text, "keyword_match", ticker, submission.url)
                    
                    # Extract entities if enabled
                    if entity_recognition_available and USE_ENHANCED_NER and document_id:
                        entities = FINANCIAL_NER.extract_entities(text)
                        save_entities(document_id, entities)
                    
                    # Analyze sentiment
                    score = get_sentiment(text)
                    save_score(ticker, score, f"Reddit submission {submission.id}")
                    
                    # Extract topics if enabled
                    if topic_modeling_available and USE_TOPIC_MODELING and document_id:
                        analyze_document_topics(document_id, text)
                    
                    saved_count += 1
                    logger.debug(f"Processed submission {submission.id} for ticker {ticker}")
                except Exception as ticker_error:
                    logger.error(f"Error processing ticker {ticker}: {ticker_error}")
        
        return saved_count
    except Exception as e:
        logger.error(f"Error processing submission: {e}")
        return 0

# Enhanced version of process_subreddit - using integration if available
def process_subreddit(subreddit_name):
    """Process a single subreddit with enhanced capabilities."""
    saved_count = 0
    try:
        logger.info(f"Processing subreddit: {subreddit_name}")
        
        # Log ENHANCED_INTEGRATION details if available
        if integration_available:
            logger.info(f"ENHANCED_INTEGRATION attributes: {dir(ENHANCED_INTEGRATION)}")
        
        # Add timeout for subreddit processing
        start_time = time.time()
        max_subreddit_time = 60  # 60 seconds max per subreddit
        
        # Semaphore to limit concurrent subreddit processing
        with SEMAPHORE:
            subreddit = REDDIT_CLIENT.subreddit(subreddit_name)
            
            # Fetch submissions
            try:
                submissions = list(subreddit.new(limit=POST_LIMIT))
                logger.info(f"Found {len(submissions)} submissions in {subreddit_name}")
            except Exception as fetch_error:
                logger.error(f"Error fetching submissions from {subreddit_name}: {fetch_error}")
                return 0
            
            # Process each submission
            for submission in submissions:
                # Check for timeout
                if time.time() - start_time > max_subreddit_time:
                    logger.warning(f"Processing time limit reached for {subreddit_name}")
                    break
                    
                if integration_available:
                    try:
                        # Use the enhanced_ingestion_processor directly from this module
                        processed = enhanced_ingestion_processor(submission, subreddit_name)
                        saved_count += processed
                    except (AttributeError, NameError) as e:
                        logger.error(f"Enhanced processing error: {e}. Falling back to original processing.")
                        # Fall back to original processing logic
                        saved_count += process_submission(submission, subreddit_name)
                else:
                    # Original processing logic
                    saved_count += process_submission(submission, subreddit_name)
            
            logger.info(f"Subreddit {subreddit_name} saved {saved_count} items")
            return saved_count
        
    except Exception as subreddit_error:
        logger.error(f"Comprehensive error in subreddit {subreddit_name}: {subreddit_error}")
        return 0

def fetch_reddit_parallel():
    """Robust parallel Reddit data collection with comprehensive error handling."""
    if not REDDIT_CLIENT:
        logger.critical("Reddit client not initialized. Aborting data collection.")
        return 0
    
    logger.info(f"Starting parallel Reddit data collection from {len(SUBREDDITS)} subreddits")
    total_saved = 0
    start_time = time.time()
    
    # Process fewer subreddits if we have too many
    subreddits_to_process = SUBREDDITS[:50] if len(SUBREDDITS) > 50 else SUBREDDITS
    logger.info(f"Processing {len(subreddits_to_process)} of {len(SUBREDDITS)} subreddits to avoid timeout")
    
    # Parallel processing with timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(subreddits_to_process))) as executor:
        # Submit tasks for each subreddit
        future_to_subreddit = {
            executor.submit(process_subreddit, subreddit): subreddit 
            for subreddit in subreddits_to_process
        }
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_subreddit):
            subreddit = future_to_subreddit[future]
            try:
                count = future.result(timeout=90)  # 90-second timeout per subreddit
                total_saved += count
                logger.info(f"Subreddit {subreddit} saved {count} items")
            except concurrent.futures.TimeoutError:
                logger.error(f"Subreddit {subreddit} processing timed out")
            except Exception as e:
                logger.error(f"Error processing results for {subreddit}: {e}")
    
    # Final summary
    elapsed = time.time() - start_time
    logger.info(f"""
    Reddit Data Collection Summary:
    - Total Subreddits Processed: {len(subreddits_to_process)}
    - Total Submissions Saved: {total_saved}
    - Time Elapsed: {elapsed:.2f} seconds
    """)
    
    return total_saved

def run_ingestion():
    """Run data ingestion tasks with timeout and error handling."""
    try:
        logger.info("Starting data ingestion process...")
        
        # Increase the timeout for Reddit API calls
        import socket
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(60)  # 60 seconds timeout for API calls
        
        # Reduce POST_LIMIT to process fewer posts per subreddit
        global POST_LIMIT
        original_post_limit = POST_LIMIT
        POST_LIMIT = min(POST_LIMIT, 100)  # Limit to 5 posts per subreddit for faster processing
        
        # Set an overall timeout for the entire ingestion process
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fetch_reddit_parallel)
            try:
                total_saved = future.result(timeout=600)  # 10-minute timeout (increased from 5)
                logger.info(f"Ingestion completed. Total submissions processed: {total_saved}")
            except concurrent.futures.TimeoutError:
                logger.error("Ingestion process timed out after 10 minutes")
            except Exception as e:
                logger.error(f"Ingestion error: {e}")
            finally:
                # Restore original settings
                socket.setdefaulttimeout(old_timeout)
                POST_LIMIT = original_post_limit
    except Exception as e:
        logger.error(f"Comprehensive ingestion error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Apply enhancements if available
    if integration_available:
        try:
            ENHANCED_INTEGRATION.patch_ingestion_functions()
            logger.info("Applied enhancements to ingestion functions")
        except Exception as e:
            logger.error(f"Error applying enhancements: {e}")
    
    run_ingestion()