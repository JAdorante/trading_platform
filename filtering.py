import re
import logging
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("filtering")

# Check for availability of enhanced components
try:
    from enhanced_entity_recognition import FINANCIAL_NER, enhanced_stock_detection
    entity_recognition_available = True
    logger.info("Enhanced entity recognition module loaded successfully")
except ImportError:
    entity_recognition_available = False
    logger.warning("Enhanced entity recognition module not available")

try:
    from data_quality_filters import CONTENT_FILTER, filter_quality_content
    content_filter_available = True
    logger.info("Content quality filter module loaded successfully")
except ImportError:
    content_filter_available = False
    logger.warning("Content quality filter module not available")

# Stocks with expanded industry terms & common references
STOCKS = {
    "NVDA": ["Nvidia", "NVDA", "AI", "semiconductors", "GPUs", "data centers", "chip stocks"],
    "AAPL": ["Apple", "AAPL", "iPhone", "MacBook", "App Store", "tariffs", "supply chain", "tech stocks"],
    "AMD": ["AMD", "Advanced Micro Devices", "Ryzen", "EPYC", "gaming GPUs", "data center", "chip stocks"],
    "TSLA": ["Tesla", "TSLA", "Elon Musk", "EV", "self-driving", "battery production", "auto stocks"],
    "META": ["Meta", "Facebook", "Instagram", "META", "advertising revenue", "social media stocks", "AI content"],
    "ORCL": ["Oracle", "ORCL", "cloud computing", "enterprise software", "databases", "AI in cloud", "business software"],
    "TSM": ["TSMC", "Taiwan Semiconductor", "TSM", "chip foundry", "wafer fabrication", "fab stocks"],
    "DELL": ["Dell", "DELL", "PC market", "enterprise sales", "server demand", "hardware stocks", "cloud computing"],
    "AVGO": ["Broadcom", "AVGO", "5G", "semiconductor supply chain", "network chips", "wireless technology"],
    "ADBE": ["Adobe", "ADBE", "Photoshop", "creative software", "digital marketing", "AI tools"],
    "MSFT": ["Microsoft", "MSFT", "Windows", "Office", "Azure", "cloud services", "enterprise software"],
    "GOOGL": ["Google", "Alphabet", "GOOGL", "search engine", "advertising", "Android", "AI research"],
    "AMZN": ["Amazon", "AMZN", "e-commerce", "AWS", "cloud services", "logistics", "online retail"],
    "INTC": ["Intel", "INTC", "processors", "CPU", "chip manufacturing", "semiconductors", "Xeon"],
    "IBM": ["IBM", "International Business Machines", "enterprise", "mainframe", "cloud", "consulting"],
    "CRM": ["Salesforce", "CRM", "SaaS", "cloud software", "enterprise", "customer relationship"],
    "NFLX": ["Netflix", "NFLX", "streaming", "content", "entertainment", "subscribers", "media"],
    "PYPL": ["PayPal", "PYPL", "fintech", "payments", "digital wallets", "online transactions"],
    "CSCO": ["Cisco", "CSCO", "networking", "infrastructure", "enterprise hardware", "routers"]
}

# Economic Terms That Impact Stocks
ECONOMIC_TERMS = {
    "inflation": ["AAPL", "TSLA", "META", "NVDA", "AMD", "DELL", "MSFT", "AMZN"],
    "interest rates": ["META", "AAPL", "TSLA", "ORCL", "NVDA", "AMD", "MSFT", "AMZN", "PYPL"],
    "national debt": ["AAPL", "TSLA", "META", "NVDA", "DELL", "MSFT", "IBM"],
    "supply chain": ["AAPL", "NVDA", "TSM", "AMD", "DELL", "MSFT", "AMZN", "INTC"],
    "chip shortages": ["NVDA", "AMD", "TSM", "AVGO", "INTC"],
    "EV subsidies": ["TSLA"],
    "advertising revenue": ["META", "GOOGL", "SNAP"],
    "tech layoffs": ["META", "AAPL", "TSLA", "NVDA", "ORCL", "ADBE", "MSFT", "GOOGL", "AMZN"],
    "recession": ["META", "AAPL", "TSLA", "NVDA", "DELL", "ORCL", "MSFT", "AMZN", "IBM", "PYPL"],
    "crypto boom": ["NVDA", "AMD", "TSLA", "PYPL"],
    "GDP": ["AAPL", "TSLA", "NVDA", "META", "MSFT", "AMZN", "IBM"],
    "consumer spending": ["AAPL", "AMZN", "TSLA", "META", "NFLX"],
    "AI investments": ["NVDA", "META", "GOOGL", "MSFT", "AMD", "IBM"],
    "cloud growth": ["MSFT", "AMZN", "GOOGL", "ORCL", "IBM", "CRM"],
    "cybersecurity": ["MSFT", "CSCO", "IBM", "GOOGL"],
    "data privacy": ["META", "GOOGL", "AAPL", "MSFT"],
    "remote work": ["MSFT", "ZOOM", "ADBE", "CRM", "CSCO"],
    "semiconductor policy": ["NVDA", "AMD", "INTC", "TSM", "AVGO"]
}

# Track added stock symbols from extended sources
EXTENDED_STOCKS = {}

def load_extended_stock_symbols():
    """Load extended stock symbols from JSON file."""
    try:
        symbols_path = Path("data/stock_symbols.json")
        if symbols_path.exists():
            with open(symbols_path, 'r') as f:
                symbols = json.load(f)
                logger.info(f"Loaded {len(symbols)} extended stock symbols")
                return symbols
        else:
            # Try to create a basic extended symbols file if we have our core ones
            if STOCKS:
                extended = {ticker: {"name": names[0] if names else ticker, "keywords": names} 
                           for ticker, names in STOCKS.items()}
                os.makedirs(symbols_path.parent, exist_ok=True)
                with open(symbols_path, 'w') as f:
                    json.dump(extended, f, indent=2)
                logger.info(f"Created basic extended stock symbols file with {len(extended)} entries")
                return extended
        return {}
    except Exception as e:
        logger.error(f"Error loading extended stock symbols: {e}")
        return {}

# Load extended stock symbols
EXTENDED_STOCKS = load_extended_stock_symbols()

def keyword_pre_scan(text):
    """
    Scan text for stock tickers, company names, and economic terms using regex with word boundaries.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        list: List of detected stock tickers, or None if none detected
    """
    # Use enhanced entity recognition if available
    if entity_recognition_available:
        tickers = enhanced_stock_detection(text)
        if tickers:
            return tickers
    
    # Fall back to basic detection
    text_upper = text.upper()
    detected_stocks = set()

    # Check for stock-specific terms using word-boundary regex matching.
    for ticker, names in STOCKS.items():
        for name in names:
            pattern = r'\b' + re.escape(name.upper()) + r'\b'
            if re.search(pattern, text_upper):
                detected_stocks.add(ticker)
                break  # Once a match is found for this ticker, move on.

    # Check for economic terms and add their associated stocks.
    for term, affected_stocks in ECONOMIC_TERMS.items():
        pattern = r'\b' + re.escape(term.upper()) + r'\b'
        if re.search(pattern, text_upper):
            detected_stocks.update(affected_stocks)

    # Check for cashtags ($AAPL, $MSFT)
    cashtag_pattern = r'\$([A-Z]{1,5})'
    cashtags = re.findall(cashtag_pattern, text_upper)
    for tag in cashtags:
        # Only add if it's a recognized ticker
        if tag in STOCKS or tag in EXTENDED_STOCKS:
            detected_stocks.add(tag)

    return list(detected_stocks) if detected_stocks else None

def calculate_topic_relevance(topics, text):
    """
    Calculate relevance of financial topics to the given text.
    
    Args:
        topics (list): List of topic keywords to check
        text (str): Text to analyze
        
    Returns:
        dict: Dictionary of topics with their relevance scores
    """
    text_lower = text.lower()
    words = set(text_lower.split())
    results = {}
    
    for topic in topics:
        topic_lower = topic.lower()
        if topic_lower in text_lower:
            # Direct mention gets high score
            results[topic] = 1.0
        else:
            # Check for related keywords
            topic_words = topic.lower().split()
            matched_words = sum(1 for word in topic_words if word in words)
            if matched_words > 0:
                results[topic] = matched_words / len(topic_words)
                
    return results

def filter_content_quality(text, min_quality_score=0.3):
    """
    Filter content for quality if quality filter is available.
    
    Args:
        text (str): Text to analyze
        min_quality_score (float): Minimum quality score threshold
        
    Returns:
        dict: Filter results with pass/fail and reason
    """
    if content_filter_available:
        result = filter_quality_content(text, "Unknown", min_quality_score)
        return result
    else:
        # Basic quality check if enhanced filter not available
        if not text or len(text) < 20:
            return {
                "is_quality_content": False,
                "reason": "Text too short",
                "quality_score": 0.0
            }
            
        # Check for excessive special characters or spam patterns
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
        
        if special_char_ratio > 0.3:
            return {
                "is_quality_content": False,
                "reason": "Too many special characters",
                "quality_score": 0.2
            }
            
        if caps_ratio > 0.5 and len(text) > 100:
            return {
                "is_quality_content": False,
                "reason": "Excessive use of capital letters",
                "quality_score": 0.2
            }
            
        # Check for common spam phrases
        spam_phrases = [
            "get rich", "buy now", "100% guaranteed", "limited time", 
            "huge gains", "to the moon", "rocket emoji", "diamond hands"
        ]
        
        text_lower = text.lower()
        for phrase in spam_phrases:
            if phrase in text_lower:
                return {
                    "is_quality_content": False,
                    "reason": f"Contains spam phrase: {phrase}",
                    "quality_score": 0.2
                }
        
        # Default to passing
        return {
            "is_quality_content": True,
            "reason": "Passed basic quality check",
            "quality_score": 0.7
        }

def preprocess_text(text):
    """
    Preprocess text for filtering and analysis.
    
    Args:
        text (str): Text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
        
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove excess whitespace and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\'"]', '', text)
    
    return text

def extract_stock_mentions(text, min_confidence=0.5):
    """
    Extract stock mentions with confidence scores.
    
    Args:
        text (str): Text to analyze
        min_confidence (float): Minimum confidence threshold
        
    Returns:
        dict: Dictionary of stock tickers with confidence scores
    """
    # Use enhanced entity recognition if available
    if entity_recognition_available:
        entities = FINANCIAL_NER.extract_entities(text)
        if "STOCK" in entities and entities["STOCK"]:
            # Create confidence scores based on number of mentions
            stock_counts = {}
            for ticker in entities["STOCK"]:
                stock_counts[ticker] = stock_counts.get(ticker, 0) + 1
                
            # Normalize to get confidence
            max_count = max(stock_counts.values()) if stock_counts else 1
            return {ticker: min(1.0, count / max_count) for ticker, count in stock_counts.items() 
                    if count / max_count >= min_confidence}
    
    # Fall back to basic detection
    tickers = keyword_pre_scan(text)
    if tickers:
        return {ticker: 1.0 for ticker in tickers}
    
    return {}

def is_financial_text(text, threshold=0.3):
    """
    Determine if text is related to financial markets or companies.
    
    Args:
        text (str): Text to analyze
        threshold (float): Minimum threshold for financial relevance
        
    Returns:
        bool: Whether the text is financial in nature
    """
    text_lower = text.lower()
    
    # Financial keywords
    financial_terms = [
        "stock", "market", "investor", "share", "price", "trading", "dividend",
        "earnings", "revenue", "profit", "loss", "quarterly", "fiscal", "forecast",
        "analyst", "portfolio", "investment", "equity", "bond", "etf", "fund",
        "bear", "bull", "rally", "correction", "volatility", "ticker", "exchange",
        "nasdaq", "dow", "s&p", "nyse", "financials", "balance sheet", "growth",
        "valuation", "pe ratio", "market cap", "technical", "fundamental"
    ]
    
    # Company/financial entity keywords
    company_terms = [
        "company", "corporation", "inc", "incorporated", "llc", "business",
        "firm", "enterprise", "startup", "conglomerate", "subsidiary",
        "acquisition", "merger", "ceo", "executive", "board", "industry",
        "sector", "tech", "technology", "financial", "bank"
    ]
    
    # Count matches
    financial_matches = sum(1 for term in financial_terms if term in text_lower)
    company_matches = sum(1 for term in company_terms if term in text_lower)
    
    # Calculate score normalized by text length
    word_count = len(text_lower.split())
    financial_score = (financial_matches + company_matches) / max(1, word_count / 10)
    
    # Check for explicit stock mentions
    stock_mentions = extract_stock_mentions(text)
    if stock_mentions:
        financial_score += 0.3  # Boost score if stocks explicitly mentioned
    
    return financial_score >= threshold

# Initialize stocks from extended source if available
def update_stocks_from_extended():
    """Update STOCKS dictionary with entries from EXTENDED_STOCKS."""
    if EXTENDED_STOCKS:
        for ticker, data in EXTENDED_STOCKS.items():
            if ticker not in STOCKS:
                keywords = data.get("keywords", [])
                if "name" in data and data["name"] and data["name"] not in keywords:
                    keywords.append(data["name"])
                STOCKS[ticker] = keywords
        logger.info(f"Updated STOCKS dictionary to {len(STOCKS)} entries from extended source")

# Update STOCKS if we have extended data
update_stocks_from_extended()