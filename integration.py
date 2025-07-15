import logging
import os
import threading
import schedule
import time
from datetime import datetime, timedelta
import sqlite3
import json
import pickle
import pandas as pd
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
logger = logging.getLogger("system_integration")

# Flexible import handling
def safe_import(module_name, default=None):
    """
    Safely import a module with optional default return.
    
    Args:
        module_name (str): Name of the module to import
        default (any, optional): Default value if import fails
    
    Returns:
        Imported module or default value
    """
    try:
        return __import__(module_name)
    except ImportError as e:
        logger.warning(f"Could not import {module_name}: {e}")
        return default

# Import modules with fallback
database = safe_import('database')
model = safe_import('model')

class SystemIntegration:
    def __init__(self):
        """
        Initialize system-wide integration capabilities.
        """
        # Fallback if model import fails
        self.model = model.StockAnalysisModel() if model and hasattr(model, 'StockAnalysisModel') else None
        self.last_system_health_check = None
        self.integration_lock = threading.Lock()
    
    def patch_ingestion_functions(self):
        """
        Apply enhancements to ingestion pipeline dynamically.
        """
        try:
            logger.info("Applying advanced ingestion enhancements")
            
            # Dynamic import to avoid circular references
            import importlib
            
            try:
                ingestion = importlib.import_module('ingestion')
            except ImportError:
                logger.error("Could not import ingestion module")
                return False
            
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
                    # Import additional modules dynamically
                    data_quality_filter = safe_import('data_quality_filters')
                    entity_recognition = safe_import('enhanced_entity_recognition')
                    sentiment_analysis = safe_import('sentiment_analysis_refinement')
                    topic_modeling = safe_import('enhanced_topic_modeling')
                    
                    # Preprocess text
                    text = (submission.title + " " + (submission.selftext or "")).strip()
                    
                    # Content quality filtering
                    if data_quality_filter:
                        quality_result = data_quality_filter.filter_quality_content(text, source)
                        if not quality_result.get('is_quality_content', False):
                            logger.debug(f"Filtered low-quality content from {source}")
                            return 0
                    
                    # Enhanced entity recognition
                    tickers = []
                    if entity_recognition:
                        tickers = entity_recognition.enhanced_stock_detection(text) or []
                    
                    # Sentiment analysis
                    sentiment_score = 0
                    if sentiment_analysis:
                        sentiment_result = sentiment_analysis.get_enhanced_sentiment(text)
                        sentiment_score = sentiment_result if sentiment_result is not None else 0
                    
                    # Save and process each detected ticker
                    saved_count = 0
                    for ticker in tickers:
                        try:
                            # Save raw data
                            if database:
                                document_id = database.save_raw_data(
                                    source=source, 
                                    text=text, 
                                    keyword=ticker, 
                                    ticker=ticker
                                )
                                
                                # Save sentiment
                                if document_id:
                                    database.save_score(
                                        ticker=ticker, 
                                        score=sentiment_score, 
                                        explanation=f"Reddit submission from {source}"
                                    )
                                
                                # Topic modeling
                                if topic_modeling and document_id:
                                    topic_modeling.analyze_document_topics(document_id, text)
                                
                                saved_count += 1
                        except Exception as ticker_error:
                            logger.error(f"Error processing ticker {ticker}: {ticker_error}")
                    
                    return saved_count
                
                except Exception as e:
                    logger.error(f"Comprehensive error in ingestion processing: {e}")
                    return 0
            
            # Attach enhanced processor to ingestion module
            ingestion.enhanced_ingestion_processor = enhanced_ingestion_processor
            
            logger.info("Ingestion pipeline enhanced successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error patching ingestion functions: {e}")
            return False
    
    def run_historical_data_import(self, tickers=None):
        """
        Run comprehensive historical data import and analysis.
        
        Args:
            tickers (List[str], optional): List of tickers to import
        
        Returns:
            dict: Import results
        """
        try:
            # Use default tickers if not provided
            if not tickers:
                tickers_str = os.environ.get("STOCK_TICKERS", "NVDA,AAPL,AMD,TSLA,META,MSFT,GOOGL,AMZN,QQQ,SPY,DIA")
                tickers = [ticker.strip() for ticker in tickers_str.split(",")]
            
            # Import historical data module dynamically
            historical_data = safe_import('historical_data_import')
            
            if not historical_data:
                logger.error("Historical data import module not available")
                return {}
            
            # Track import results
            import_results = {}
            
            for ticker in tickers:
                try:
                    # Get historical data for a longer period
                    start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
                    end_date = datetime.now().strftime("%Y-%m-%d")
                    
                    # Use method from imported module
                    historical_data_func = getattr(historical_data, 'get_historical_price_data', None)
                    
                    if not historical_data_func:
                        logger.error(f"No historical data method found for {ticker}")
                        continue
                    
                    historical_price_data = historical_data_func(
                        ticker, 
                        start_date=start_date, 
                        end_date=end_date
                    )
                    
                    if historical_price_data is not None and not historical_price_data.empty:
                        # Train model on historical data if model is available
                        if self.model:
                            try:
                                self.model.train_model(ticker, retrain=True)
                            except Exception as model_err:
                                logger.error(f"Model training error for {ticker}: {model_err}")
                        
                        import_results[ticker] = {
                            "data_points": len(historical_price_data),
                            "date_range": f"{start_date} to {end_date}",
                            "status": "Success"
                        }
                        
                        logger.info(f"Imported historical data for {ticker}")
                    else:
                        import_results[ticker] = {
                            "status": "No data available",
                            "reason": "Empty or None historical data"
                        }
                
                except Exception as ticker_error:
                    logger.error(f"Error importing data for {ticker}: {ticker_error}")
                    import_results[ticker] = {
                        "status": "Failed",
                        "reason": str(ticker_error)
                    }
            
            return import_results
        
        except Exception as e:
            logger.error(f"Comprehensive historical data import error: {e}")
            return {}
    
    def perform_system_health_check(self, light_mode=False):
        """
        Comprehensive system health and maintenance check.
        
        Performs:
        - Database optimization
        - Old data pruning
        - Backup creation
        - Model retraining check
        
        Args:
            light_mode (bool): If True, skip resource-intensive operations
        """
        try:
            # Use a timeout to avoid deadlocks
            if not self.integration_lock.acquire(blocking=False):
                logger.warning("Could not acquire integration lock for health check - skipping")
                return False
                
            try:
                logger.info("Performing comprehensive system health check")
                
                # Only run intensive operations in full mode
                if not light_mode:
                    # Optimize database if module is available
                    if database and hasattr(database, 'optimize_database'):
                        database.optimize_database()
                    
                    # Prune old data if method exists
                    if database and hasattr(database, 'prune_old_data'):
                        pruned_count = database.prune_old_data(days_to_keep=180)
                        logger.info(f"Pruned {pruned_count} old records")
                    
                    # Create database backup
                    if database and hasattr(database, 'backup_database'):
                        database.backup_database()
                    
                    # Check and retrain models if needed
                    if self.model:
                        tickers_str = os.environ.get("STOCK_TICKERS", "NVDA,AAPL,AMD,TSLA,META,MSFT,GOOGL,AMZN")
                        tickers = [ticker.strip() for ticker in tickers_str.split(",")]
                        
                        for ticker in tickers:
                            # Check if model needs retraining
                            metrics_path = os.path.join("models", f"{ticker}_metrics.json")
                            
                            if os.path.exists(metrics_path):
                                try:
                                    with open(metrics_path, 'r') as f:
                                        metrics = json.load(f)
                                        last_training = datetime.fromisoformat(metrics.get('training_date', '2000-01-01'))
                                        
                                        # Retrain if model is older than 30 days
                                        if (datetime.now() - last_training).days > 30:
                                            logger.info(f"Retraining model for {ticker}")
                                            self.model.train_model(ticker, retrain=True)
                                except Exception as model_load_err:
                                    logger.error(f"Error checking model for {ticker}: {model_load_err}")
                
                # Update last health check timestamp
                self.last_system_health_check = datetime.now()
                
                logger.info("System health check completed successfully")
                
                return True
            finally:
                # Always release the lock
                self.integration_lock.release()
        
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            # Make sure to release the lock even in case of an exception
            if hasattr(self.integration_lock, '_is_owned') and self.integration_lock._is_owned():
                self.integration_lock.release()
            return False
    
    def schedule_periodic_tasks(self):
        """
        Schedule periodic system maintenance and enhancement tasks.
        """
        try:
            # Dynamic import of run_ingestion to avoid circular import
            import importlib
            
            try:
                ingestion = importlib.import_module('ingestion')
                run_ingestion = getattr(ingestion, 'run_ingestion', None)
            except (ImportError, AttributeError):
                logger.error("Could not import run_ingestion function")
                run_ingestion = None
            
            # Daily system health check (1 AM)
            schedule.every().day.at("01:00").do(lambda: self.perform_system_health_check(light_mode=False))
            
            # Periodic data ingestion if function exists
            if run_ingestion:
                schedule.every(30).minutes.do(run_ingestion)
            
            # Periodic topic modeling
            topic_modeling = safe_import('enhanced_topic_modeling')
            if topic_modeling and hasattr(topic_modeling, 'TOPIC_MODELER'):
                schedule.every().day.at("02:00").do(
                    lambda: topic_modeling.TOPIC_MODELER.run_periodic_topic_modeling(days=7, min_documents=100)
                )
            
            # Periodic historical data refresh (weekly)
            schedule.every().week.do(self.run_historical_data_import)
            
            logger.info("Periodic tasks scheduled successfully")
        
        except Exception as e:
            logger.error(f"Error scheduling periodic tasks: {e}")
    
    def run_initial_system_setup(self):
        """
        Perform initial system setup and preparation.
        """
        try:
            logger.info("Running initial system setup")
            
            # Patch ingestion functions
            self.patch_ingestion_functions()
            
            # Import initial historical data
            self.run_historical_data_import()
            
            # Skip health check during startup to avoid hanging
            # self.perform_system_health_check()  # COMMENTED OUT
            
            # Instead, run a light health check that skips intensive operations
            self.perform_system_health_check(light_mode=True)
            
            # Schedule periodic tasks
            self.schedule_periodic_tasks()
            
            logger.info("Initial system setup completed")
        
        except Exception as e:
            logger.error(f"Error in initial system setup: {e}")

# Create global integration instance
ENHANCED_INTEGRATION = SystemIntegration()