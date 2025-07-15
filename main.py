# main.py
import logging
import time
import sys
import gc
import os
import psutil
import threading
import traceback
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")

class StageManager:
    def __init__(self):
        self.stages = []
        self.current_stage = 0
        self.initialized_components = {}
    
    def add_stage(self, name, init_functions, critical=True):
        """
        Add a stage to the initialization process
        
        Args:
            name (str): Stage name
            init_functions (list): List of (function, args, kwargs) tuples to run
            critical (bool): If True, failure causes system exit
        """
        self.stages.append({
            "name": name,
            "functions": init_functions,
            "critical": critical
        })
    
    def run_stages(self):
        """Run all stages sequentially"""
        total_stages = len(self.stages)
        logger.info(f"Starting sequential initialization ({total_stages} stages)")
        
        for i, stage in enumerate(self.stages):
            self.current_stage = i
            stage_name = stage["name"]
            logger.info(f"Stage {i+1}/{total_stages}: {stage_name}")
            
            success = self._run_stage(stage)
            
            if not success and stage["critical"]:
                logger.critical(f"Critical stage '{stage_name}' failed, exiting")
                sys.exit(1)
            elif not success:
                logger.warning(f"Non-critical stage '{stage_name}' failed, continuing")
    
    def _run_stage(self, stage):
        """Run all functions in a stage"""
        failed_functions = []
        
        for func, args, kwargs in stage["functions"]:
            try:
                func_name = func.__name__
                logger.info(f"Initializing {func_name}")
                start_time = time.time()
                
                # Resolve any lambda arguments with better error handling
                resolved_args = []
                for arg in args:
                    if callable(arg) and not hasattr(arg, '__name__'):
                        # This is likely a lambda function
                        try:
                            result = arg()
                            
                            # Special handling for model-related functions
                            if func_name == 'train_basic_models' and result is None:
                                logger.warning("Lambda returned None for model component, will be handled by train_basic_models")
                                
                            resolved_args.append(result)
                        except Exception as lambda_err:
                            logger.error(f"Error resolving lambda argument: {lambda_err}")
                            logger.error(traceback.format_exc())
                            # Append None instead of failing completely
                            resolved_args.append(None)
                    else:
                        resolved_args.append(arg)
                
                # Execute the function with resolved arguments
                result = func(*resolved_args, **kwargs)
                
                elapsed = time.time() - start_time
                logger.info(f"Successfully initialized {func_name} in {elapsed:.2f}s")
                
                # Store result if needed later - use function name as key
                self.initialized_components[func_name] = result
                
                # Log the component was stored
                logger.debug(f"Stored {func_name} in initialized_components")
            except Exception as e:
                logger.error(f"Failed to initialize {func_name}: {e}")
                logger.error(traceback.format_exc())
                failed_functions.append(func_name)
        
        return len(failed_functions) == 0
    
    def get_component(self, component_name):
        """Get a component by name with fallback logic if it doesn't exist"""
        if component_name in self.initialized_components:
            component = self.initialized_components[component_name]
            logger.debug(f"Found component {component_name} in initialized_components")
            return component
        else:
            logger.warning(f"Component {component_name} not found in initialized_components")
            
            # Try alternative keys (function might be registered under a different name)
            for key, component in self.initialized_components.items():
                if component_name.lower() in key.lower():
                    logger.info(f"Found component {component_name} under key {key}")
                    return component
                    
            # Try to initialize the component directly if possible
            if component_name == "init_model":
                logger.warning("Attempting to initialize model directly")
                try:
                    from model import StockAnalysisModel
                    return StockAnalysisModel(load_models=False)
                except Exception as e:
                    logger.error(f"Failed to initialize model directly: {e}")
            
            return None

def init_database():
    """Initialize database with tables and indexes."""
    from database import init_db
    return init_db()

def init_model():
    """Initialize stock analysis model without loading data."""
    try:
        from model import StockAnalysisModel
        model = StockAnalysisModel(load_models=False)  # Don't load models yet
        
        # Verify the model was created successfully
        if model is None:
            logger.error("StockAnalysisModel initialization returned None")
            return None
            
        # Verify the model has the train_model attribute
        if not hasattr(model, 'train_model'):
            logger.error("StockAnalysisModel does not have train_model method")
            return None
            
        logger.info("StockAnalysisModel initialized successfully with required methods")
        return model
    except ImportError as e:
        logger.error(f"Failed to import StockAnalysisModel: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in init_model: {e}")
        logger.error(traceback.format_exc())
        return None

def init_historical_data(tickers=None):
    """Initialize historical market data for given tickers."""
    try:
        from historical_data_import import HISTORICAL_DATA_IMPORTER
        
        if tickers is None:
            tickers = ['NVDA', 'AAPL', 'AMD', 'TSLA', 'META', 'GOOGL', 'AMZN', 'MSFT','QQQ','SPY','DIA']  # Default tickers
        
        results = {}
        for ticker in tickers:
            try:
                logger.info(f"Importing historical data for {ticker}")
                from datetime import datetime, timedelta
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                
                data = HISTORICAL_DATA_IMPORTER.get_historical_price_data(
                    ticker, 
                    start_date=start_date, 
                    end_date=end_date
                )
                
                if data is not None and not data.empty:
                    results[ticker] = True
                    logger.info(f"Successfully imported historical data for {ticker}")
                else:
                    results[ticker] = False
                    logger.warning(f"No data retrieved for {ticker}")
            except Exception as e:
                logger.error(f"Error importing data for {ticker}: {e}")
                logger.error(traceback.format_exc())
                results[ticker] = False
                
            # Give the system a moment to breathe between tickers
            time.sleep(1)
            gc.collect()  # Explicit garbage collection
        
        return results
    except Exception as e:
        logger.error(f"Error in init_historical_data: {e}")
        logger.error(traceback.format_exc())
        return {}

def train_basic_models(model=None, tickers=None):
    """Train basic models for the specified tickers."""
    if tickers is None:
        tickers = ['NVDA', 'AAPL', 'AMD', 'TSLA', 'META', 'GOOGL', 'AMZN', 'MSFT','QQQ','SPY','DIA']
    
    # If model is None, try to create it
    if model is None:
        logger.warning("Model is None, attempting to create a new instance directly")
        try:
            # Try dynamic import to ensure we get a fresh copy
            import importlib
            import sys
            
            # First, check if model.py exists
            if not os.path.exists("model.py"):
                logger.error("model.py file not found in working directory")
                return []
                
            # Force reload the module if it's already loaded
            if 'model' in sys.modules:
                logger.info("Reloading model module")
                importlib.reload(sys.modules['model'])
            
            # Import the module
            model_module = importlib.import_module('model')
            
            # Check if the module has StockAnalysisModel
            if not hasattr(model_module, 'StockAnalysisModel'):
                logger.error("model module does not have StockAnalysisModel class")
                return []
                
            # Create a new instance
            model = model_module.StockAnalysisModel(load_models=False)
            logger.info("Successfully created a fresh StockAnalysisModel instance")
            
        except Exception as e:
            logger.error(f"Failed to create model instance: {e}")
            logger.error(traceback.format_exc())
            return []
    
    # Verify the model has the train_model method
    if not model:
        logger.error("Model is None after attempted initialization")
        return []
        
    if not hasattr(model, 'train_model'):
        logger.error(f"Model does not have train_model method: {type(model)}")
        # Print model attributes for debugging
        logger.error(f"Model attributes: {dir(model)}")
        return []
        
    if not callable(getattr(model, 'train_model', None)):
        logger.error("train_model attribute exists but is not callable")
        return []
    
    successful_tickers = []
    for ticker in tickers:
        try:
            logger.info(f"Training model for {ticker}")
            
            # Enable more detailed logging during training
            old_level = logger.level
            logger.setLevel(logging.DEBUG)
            
            # Call train_model with explicit try/except
            try:
                success = model.train_model(ticker)
                
                # If training succeeded, analyze feature importance
                if success:
                    successful_tickers.append(ticker)
                    logger.info(f"Successfully trained model for {ticker}")
                    
                    # Analyze feature importance if method exists
                    if hasattr(model, 'analyze_feature_importance') and callable(getattr(model, 'analyze_feature_importance')):
                        logger.info(f"Analyzing feature importance for {ticker}")
                        try:
                            model.analyze_feature_importance(ticker)
                        except Exception as feat_err:
                            logger.error(f"Error analyzing feature importance for {ticker}: {feat_err}")
                            # Continue despite feature importance error
                    else:
                        logger.warning("Feature importance analysis not available - need to add method to model.py")
                else:
                    logger.warning(f"Failed to train model for {ticker}")
                    
            except AttributeError as ae:
                logger.error(f"AttributeError calling train_model: {ae}")
                logger.error(traceback.format_exc())
                break  # Break the loop as this is likely a fundamental issue
            except Exception as train_err:
                logger.error(f"Error in train_model for {ticker}: {train_err}")
                logger.error(traceback.format_exc())
                continue  # Try next ticker
            
            # Reset logging level
            logger.setLevel(old_level)
            
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {e}")
            logger.error(traceback.format_exc())
            
        # Give the system a moment to breathe between models
        time.sleep(1)
        gc.collect()  # Explicit garbage collection
    
    return successful_tickers

def init_integration():
    """Initialize system integration."""
    try:
        from integration import ENHANCED_INTEGRATION
        ENHANCED_INTEGRATION.patch_ingestion_functions()
        return ENHANCED_INTEGRATION
    except Exception as e:
        logger.error(f"Error initializing system integration: {e}")
        logger.error(traceback.format_exc())
        return None

def init_ui():
    """Initialize Flask app and Socket.IO for the web UI."""
    try:
        from ui import app, socketio
        
        # Register routes and other app configurations here
        
        # Add security headers
        @app.after_request
        def add_security_headers(response):
            response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://cdn.socket.io; connect-src 'self' wss: ws:; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; font-src 'self' data: https://cdnjs.cloudflare.com; img-src 'self' data:;"
            return response
        
        # REMOVED feature_importance_api endpoint to avoid conflict with ui.py
        
        return (app, socketio)
    except Exception as e:
        logger.error(f"Error initializing UI: {e}")
        logger.error(traceback.format_exc())
        return (None, None)

def init_nlp_models():
    """Initialize NLP models for sentiment analysis."""
    try:
        # Check available memory first
        mem = psutil.virtual_memory()
        avail_gb = mem.available / (1024**3)
        
        if avail_gb < 4:  # Less than 4GB available
            logger.warning(f"Low memory available ({avail_gb:.2f}GB). Using light-weight NLP models.")
            
        # Import ingestion module
        from ingestion import load_robust_nlp_models
        return load_robust_nlp_models()
    except Exception as e:
        logger.error(f"Error initializing NLP models: {e}")
        logger.error(traceback.format_exc())
        return (None, None, None)

def init_data_collection(start_now=False):
    """Initialize data collection components."""
    try:
        from ingestion import initialize_reddit_client
        reddit_client = initialize_reddit_client()
        
        if start_now and reddit_client:
            from ingestion import run_ingestion
            # Start in a background thread
            threading.Thread(target=run_ingestion, daemon=True).start()
            logger.info("Started background thread for initial data collection")
        
        return reddit_client
    except Exception as e:
        logger.error(f"Error initializing data collection: {e}")
        logger.error(traceback.format_exc())
        return None

def run_db_maintenance():
    """Run periodic database maintenance to prevent locks."""
    while True:
        try:
            logger.info("Running database maintenance")
            from database import optimize_database
            optimize_database()
        except Exception as e:
            logger.error(f"Error in database maintenance: {e}")
        finally:
            # Run every hour
            time.sleep(3600)

# Start maintenance thread
maintenance_thread = threading.Thread(target=run_db_maintenance, daemon=True)
maintenance_thread.start()

# Initialize global stage_manager
stage_manager = None

def main():
    """Main entry point for the trading platform."""
    try:
        logger.info("==================================================")
        logger.info("TRADING PLATFORM INITIALIZATION")
        logger.info("==================================================")
        
        # Log system information
        logger.info(f"Python Version: {sys.version}")
        logger.info(f"Working Directory: {os.getcwd()}")
        logger.info(f"Platform: {sys.platform}")
        
        mem = psutil.virtual_memory()
        logger.info(f"System Memory: Total: {mem.total / (1024**2):.2f} MB, Available: {mem.available / (1024**2):.2f} MB, Used: {mem.percent}%")
        logger.info(f"CPU Usage: {psutil.cpu_percent()}%")
        logger.info(f"Current Process Memory: {psutil.Process().memory_info().rss / (1024**2):.2f} MB")
        
        logger.info("Starting SystemInitializer...")
        
        # Create stage manager
        global stage_manager
        stage_manager = StageManager()
        
        # Stage 1: Critical database infrastructure
        stage_manager.add_stage(
            "Database Initialization",
            [
                (init_database, [], {})
            ],
            critical=True
        )
        
        # Stage 2: Import historical data for basic tickers
        stage_manager.add_stage(
            "Historical Data Import",
            [
                (init_historical_data, [], {})
            ],
            critical=False  # Not critical, can proceed with limited data
        )
        
        # Stage 3: Initialize model without training
        stage_manager.add_stage(
            "Model Initialization",
            [
                (init_model, [], {})
            ],
            critical=False  # Changed to non-critical to allow progress without model
        )
        
        # Stage 4: Train models for basic tickers
        stage_manager.add_stage(
            "Model Training",
            [
                # Pass model parameter with default None value
                # This allows train_basic_models to handle None case
                (train_basic_models, [], {})
            ],
            critical=False  # Can proceed with untrained models
        )
        
        # Stage 5: System integration components
        stage_manager.add_stage(
            "System Integration",
            [
                (init_integration, [], {})
            ],
            critical=False
        )
        
        # Stage 6: UI initialization
        stage_manager.add_stage(
            "UI Initialization",
            [
                (init_ui, [], {})
            ],
            critical=True  # UI is critical for user interaction
        )
        
        # Stage 7: NLP models (resource-intensive)
        stage_manager.add_stage(
            "NLP Components",
            [
                (init_nlp_models, [], {})
            ],
            critical=False  # Can fall back to simpler methods
        )
        
        # Stage 8: Data collection (least critical, can start later)
        stage_manager.add_stage(
            "Data Collection",
            [
                (init_data_collection, [True], {})  # Start data collection immediately
            ],
            critical=False
        )
        
        # Run all stages
        stage_manager.run_stages()
        
        # Explicit garbage collection after initialization
        gc.collect()
        
        # Log memory usage after initialization
        mem = psutil.virtual_memory()
        logger.info(f"System Memory: Total: {mem.total / (1024**2):.2f} MB, Available: {mem.available / (1024**2):.2f} MB, Used: {mem.percent}%")
        logger.info(f"CPU Usage: {psutil.cpu_percent()}%")
        logger.info(f"Current Process Memory: {psutil.Process().memory_info().rss / (1024**2):.2f} MB")
        
        # Check UI initialization
        app, socketio = stage_manager.get_component("init_ui") or (None, None)
        ui_initialized = app is not None and socketio is not None
        
        # Check model initialization
        model = stage_manager.get_component("init_model")
        model_initialized = model is not None
        
        # Check integration initialization
        integration = stage_manager.get_component("init_integration")
        integration_initialized = integration is not None
        
        logger.info(f"UI app initialized: {ui_initialized}")
        logger.info(f"UI socketio initialized: {socketio is not None}")
        logger.info(f"Model class initialized: {model_initialized}")
        logger.info(f"Integration module initialized: {integration_initialized}")
        
        # Add model to app config if UI initialized
        if ui_initialized and model_initialized and app:
            app.config['model_instance'] = model
            logger.info("Added model instance to app configuration")
        
        logger.info("Starting system initialization process...")
        
        # Log memory usage before launching web server
        mem = psutil.virtual_memory()
        logger.info(f"System Memory: Total: {mem.total / (1024**2):.2f} MB, Available: {mem.available / (1024**2):.2f} MB, Used: {mem.percent}%")
        logger.info(f"CPU Usage: {psutil.cpu_percent()}%")
        logger.info(f"Current Process Memory: {psutil.Process().memory_info().rss / (1024**2):.2f} MB")
        
        # Start web server if properly initialized
        if app and socketio:
            # Log registered routes
            logger.info("Registered Flask routes:")
            for rule in app.url_map.iter_rules():
                logger.info(f"  {rule}")
            
            logger.info("Launching web server...")
            # Log memory usage one final time before launching server
            mem = psutil.virtual_memory()
            logger.info(f"System Memory: Total: {mem.total / (1024**2):.2f} MB, Available: {mem.available / (1024**2):.2f} MB, Used: {mem.percent}%")
            logger.info(f"CPU Usage: {psutil.cpu_percent()}%")
            logger.info(f"Current Process Memory: {psutil.Process().memory_info().rss / (1024**2):.2f} MB")
            
            # Start initial data ingestion
            logger.info("Starting initial data ingestion process...")
            if stage_manager.get_component("init_data_collection"):
                logger.info("Started background thread for initial data collection")
            
            # Run the application
            socketio.run(app, host='0.0.0.0', port=5000)
            return 0
        else:
            logger.critical("Failed to initialize web server")
            return 1
    
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}")
        logger.critical(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())