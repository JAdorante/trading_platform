import logging
import os
import pandas as pd
import numpy as np
import secrets
import pickle
from datetime import datetime, timedelta
from typing import Optional, List, Union, Dict, Any  # Added more type hints
import traceback
import time
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model")

# Import required libraries with fallback mechanisms
try:
    from sklearn.ensemble import RandomForestRegressor
    sklearn_available = True
except ImportError:
    logger.warning("scikit-learn not available, RandomForestRegressor will not work")
    sklearn_available = False

try:
    import xgboost as xgb
    xgboost_available = True
except ImportError:
    logger.warning("xgboost not available, XGBRegressor will not work")
    xgboost_available = False

try:
    from sklearn.metrics import mean_squared_error
    metrics_available = True
except ImportError:
    logger.warning("scikit-learn metrics not available, MSE calculation will not work")
    metrics_available = False

try:
    import joblib
    joblib_available = True
except ImportError:
    logger.warning("joblib not available, model saving/loading will not work")
    joblib_available = False

# Import local modules with error handling
try:
    from historical_data_import import get_historical_price_data
    historical_data_available = True
except ImportError:
    logger.warning("historical_data_import module not available, will use mock data")
    historical_data_available = False

try:
    from database import save_price_prediction, save_technical_indicators, DBConnection
    database_available = True
except ImportError:
    logger.warning("database module not available, will not save predictions to database")
    database_available = False

class StockAnalysisModel:
    def __init__(self, tickers=None, load_models=True):
        """Initialize the stock prediction model with options for partial loading."""
        try:
            self.models = {}
            self.tickers = tickers or ['NVDA', 'AAPL', 'AMD', 'TSLA', 'META','AMZN', 'MSFT', 'GOOGL','QQQ','DIA','SPY']
            self.model_dir = "models"
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Check for required dependencies
            self.dependencies_available = self._check_dependencies()
            
            # Only load existing models if requested and dependencies are available
            if load_models and self.dependencies_available:
                self.load_models()
                
            logger.info("StockAnalysisModel initialized")
        except Exception as e:
            logger.error(f"Error initializing StockAnalysisModel: {e}")
            logger.error(traceback.format_exc())
            self.models = {}
            
    def _check_dependencies(self):
        """Check if all required dependencies are available"""
        required_deps = []
        
        if not sklearn_available:
            logger.error("scikit-learn is required but not available")
            required_deps.append("scikit-learn")
        
        if not xgboost_available:
            logger.error("xgboost is required but not available")
            required_deps.append("xgboost")
        
        if not joblib_available:
            logger.error("joblib is required but not available")
            required_deps.append("joblib")
        
        if required_deps:
            logger.error(f"Missing required dependencies: {', '.join(required_deps)}")
            return False
        
        return True

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the input data.
        """
        try:
            if not isinstance(data, pd.DataFrame):
                logger.error(f"Input data is not a DataFrame: {type(data)}")
                return pd.DataFrame()
                
            if data.empty:
                logger.error("Input data is empty DataFrame")
                return pd.DataFrame()

            # Make a copy to avoid modifying the original
            data_copy = data.copy()

            # Simple Moving Average (20-day)
            data_copy['SMA20'] = data_copy['Close'].rolling(window=20).mean()
            
            # Add additional SMAs
            data_copy['SMA5'] = data_copy['Close'].rolling(window=5).mean()
            data_copy['SMA50'] = data_copy['Close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            data_copy['EMA12'] = data_copy['Close'].ewm(span=12, adjust=False).mean()
            data_copy['EMA26'] = data_copy['Close'].ewm(span=26, adjust=False).mean()
            
            # Relative Strength Index (14-day)
            delta = data_copy['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = gain / loss
                rs = np.where(np.isnan(rs) | np.isinf(rs), 0, rs)
            data_copy['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD and its components
            data_copy['MACD'] = data_copy['EMA12'] - data_copy['EMA26']
            data_copy['MACD_Signal'] = data_copy['MACD'].ewm(span=9, adjust=False).mean()
            data_copy['MACD_Histogram'] = data_copy['MACD'] - data_copy['MACD_Signal']
            
            # Bollinger Bands
            data_copy['BB_Middle'] = data_copy['Close'].rolling(window=20).mean()
            data_copy['BB_Std'] = data_copy['Close'].rolling(window=20).std()
            data_copy['BB_Upper'] = data_copy['BB_Middle'] + (data_copy['BB_Std'] * 2)
            data_copy['BB_Lower'] = data_copy['BB_Middle'] - (data_copy['BB_Std'] * 2)
            data_copy['BB_Width'] = (data_copy['BB_Upper'] - data_copy['BB_Lower']) / data_copy['BB_Middle']
            
            # Average True Range (ATR)
            high_low = data_copy['High'] - data_copy['Low']
            high_close = np.abs(data_copy['High'] - data_copy['Close'].shift())
            low_close = np.abs(data_copy['Low'] - data_copy['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data_copy['ATR'] = true_range.rolling(14).mean()
            
            # Rate of Change
            data_copy['ROC_5'] = data_copy['Close'].pct_change(periods=5) * 100
            data_copy['ROC_10'] = data_copy['Close'].pct_change(periods=10) * 100
            
            # Stochastic Oscillator
            low_14 = data_copy['Low'].rolling(window=14).min()
            high_14 = data_copy['High'].rolling(window=14).max()
            data_copy['%K'] = ((data_copy['Close'] - low_14) / (high_14 - low_14)) * 100
            data_copy['%D'] = data_copy['%K'].rolling(window=3).mean()
            
            # On-Balance Volume (OBV)
            obv = np.zeros(len(data_copy))
            for i in range(1, len(data_copy)):
                if data_copy['Close'].iloc[i] > data_copy['Close'].iloc[i-1]:
                    obv[i] = obv[i-1] + data_copy['Volume'].iloc[i]
                elif data_copy['Close'].iloc[i] < data_copy['Close'].iloc[i-1]:
                    obv[i] = obv[i-1] - data_copy['Volume'].iloc[i]
                else:
                    obv[i] = obv[i-1]
            data_copy['OBV'] = obv
            
            # Volume Weighted Average Price (VWAP) - daily
            data_copy['VWAP'] = (data_copy['Close'] * data_copy['Volume']).rolling(window=20).sum() / data_copy['Volume'].rolling(window=20).sum()
            
            # Price Momentum Oscillator
            data_copy['PMO'] = data_copy['Close'].diff(10) / data_copy['Close'].shift(10) * 100
            
            # Average Directional Index (ADX) components
            plus_dm = np.zeros(len(data_copy))
            minus_dm = np.zeros(len(data_copy))
            for i in range(1, len(data_copy)):
                up_move = data_copy['High'].iloc[i] - data_copy['High'].iloc[i-1]
                down_move = data_copy['Low'].iloc[i-1] - data_copy['Low'].iloc[i]
                plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
                minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0
            data_copy['Plus_DM'] = plus_dm
            data_copy['Minus_DM'] = minus_dm

            # Smooth +DM and -DM over 14 periods
            data_copy['Plus_DM_14'] = data_copy['Plus_DM'].rolling(window=14).sum()
            data_copy['Minus_DM_14'] = data_copy['Minus_DM'].rolling(window=14).sum()
            data_copy['TR_14'] = true_range.rolling(window=14).sum()
            data_copy['Plus_DI_14'] = 100 * (data_copy['Plus_DM_14'] / data_copy['TR_14'])
            data_copy['Minus_DI_14'] = 100 * (data_copy['Minus_DM_14'] / data_copy['TR_14'])
            data_copy['DX'] = 100 * (np.abs(data_copy['Plus_DI_14'] - data_copy['Minus_DI_14']) / (data_copy['Plus_DI_14'] + data_copy['Minus_DI_14']))
            data_copy['ADX'] = data_copy['DX'].rolling(window=14).mean()
            
            # Volatility indicator - Standard deviation of returns
            data_copy['Volatility'] = data_copy['Close'].pct_change().rolling(window=20).std() * 100
            
            logger.debug(f"Calculated technical indicators, new columns: {[col for col in data_copy.columns if col not in data.columns]}")
            return data_copy
        
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def prepare_training_data(self, data: pd.DataFrame, ticker: str = None) -> tuple:
        """Prepare features and target for model training."""
        try:
            if not isinstance(data, pd.DataFrame):
                logger.error(f"Data is not a DataFrame: {type(data)}")
                return None, None

            # Additional data validation
            logger.debug(f"Data columns: {data.columns.tolist()}")
            logger.debug(f"Data shape: {data.shape}")
            logger.debug(f"Data types: {data.dtypes}")
            logger.debug(f"First few rows: {data.head(2)}")

            # Ensure data has enough rows
            if len(data) < 30:
                logger.warning(f"Insufficient data rows: {len(data)}")
                return None, None

            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)

            # Handle if technical indicators calculation failed
            if data.empty:
                logger.warning("Technical indicators calculation failed")
                return None, None

            # Fill NaN values
            data = data.fillna(method='ffill').fillna(method='bfill')

            # Check required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA20', 'RSI', 'MACD', 'MACD_Signal']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return None, None

            # Initialize sentiment columns with defaults
            data['sentiment'] = 0
            data['mention_count'] = 0

            # Add sentiment data if database is available and ticker is provided
            if database_available and ticker:
                try:
                    with DBConnection() as conn:
                        # Get sentiment for dates in the data index
                        start_date = data.index.min().isoformat()
                        end_date = data.index.max().isoformat()
                        query = """
                            SELECT timestamp, AVG(score) as avg_sentiment, COUNT(*) as mention_count
                            FROM scores
                            WHERE stock_ticker = ? AND timestamp BETWEEN ? AND ?
                            GROUP BY DATE(timestamp)
                        """
                        sentiment_df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
                        
                        if not sentiment_df.empty:
                            # Convert to datetime index and join with price data
                            sentiment_df['date'] = pd.to_datetime(sentiment_df['timestamp']).dt.date
                            sentiment_df.set_index('date', inplace=True)
                            
                            # Match dates and add sentiment
                            for date, row in sentiment_df.iterrows():
                                matching_idx = data.index.date == date
                                if any(matching_idx):
                                    data.loc[matching_idx, 'sentiment'] = row['avg_sentiment']
                                    data.loc[matching_idx, 'mention_count'] = row['mention_count']
                except Exception as db_err:
                    logger.warning(f"Error fetching sentiment data: {db_err}")
                    # Continue without sentiment data if there's an error

            # Fill missing sentiment values
            data['sentiment'] = data['sentiment'].fillna(method='ffill').fillna(0)
            data['mention_count'] = data['mention_count'].fillna(0)

            # Assemble features list (including sentiment data)
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA20', 'RSI', 'MACD', 'MACD_Signal', 'sentiment', 'mention_count']

            # Create features and target
            X = data[features].iloc[:-1].copy()  # All but last row
            y = data['Close'].shift(-1).iloc[:-1].copy()  # Next day's close

            if X.empty or y.empty:
                logger.warning("Empty features or target after preparation")
                return None, None

            # Check for NaN values and fill if necessary
            if X.isna().any().any() or y.isna().any():
                logger.warning("NaN values in features or target - applying additional filling")
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())

            return X, y

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            logger.error(traceback.format_exc())
            return None, None

    def _create_mock_data(self, ticker: str) -> pd.DataFrame:
        """Create mock data for a ticker when historical data is not available."""
        logger.warning(f"Creating mock data for {ticker} since historical data is unavailable")
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Generate mock stock prices
        np.random.seed(hash(ticker) % 2**32)  # Deterministic seed based on ticker
        
        # Set initial price based on ticker
        if ticker == 'NVDA':
            initial_price = 90
        elif ticker == 'AAPL':
            initial_price = 180
        elif ticker == 'TSLA':
            initial_price = 220
        elif ticker == 'META':
            initial_price = 350
        elif ticker == 'AMZN':
            initial_price = 190
        elif ticker == 'MSFT':
            initial_price = 390
        elif ticker == 'GOOGL':
            initial_price = 150
        else:
            initial_price = 100
        
        # Generate price series with random walk
        price_series = [initial_price]
        for i in range(1, len(date_range)):
            # Daily change with some randomness
            daily_change = np.random.normal(0, 0.015)  # 1.5% standard deviation
            price_series.append(price_series[-1] * (1 + daily_change))
        
        # Create DataFrame
        data = pd.DataFrame(index=date_range)
        data['Close'] = price_series
        
        # Generate other price columns based on Close
        data['Open'] = data['Close'] * (1 + np.random.normal(0, 0.005, len(data)))
        data['High'] = data[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.003, len(data))))
        data['Low'] = data[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.003, len(data))))
        data['Volume'] = np.random.randint(1000000, 10000000, len(data))
        
        return data

    def train_model(self, ticker: str, retrain: bool = True) -> bool:
        """
        Train a model for the given ticker.
    
        Args:
            ticker (str): The stock ticker symbol
            retrain (bool, optional): Force retraining even if model exists. Defaults to False.
        
        Returns:
            bool: Success status of the training
        """
        try:
            logger.info(f"Training model for {ticker}")
            
            # Check dependencies
            if not self.dependencies_available:
                logger.error("Required dependencies are not available for training")
                return False
            
            # Check if model already exists and retrain is False
            model_path = os.path.join(self.model_dir, f"{ticker}_rf_model.pkl")
            if not retrain and os.path.exists(model_path):
                logger.info(f"Model for {ticker} already exists and retrain=False, skipping training")
                self.load_models([ticker])
                return True
            
            # Fetch historical data
            if historical_data_available:
                data = get_historical_price_data(ticker)
                if data is None or data.empty:
                    logger.warning(f"No historical data retrieved for {ticker}, using mock data")
                    data = self._create_mock_data(ticker)
            else:
                logger.warning("Historical data module not available, using mock data")
                data = self._create_mock_data(ticker)
            
            logger.debug(f"Data for {ticker}: shape={data.shape}, columns={data.columns.tolist()}")
            
            # Prepare training data (pass ticker to get sentiment data)
            X, y = self.prepare_training_data(data, ticker)
            if X is None or y is None:
                logger.warning(f"Insufficient data to train model for {ticker} (rows: {len(data)})")
                return False
            
            logger.info(f"Training with features: {X.columns.tolist()}")
            
            # Define models
            if sklearn_available:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                logger.error("scikit-learn not available, cannot create RandomForestRegressor")
                return False
                
            if xgboost_available:
                xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            else:
                logger.error("xgboost not available, cannot create XGBRegressor")
                return False
            
            # Optionally perform cross-validation
            try:
                from sklearn.model_selection import TimeSeriesSplit
                tscv = TimeSeriesSplit(n_splits=5)
                rf_scores = []
                xgb_scores = []
                ensemble_scores = []
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    rf_model.fit(X_train, y_train)
                    xgb_model.fit(X_train, y_train)
                    rf_pred = rf_model.predict(X_test)
                    xgb_pred = xgb_model.predict(X_test)
                    ensemble_pred = (rf_pred + xgb_pred) / 2
                    if metrics_available:
                        rf_mse = mean_squared_error(y_test, rf_pred)
                        xgb_mse = mean_squared_error(y_test, xgb_pred)
                        ens_mse = mean_squared_error(y_test, ensemble_pred)
                        rf_scores.append(np.sqrt(rf_mse))
                        xgb_scores.append(np.sqrt(xgb_mse))
                        ensemble_scores.append(np.sqrt(ens_mse))
                    else:
                        rf_scores.append(np.sqrt(np.mean((y_test.values - rf_pred) ** 2)))
                        xgb_scores.append(np.sqrt(np.mean((y_test.values - xgb_pred) ** 2)))
                        ensemble_scores.append(np.sqrt(np.mean((y_test.values - ensemble_pred) ** 2)))
                logger.info(f"Cross-validation RMSE for {ticker} - RF: {np.mean(rf_scores):.4f}, "
                            f"XGB: {np.mean(xgb_scores):.4f}, Ensemble: {np.mean(ensemble_scores):.4f}")
            except Exception as cv_err:
                logger.warning(f"TimeSeriesSplit error or not available: {cv_err}")
                # Fallback to simple split if CV fails.
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                rf_model.fit(X_train, y_train)
                xgb_model.fit(X_train, y_train)
            
            # Final training on all data
            logger.info(f"Training final models for {ticker} on all data")
            rf_model.fit(X, y)
            xgb_model.fit(X, y)
            
            # Save models if joblib is available
            if joblib_available:
                os.makedirs(self.model_dir, exist_ok=True)
                joblib.dump(rf_model, os.path.join(self.model_dir, f"{ticker}_rf_model.pkl"))
                joblib.dump(xgb_model, os.path.join(self.model_dir, f"{ticker}_xgb_model.pkl"))
                import json
                metrics_path = os.path.join(self.model_dir, f"{ticker}_metrics.json")
                metrics_data = {
                    'training_date': datetime.now().isoformat(),
                    'features_used': X.columns.tolist(),
                    'data_rows': len(X)
                }
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
                logger.info(f"Models and metrics saved for {ticker}")
            else:
                logger.warning("joblib not available, models not saved to disk")
            
            self.models[ticker] = {'rf': rf_model, 'xgb': xgb_model}
            logger.info(f"Successfully trained and saved model for {ticker}")
            
            # Analyze feature importance after successful training
            self.analyze_feature_importance(ticker)
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {e}")
            logger.error(traceback.format_exc())
            return False
        
    def analyze_feature_importance(self, ticker: str):
        """
        Analyze and log which features are most important for prediction.
        
        Args:
            ticker (str): Stock ticker symbol.
        
        Returns:
            dict or None: Sorted dictionary of features and their importance, or None if an error occurs.
        """
        try:
            if ticker not in self.models:
                logger.warning(f"No trained model for {ticker}")
                return None
            
            # Retrieve the RandomForest model for feature importance
            rf_model = self.models[ticker].get('rf')
            if rf_model is None:
                logger.warning(f"RandomForest model not found for {ticker}")
                return None
            
            # Try to retrieve feature names from saved metrics metadata
            import json
            metrics_path = os.path.join(self.model_dir, f"{ticker}_metrics.json")
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    features = metrics.get('features_used', [])
                except Exception as load_err:
                    logger.warning(f"Error loading metrics for {ticker}: {load_err}")
                    features = []
            else:
                features = []
            
            # Fallback to default feature list if metadata is not available
            if not features:
                features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                            'SMA20', 'RSI', 'MACD', 'MACD_Signal', 'sentiment', 'mention_count']
            
            # Ensure the feature list matches the length of importance values
            if not hasattr(rf_model, 'feature_importances_'):
                logger.warning(f"Model for {ticker} does not support feature importance")
                return None
            
            importance_values = rf_model.feature_importances_
            if len(features) < len(importance_values):
                features.extend([f'Unknown_{i}' for i in range(len(features), len(importance_values))])
            else:
                features = features[:len(importance_values)]
            
            # Create a dictionary and sort by importance (descending)
            importance_dict = dict(zip(features, importance_values))
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            logger.info(f"Feature importance for {ticker}: {sorted_importance}")
            return sorted_importance
        
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            logger.error(traceback.format_exc())
            return None

    def predict(self, ticker: str, days_ahead: int = 1) -> Optional[float]:
        """Predict the stock price for the given ticker."""
        try:
            if ticker not in self.models:
                logger.warning(f"No trained model for {ticker}")
                return None
            
            # Make sure we have both RF and XGB models
            if 'rf' not in self.models[ticker] or 'xgb' not in self.models[ticker]:
                logger.warning(f"Missing models for {ticker}")
                return None
                
            # Fetch recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # Last 60 days for context
            
            if historical_data_available:
                data = get_historical_price_data(ticker, start_date, end_date)
                if data is None or data.empty:
                    logger.warning(f"No recent data for prediction on {ticker}, using mock data")
                    data = self._create_mock_data(ticker)
            else:
                logger.warning(f"Historical data not available, using mock data for {ticker}")
                data = self._create_mock_data(ticker)
            
            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)
            if data.empty:
                logger.warning(f"Failed to calculate technical indicators for {ticker}")
                return None
            
            # Fill NaN values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Initialize sentiment columns with defaults
            data['sentiment'] = 0
            data['mention_count'] = 0
            
            # Add sentiment data if database is available
            if database_available:
                try:
                    with DBConnection() as conn:
                        # Get sentiment for dates in the data index
                        start_date_iso = data.index.min().isoformat()
                        end_date_iso = data.index.max().isoformat()
                        query = """
                            SELECT timestamp, AVG(score) as avg_sentiment, COUNT(*) as mention_count
                            FROM scores
                            WHERE stock_ticker = ? AND timestamp BETWEEN ? AND ?
                            GROUP BY DATE(timestamp)
                        """
                        sentiment_df = pd.read_sql_query(query, conn, params=(ticker, start_date_iso, end_date_iso))
                        
                        if not sentiment_df.empty:
                            # Convert to datetime index and join with price data
                            sentiment_df['date'] = pd.to_datetime(sentiment_df['timestamp']).dt.date
                            sentiment_df.set_index('date', inplace=True)
                            
                            # Match dates and add sentiment
                            for date, row in sentiment_df.iterrows():
                                matching_idx = data.index.date == date
                                if any(matching_idx):
                                    data.loc[matching_idx, 'sentiment'] = row['avg_sentiment']
                                    data.loc[matching_idx, 'mention_count'] = row['mention_count']
                except Exception as db_err:
                    logger.warning(f"Error fetching sentiment data for prediction: {db_err}")
            
            # Fill missing sentiment values
            data['sentiment'] = data['sentiment'].fillna(method='ffill').fillna(0)
            data['mention_count'] = data['mention_count'].fillna(0)
            
            # Use the same feature list as in training - INCLUDING sentiment features
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA20', 'RSI', 'MACD', 'MACD_Signal', 'sentiment', 'mention_count']
            
            # Check for missing features
            missing_features = [f for f in features if f not in data.columns]
            if missing_features:
                logger.warning(f"Missing features for {ticker}: {missing_features}")
                # Add missing features with default values
                for feature in missing_features:
                    data[feature] = 0
            
            # Get last row for prediction
            if len(data) == 0:
                logger.warning(f"No data rows after preprocessing for {ticker}")
                return None
                
            X = data[features].iloc[-1:].values  # Last row as input
            
            # Predict
            rf_model = self.models[ticker]['rf']
            xgb_model = self.models[ticker]['xgb']
            rf_pred = rf_model.predict(X)[0]
            xgb_pred = xgb_model.predict(X)[0]
            prediction = (rf_pred + xgb_pred) / 2
            
            # Calculate predicted change
            current_price = data['Close'].iloc[-1]
            predicted_change = (prediction - current_price) / current_price
            
            # Save prediction if database is available
            if database_available:
                try:
                    save_price_prediction(ticker, predicted_change)
                except Exception as save_err:
                    logger.error(f"Error saving prediction for {ticker}: {save_err}")
                
                # Save technical indicators
                try:
                    indicators = {
                        'RSI': data['RSI'].iloc[-1],
                        'MACD': data['MACD'].iloc[-1],
                        'MACD_Signal': data['MACD_Signal'].iloc[-1],
                        'SMA20': data['SMA20'].iloc[-1]
                    }
                    save_technical_indicators(ticker, indicators)
                except Exception as tech_err:
                    logger.error(f"Error saving technical indicators for {ticker}: {tech_err}")
            else:
                logger.warning("Database not available, prediction not saved")
                    
            logger.info(f"Predicted price for {ticker}: {prediction:.2f}")
            return prediction
        
        except Exception as e:
            logger.error(f"Error predicting for {ticker}: {e}")
            logger.error(traceback.format_exc())
            return None

    def train_all_models(self, tickers=None) -> int:
        """Train models for all configured tickers.
        
        Args:
            tickers (List[str], optional): List of stock tickers to train models for.
                                          Defaults to None (uses self.tickers).
        
        Returns:
            int: Number of successfully trained models
        """
        success_count = 0
        # Use provided tickers or fall back to default tickers
        ticker_list = tickers if tickers else self.tickers
        
        for ticker in ticker_list:
            try:
                if self.train_model(ticker):
                    success_count += 1
                # Add delay between training models to avoid resource contention
                time.sleep(2)
            except Exception as e:
                logger.error(f"Unexpected error training model for {ticker}: {e}")
        
        logger.info(f"Model training completed. {success_count}/{len(ticker_list)} models successfully trained.")
        return success_count

    def load_models(self, tickers=None):
        """Load pre-trained models from disk."""
        try:
            # Check if joblib is available
            if not joblib_available:
                logger.error("joblib not available, cannot load models")
                return False
                
            # Load only specific tickers or all tickers
            load_tickers = tickers if tickers else self.tickers
            
            for ticker in load_tickers:
                rf_path = os.path.join(self.model_dir, f"{ticker}_rf_model.pkl")
                xgb_path = os.path.join(self.model_dir, f"{ticker}_xgb_model.pkl")
                
                if os.path.exists(rf_path) and os.path.exists(xgb_path):
                    try:
                        rf_model = joblib.load(rf_path)
                        xgb_model = joblib.load(xgb_path)
                        self.models[ticker] = {
                            'rf': rf_model,
                            'xgb': xgb_model
                        }
                        logger.info(f"Loaded models for {ticker}")
                    except Exception as load_err:
                        logger.error(f"Error loading models for {ticker}: {load_err}")
                else:
                    logger.warning(f"Models not found for {ticker}")
            
            return True
        except Exception as e:
            logger.error(f"Error in load_models: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def generate_trading_signals(self, ticker: str) -> dict:
        """Generate trading signals for a ticker based on technical indicators and predictions.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Dictionary with trading signals and metadata
        """
        try:
            # Get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            if historical_data_available:
                data = get_historical_price_data(ticker, start_date, end_date)
                if data is None or data.empty:
                    logger.warning(f"No historical data for {ticker}, using mock data")
                    data = self._create_mock_data(ticker)
            else:
                logger.warning(f"Historical data not available, using mock data for {ticker}")
                data = self._create_mock_data(ticker)
            
            if data is None or data.empty:
                return {"error": "No data available for trading signals"}
            
            # Calculate indicators
            data = self.calculate_technical_indicators(data)
            if data.empty:
                return {"error": "Failed to calculate technical indicators"}
            
            # Fill missing values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Get latest values
            if len(data) < 2:
                return {"error": "Insufficient data points for signal generation"}
                
            latest = data.iloc[-1]
            previous = data.iloc[-2]
            
            # Generate signals
            rsi = latest['RSI'] if 'RSI' in latest else 50
            macd = latest['MACD'] if 'MACD' in latest else 0
            macd_signal = latest['MACD_Signal'] if 'MACD_Signal' in latest else 0
            sma = latest['SMA20'] if 'SMA20' in latest else latest['Close']
            price = latest['Close']
            
            # Determine signal
            signals = {
                "ticker": ticker,
                "price": price,
                "timestamp": latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else str(latest.name),
                "indicators": {
                    "RSI": rsi,
                    "MACD": macd,
                    "MACD_Signal": macd_signal,
                    "SMA20": sma
                },
                "signals": {}
            }
            
            # RSI signal
            if rsi < 30:
                signals["signals"]["RSI"] = "BUY"
            elif rsi > 70:
                signals["signals"]["RSI"] = "SELL"
            else:
                signals["signals"]["RSI"] = "HOLD"
            
            # MACD signal
            if macd > macd_signal and (not hasattr(previous, 'MACD') or not hasattr(previous, 'MACD_Signal') or
                                       previous['MACD'] <= previous['MACD_Signal']):
                signals["signals"]["MACD"] = "BUY"
            elif macd < macd_signal and (not hasattr(previous, 'MACD') or not hasattr(previous, 'MACD_Signal') or
                                         previous['MACD'] >= previous['MACD_Signal']):
                signals["signals"]["MACD"] = "SELL"
            else:
                signals["signals"]["MACD"] = "HOLD"
            
            # SMA signal
            if price > sma:
                signals["signals"]["SMA"] = "BUY"
            else:
                signals["signals"]["SMA"] = "SELL"
            
            # Overall signal
            buy_signals = sum(1 for signal in signals["signals"].values() if signal == "BUY")
            sell_signals = sum(1 for signal in signals["signals"].values() if signal == "SELL")
            
            if buy_signals > sell_signals:
                signals["overall"] = "BUY"
            elif sell_signals > buy_signals:
                signals["overall"] = "SELL"
            else:
                signals["overall"] = "HOLD"
            
            return signals
        
        except Exception as e:
            logger.error(f"Error generating trading signals for {ticker}: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def backtest_model(self, ticker: str, days: int = 90) -> dict:
        """Run backtest for a ticker to evaluate model performance.
        
        Args:
            ticker (str): Stock ticker symbol
            days (int): Number of historical days to backtest
            
        Returns:
            dict: Backtest results including performance metrics
        """
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            if historical_data_available:
                data = get_historical_price_data(ticker, start_date, end_date)
                if data is None or data.empty:
                    logger.warning(f"No historical data for {ticker}, using mock data")
                    data = self._create_mock_data(ticker)
            else:
                logger.warning(f"Historical data not available, using mock data for {ticker}")
                data = self._create_mock_data(ticker)
            
            if data is None or data.empty:
                return {"error": "No data available for backtesting"}
            
            # Calculate indicators
            data = self.calculate_technical_indicators(data)
            if data.empty:
                return {"error": "Failed to calculate technical indicators"}
            
            # Fill NaN values that might have been introduced
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Initialize results
            trades = []
            position = None
            buy_price = 0
            
            # Generate signals for each day
            for i in range(20, len(data) - 1):  # Start after enough data for indicators
                current_day = data.iloc[i]
                next_day = data.iloc[i + 1]
                
                # Get indicators with fallback values
                rsi = current_day.get('RSI', 50)
                macd = current_day.get('MACD', 0)
                macd_signal = current_day.get('MACD_Signal', 0)
                sma = current_day.get('SMA20', current_day['Close'])
                price = current_day['Close']
                
                # Determine signal
                buy_signal = (rsi < 30 or macd > macd_signal or price > sma)
                sell_signal = (rsi > 70 or macd < macd_signal or price < sma)
                
                # Execute trades
                if position is None and buy_signal:
                    position = "LONG"
                    buy_price = next_day['Open']  # Buy at next day's open
                    trades.append({
                        "date": next_day.name.strftime('%Y-%m-%d') if hasattr(next_day.name, 'strftime') else str(next_day.name),
                        "action": "BUY",
                        "price": buy_price
                    })
                elif position == "LONG" and sell_signal:
                    sell_price = next_day['Open']  # Sell at next day's open
                    profit_pct = (sell_price - buy_price) / buy_price * 100
                    trades.append({
                        "date": next_day.name.strftime('%Y-%m-%d') if hasattr(next_day.name, 'strftime') else str(next_day.name),
                        "action": "SELL",
                        "price": sell_price,
                        "profit": profit_pct
                    })
                    position = None
            
            # Calculate performance metrics
            if trades:
                # Close any open position at the end
                if position == "LONG":
                    sell_price = data.iloc[-1]['Close']
                    profit_pct = (sell_price - buy_price) / buy_price * 100
                    trades.append({
                        "date": data.iloc[-1].name.strftime('%Y-%m-%d') if hasattr(data.iloc[-1].name, 'strftime') else str(data.iloc[-1].name),
                        "action": "SELL (END)",
                        "price": sell_price,
                        "profit": profit_pct
                    })
                
                # Calculate metrics
                total_trades = sum(1 for trade in trades if trade["action"] == "SELL" or trade["action"] == "SELL (END)")
                winning_trades = sum(1 for trade in trades if trade.get("profit", 0) > 0)
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                # Calculate total return
                total_return = 1.0
                for trade in trades:
                    if "profit" in trade:
                        total_return *= (1 + trade["profit"] / 100)
                total_return = (total_return - 1) * 100
                
                # Calculate buy and hold return
                buy_hold_return = (data.iloc[-1]['Close'] - data.iloc[20]['Close']) / data.iloc[20]['Close'] * 100
                
                # Calculate Sharpe ratio (simplified)
                profits = [trade.get("profit", 0) for trade in trades if "profit" in trade]
                if profits:
                    avg_return = np.mean(profits)
                    std_return = np.std(profits) if len(profits) > 1 else 1
                    sharpe_ratio = avg_return / std_return if std_return > 0 else 0
                else:
                    sharpe_ratio = 0
                
                # Save backtest results to database
                if database_available:
                    try:
                        from database import save_backtest_results
                        backtest_result = {
                            'start_date': start_date.strftime('%Y-%m-%d'),
                            'end_date': end_date.strftime('%Y-%m-%d'),
                            'total_trades': total_trades,
                            'win_rate': win_rate,
                            'total_return': total_return / 100,
                            'buy_hold_return': buy_hold_return / 100,
                            'sharpe_ratio': sharpe_ratio
                        }
                        save_backtest_results(ticker, backtest_result)
                    except Exception as save_err:
                        logger.warning(f"Could not save backtest results: {save_err}")
                
                return {
                    "ticker": ticker,
                    "start_date": start_date.strftime('%Y-%m-%d'),
                    "end_date": end_date.strftime('%Y-%m-%d'),
                    "total_trades": total_trades,
                    "win_rate": win_rate,
                    "total_return": total_return / 100,  # Convert to decimal
                    "buy_hold_return": buy_hold_return / 100,  # Convert to decimal
                    "sharpe_ratio": sharpe_ratio,
                    "trades": trades
                }
            else:
                return {
                    "ticker": ticker,
                    "start_date": start_date.strftime('%Y-%m-%d'),
                    "end_date": end_date.strftime('%Y-%m-%d'),
                    "total_trades": 0,
                    "message": "No trades executed during backtest period"
                }
        
        except Exception as e:
            logger.error(f"Error during backtesting for {ticker}: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

if __name__ == "__main__":
    try:
        model = StockAnalysisModel()
        success_count = model.train_all_models()
        logger.info(f"Successfully trained {success_count} models")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error(traceback.format_exc())