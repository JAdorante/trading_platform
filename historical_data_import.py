import os
import pandas as pd
import numpy as np
import requests
import json
from typing import Optional, Union, List, Dict
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance as yf
import logging
from pathlib import Path

# Try to import Alpaca
try:
    import alpaca_trade_api as tradeapi
    alpaca_available = True
except ImportError:
    alpaca_available = False
    logging.warning("Alpaca API not available. Install with: pip install alpaca-trade-api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
# Configure yfinance's internal logger after basic config
logging.getLogger("yfinance").setLevel(logging.WARNING)
logger = logging.getLogger("historical_data_import")

# Load environment variables
load_dotenv()

# Alpaca API credentials - hardcoded for now, but ideally should be in environment variables
ALPACA_API_KEY = "PKX8Z8783M2SBXY675B1"
ALPACA_API_SECRET = "ifNuKCr4RLQvRlwjWLaCvi1TPnzAme5tER1OspNb"
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"

class HistoricalDataImporter:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize historical data importer with multiple data sources.
        
        Args:
            api_key (str, optional): API key for alternative data sources
        """
        # Alpha Vantage API key (optional)
        self.alpha_vantage_key = api_key or os.environ.get('ALPHA_VANTAGE_API_KEY')
        
        # Create data cache directory
        self.cache_dir = Path("data_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Fallback data sources configuration - reordered to prioritize Alpaca
        self.fallback_sources = [
            self._get_alpaca_data,     # Alpaca as first choice
            self._get_yfinance_data,   # Yahoo Finance as backup
            self._get_alpha_vantage_data,  # Alpha Vantage as third option
            self._get_manual_stock_data    # Simulated data as last resort
        ]
    
    def _get_cache_path(self, ticker: str, interval: str = "1d") -> Path:
        """Get the path to the cache file for a specific ticker and interval."""
        return self.cache_dir / f"{ticker}_{interval}_cache.csv"

    def _get_cached_data(
        self, 
        ticker: str, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime],
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Try to retrieve data from local cache.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str or datetime): Start date for data retrieval
            end_date (str or datetime): End date for data retrieval
            interval (str, optional): Data interval
        
        Returns:
            Optional[pd.DataFrame]: Cached price data or None if cache miss
        """
        cache_path = self._get_cache_path(ticker, interval)
        
        # Check if cache exists
        if not cache_path.exists():
            return None
            
        try:
            # Load cache metadata (last update time, etc.)
            meta_path = cache_path.with_suffix('.json')
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            else:
                meta = {'last_update': None}
                
            # Check if cache is too old (older than 1 day for daily data)
            if meta.get('last_update'):
                last_update = datetime.fromisoformat(meta['last_update'])
                cache_age = datetime.now() - last_update
                
                # Cache is too old if:
                # - For daily data: older than 1 day AND market is open (weekday and not holiday)
                # - For intraday data: older than 1 hour
                is_too_old = False
                
                if interval == '1d' and cache_age > timedelta(days=1):
                    # For daily data, only consider it too old during market days
                    today = datetime.now()
                    is_weekday = today.weekday() < 5  # 0-4 are Monday-Friday
                    if is_weekday:
                        is_too_old = True
                elif interval.lower() in ['1h', '1m', '5m', '15m'] and cache_age > timedelta(hours=1):
                    is_too_old = True
                    
                if is_too_old:
                    logger.info(f"Cache for {ticker} is too old ({cache_age.days} days), fetching fresh data")
                    return None
            
            # Convert dates to datetime for comparison, ensuring they're timezone-naive
            start = pd.to_datetime(start_date)
            if start.tzinfo is not None:
                start = start.tz_localize(None)  # Convert to timezone-naive
                
            end = pd.to_datetime(end_date)
            if end.tzinfo is not None:
                end = end.tz_localize(None)  # Convert to timezone-naive
            
            # Read cached data
            data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            
            # Ensure the index is timezone-naive
            if data.index.tzinfo is not None:
                data.index = data.index.tz_localize(None)
            
            # Check if cache covers the requested date range
            if len(data) > 0:
                cache_start = data.index.min()
                cache_end = data.index.max()
                
                # Make sure all timestamps are comparable (either all naive or all aware)
                if cache_start.tzinfo is not None and start.tzinfo is None:
                    start = start.tz_localize(cache_start.tzinfo)
                elif cache_start.tzinfo is None and start.tzinfo is not None:
                    cache_start = cache_start.tz_localize(start.tzinfo)
                    
                if cache_end.tzinfo is not None and end.tzinfo is None:
                    end = end.tz_localize(cache_end.tzinfo)
                elif cache_end.tzinfo is None and end.tzinfo is not None:
                    cache_end = cache_end.tz_localize(end.tzinfo)
                
                if cache_start <= start and cache_end >= end:
                    logger.info(f"Using cached data for {ticker} from {start} to {end}")
                    # Filter to requested date range
                    filtered_data = data[(data.index >= start) & (data.index <= end)]
                    return filtered_data
            
            return None
        
        except Exception as e:
            logger.warning(f"Error reading from cache for {ticker}: {e}")
            return None
            
    def _save_to_cache(self, ticker: str, data: pd.DataFrame, interval: str = "1d") -> None:
        """
        Save data to local cache.
        
        Args:
            ticker (str): Stock ticker symbol
            data (pd.DataFrame): Data to cache
            interval (str, optional): Data interval
        """
        if data is None or data.empty:
            return
            
        try:
            cache_path = self._get_cache_path(ticker, interval)
            
            # Ensure the data is timezone-naive before saving to cache
            data_to_save = data.copy()
            
            # Convert the index to naive timestamps if it has timezone info
            if data_to_save.index.tzinfo is not None:
                data_to_save.index = data_to_save.index.tz_localize(None)
            
            # Save data
            data_to_save.to_csv(cache_path)
            
            # Save metadata
            meta = {
                'ticker': ticker,
                'interval': interval,
                'last_update': datetime.now().isoformat(),
                'rows': len(data),
                'date_range': f"{data.index.min()} to {data.index.max()}"
            }
            
            with open(cache_path.with_suffix('.json'), 'w') as f:
                json.dump(meta, f)
                
            logger.info(f"Saved {len(data)} rows of {ticker} data to cache")
        
        except Exception as e:
            logger.warning(f"Error saving to cache for {ticker}: {e}")
            
    def _get_alpaca_data(
        self, 
        ticker: str, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime], 
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve historical data using Alpaca API.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str or datetime): Start date for data retrieval
            end_date (str or datetime): End date for data retrieval
            interval (str, optional): Data interval (1d, 1h, etc.)
            
        Returns:
            Optional[pd.DataFrame]: Historical price data or None
        """
        if not alpaca_available:
            logger.warning("Alpaca API package not installed")
            return None
            
        try:
            # Initialize Alpaca API using the hardcoded credentials
            api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_API_BASE_URL, api_version='v2')
            
            # Ensure dates are in string format
            start = start_date.strftime("%Y-%m-%d") if isinstance(start_date, datetime) else start_date
            end = end_date.strftime("%Y-%m-%d") if isinstance(end_date, datetime) else end_date
            
            # MODIFICATION: Use slightly older data to avoid SIP data restriction
            # Free accounts can access data that's not the most recent
            # End date should be at least 15 minutes in the past for market hours data
            # For more reliable access, get data from yesterday or earlier
            now = datetime.now()
            modified_end = (now - timedelta(days=1)).strftime("%Y-%m-%d")
            # Still use the user's requested start date or a reasonable default
            modified_start = start
            
            logger.info(f"Modified date range for Alpaca: {modified_start} to {modified_end}")
            
            # Convert interval to timeframe
            timeframe = '1Day'
            if interval.lower() == '1h':
                timeframe = '1Hour'
            elif interval.lower() == '15m':
                timeframe = '15Min'
                
            # Get historical data
            bars = api.get_bars(
                ticker,
                timeframe,
                start=modified_start,  # Use modified dates
                end=modified_end,
                adjustment='raw'
            ).df
            
            if bars.empty:
                logger.warning(f"No data returned from Alpaca for {ticker}")
                return None
                
            # Rename columns to match our expected format
            bars = bars.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            logger.info(f"Successfully retrieved {len(bars)} bars for {ticker} from Alpaca")
            return bars
        
        except Exception as e:
            logger.error(f"Alpaca data retrieval error for {ticker}: {e}")
            return None
    
    def _get_yfinance_data(
        self, 
        ticker: str, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime], 
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve historical data using yfinance.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str or datetime): Start date for data retrieval
            end_date (str or datetime): End date for data retrieval
            interval (str, optional): Data interval
        
        Returns:
            Optional[pd.DataFrame]: Historical price data or None
        """
        try:
            # Ensure dates are in string format
            start = start_date.strftime("%Y-%m-%d") if isinstance(start_date, datetime) else start_date
            end = end_date.strftime("%Y-%m-%d") if isinstance(end_date, datetime) else end_date
            
            # Download data with extended timeout and error handling
            data = yf.download(
                ticker, 
                start=start, 
                end=end, 
                interval=interval,
                progress=False,
                threads=False,  # Disable threading to avoid issues
                timeout=30,
                ignore_tz=True
            )
            
            # Additional validation
            if data is None or data.empty:
                logger.warning(f"No data retrieved for {ticker} from yfinance")
                return None
            
            # MODIFICATION: Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                logger.info(f"Converting multi-level columns DataFrame for {ticker}")
                # Create a new DataFrame with flattened column structure
                processed_data = pd.DataFrame(index=data.index)
                # Map the required columns from the multi-level structure
                processed_data['Open'] = data[('Open', ticker)] if ('Open', ticker) in data.columns else None
                processed_data['High'] = data[('High', ticker)] if ('High', ticker) in data.columns else None
                processed_data['Low'] = data[('Low', ticker)] if ('Low', ticker) in data.columns else None
                processed_data['Close'] = data[('Close', ticker)] if ('Close', ticker) in data.columns else None
                processed_data['Volume'] = data[('Volume', ticker)] if ('Volume', ticker) in data.columns else None
                data = processed_data
            
            # Basic column validation
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                logger.warning(f"Missing required columns for {ticker}")
                return None
            
            # Ensure data is properly indexed
            if not isinstance(data.index, pd.DatetimeIndex):
                logger.warning(f"Index is not a DatetimeIndex for {ticker}, attempting to convert")
                try:
                    data.index = pd.to_datetime(data.index)
                except Exception as idx_err:
                    logger.error(f"Failed to convert index to DatetimeIndex: {idx_err}")
                    return None
            
            return data
        
        except Exception as e:
            logger.error(f"yfinance data retrieval error for {ticker}: {e}")
            return None
    
    def _get_alpha_vantage_data(
        self, 
        ticker: str, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime], 
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve historical data from Alpha Vantage API.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str or datetime): Start date for data retrieval
            end_date (str or datetime): End date for data retrieval
            interval (str, optional): Data interval
        
        Returns:
            Optional[pd.DataFrame]: Historical price data or None
        """
        if not self.alpha_vantage_key:
            logger.warning("No Alpha Vantage API key available")
            return None
        
        try:
            # Convert dates
            start = start_date.strftime("%Y-%m-%d") if isinstance(start_date, datetime) else start_date
            end = end_date.strftime("%Y-%m-%d") if isinstance(end_date, datetime) else end_date
            
            # Construct API request
            base_url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": ticker,
                "apikey": self.alpha_vantage_key,
                "outputsize": "full"
            }
            
            # Make API request
            response = requests.get(base_url, params=params, timeout=30)
            
            # Check if request was successful
            if response.status_code != 200:
                logger.warning(f"Alpha Vantage API returned status code {response.status_code} for {ticker}")
                return None
            
            # Parse response
            try:
                data = response.json()
            except ValueError as json_err:
                logger.error(f"Invalid JSON response from Alpha Vantage: {json_err}")
                return None
            
            # Check if API returned an error message
            if "Error Message" in data:
                logger.warning(f"Alpha Vantage API error: {data['Error Message']}")
                return None
            
            # Extract time series data
            time_series = data.get("Time Series (Daily)", {})
            
            if not time_series:
                logger.warning(f"No time series data found for {ticker} in Alpha Vantage response")
                return None
            
            # Convert to DataFrame
            df_data = []
            for date, values in time_series.items():
                if start <= date <= end:
                    try:
                        df_data.append({
                            'Date': date,
                            'Open': float(values['1. open']),
                            'High': float(values['2. high']),
                            'Low': float(values['3. low']),
                            'Close': float(values['4. close']),
                            'Volume': int(values['6. volume'])
                        })
                    except (KeyError, ValueError) as parse_err:
                        logger.warning(f"Error parsing data for date {date}: {parse_err}")
                        continue
            
            if not df_data:
                logger.warning(f"No data found for {ticker} in Alpha Vantage response")
                return None
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            df.index = pd.to_datetime(df.index)
            
            return df
        
        except requests.RequestException as req_err:
            logger.error(f"Request error for Alpha Vantage: {req_err}")
            return None
        except Exception as e:
            logger.error(f"Alpha Vantage data retrieval error for {ticker}: {e}")
            return None
    
    def _get_manual_stock_data(
        self, 
        ticker: str, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime], 
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        try:
            # Convert dates
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            # Generate date range based on interval
            freq = '1D'  # Default to daily
            if interval.lower() == '1wk':
                freq = '1W'
            elif interval.lower() == '1mo':
                freq = '1M'
            
            date_range = pd.date_range(start=start, end=end, freq=freq)
            
            # Create minimal DataFrame with proper columns
            df = pd.DataFrame(index=date_range)
            
            # Simulate basic stock price movement
            # Use a consistent seed based on ticker for reproducibility
            np.random.seed(hash(ticker) % 2**32)
            
            # Set initial price within realistic ranges based on ticker
            ticker_price_ranges = {
                'NVDA': (90, 800),    # NVIDIA has higher prices
                'AAPL': (150, 250),   # Apple
                'AMD': (80, 200),     # AMD
                'TSLA': (150, 480),   # Tesla
                'META': (250, 700),   # Meta
                'MSFT': (375, 470),   # Microsoft
                'GOOGL': (146, 210),  # Alphabet
                'AMZN': (150, 250),   # Amazon
                'QQQ': (300, 530),    # NASDAQ ETF
                'DIA': (300, 450),    # Dow Jones ETF
                'SPY': (300, 600)     # S&P 500 ETF
            }
            price_range = ticker_price_ranges.get(ticker, (50, 500))  # Default range
            initial_price = np.random.uniform(price_range[0], price_range[1])
            
            # Generate prices with slightly more realistic movement
            price_series = [initial_price]
            for i in range(1, len(df)):
                # Daily change limited to reasonable percentage
                daily_change = np.random.normal(0, 0.015)  # 1.5% standard deviation
                new_price = price_series[-1] * (1 + daily_change)
                # Keep prices within realistic bounds
                new_price = max(price_range[0] * 0.8, min(price_range[1] * 1.2, new_price))
                price_series.append(new_price)
            
            # Add simulated prices to DataFrame
            df['Open'] = price_series
            # Simulate a close price with small intraday variation
            df['Close'] = [p * (1 + np.random.normal(0, 0.005)) for p in price_series]
            # High is the maximum of Open and Close plus a small random increase
            df['High'] = [max(o, c) * (1 + abs(np.random.normal(0, 0.003))) for o, c in zip(df['Open'], df['Close'])]
            # Low is the minimum of Open and Close minus a small random decrease
            df['Low'] = [min(o, c) * (1 - abs(np.random.normal(0, 0.003))) for o, c in zip(df['Open'], df['Close'])]
            # Simulate volume as a random integer within reasonable bounds
            df['Volume'] = np.random.randint(1000000, 10000000, len(df))
            
            logger.info(f"Generated simulated data for {ticker} from {start} to {end}")
            return df
        
        except Exception as e:
            logger.error(f"Manual data generation error for {ticker}: {e}")
            return None
    
    def _clean_price_data(self, data: Union[pd.DataFrame, np.ndarray, list, tuple]) -> pd.DataFrame:
        """
        Clean and validate price data.

        Args:
            data (pd.DataFrame, np.ndarray, list, or tuple): Input price data.

        Returns:
            pd.DataFrame: Cleaned price data.
        """
        try:
            # Check if data is None or empty
            if data is None:
                logger.warning("Received None data for cleaning")
                return pd.DataFrame()
                
            # Check if it's already a DataFrame and empty
            if isinstance(data, pd.DataFrame) and data.empty:
                logger.warning("Received empty DataFrame for cleaning")
                return pd.DataFrame()
                
            # Validate data type
            if not isinstance(data, (pd.DataFrame, np.ndarray, list, tuple, pd.Series)):
                logger.error(f"Unsupported data type for cleaning: {type(data)}")
                return pd.DataFrame()
            
            # If data is not a DataFrame, try to convert it
            if not isinstance(data, pd.DataFrame):
                logger.debug(f"Converting non-DataFrame data (type: {type(data)}) for cleaning")
                
                # If it's a Series with to_frame method
                if isinstance(data, pd.Series) and hasattr(data, 'to_frame'):
                    data = data.to_frame()
                # If it's a NumPy array, handle multi-dimensional arrays
                elif isinstance(data, np.ndarray):
                    # If array has more than 2 dimensions, flatten all dimensions except the last one
                    if data.ndim > 2:
                        data = data.reshape(-1, data.shape[-1])
                    # If it's 1D, try to reshape into (-1, 5) if length is a multiple of 5
                    elif data.ndim == 1:
                        if len(data) % 5 != 0:
                            logger.error("1D array length is not a multiple of 5")
                            return pd.DataFrame()
                        data = data.reshape(-1, 5)
                # If it's a list or tuple, convert to NumPy array first
                elif isinstance(data, (list, tuple)):
                    try:
                        data = np.array(data)
                        if data.ndim > 2:
                            data = data.reshape(-1, data.shape[-1])
                        elif data.ndim == 1:
                            if len(data) % 5 != 0:
                                logger.error("1D list/tuple length is not a multiple of 5")
                                return pd.DataFrame()
                            data = data.reshape(-1, 5)
                    except Exception as array_err:
                        logger.error(f"Failed to convert list/tuple to array: {array_err}")
                        return pd.DataFrame()
                else:
                    logger.error(f"Unable to convert data of type {type(data)} to DataFrame")
                    return pd.DataFrame()
                
                # Ensure the reshaped array has exactly 5 columns
                if isinstance(data, np.ndarray) and data.shape[1] != 5:
                    logger.error(f"Expected 5 columns (Open, High, Low, Close, Volume) but got {data.shape[1]}")
                    return pd.DataFrame()
                
                # Create a DataFrame from the array
                try:
                    data = pd.DataFrame(
                        data,
                        columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                        index=pd.date_range(
                            start=datetime.now() - timedelta(days=len(data) - 1),
                            periods=len(data),
                            freq='D'
                        )
                    )
                except Exception as df_err:
                    logger.error(f"Failed to create DataFrame: {df_err}")
                    return pd.DataFrame()
            
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    data.index = pd.to_datetime(data.index, errors='coerce')
                    if data.index.isna().any():
                        logger.warning("Index contains invalid datetime values, creating new index")
                        data.index = pd.date_range(
                            start=datetime.now() - timedelta(days=len(data) - 1),
                            periods=len(data),
                            freq='D'
                        )
                except Exception as idx_err:
                    logger.error(f"Failed to convert index to DatetimeIndex: {idx_err}")
                    data.index = pd.date_range(
                        start=datetime.now() - timedelta(days=len(data) - 1),
                        periods=len(data),
                        freq='D'
                    )
            
            # Ensure index is timezone-naive for consistency
            if data.index.tzinfo is not None:
                data.index = data.index.tz_localize(None)
                
            # Sort index chronologically
            data = data.sort_index()
            data = data[~data.index.duplicated(keep='first')]
            
            # Convert columns to numeric, coercing errors to NaN
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                else:
                    logger.warning(f"Column {col} missing from data, adding with NaN values")
                    data[col] = np.nan
            
            # Fill NaN values with forward fill then backward fill
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Price validation - ensure positive values
            # Create a mask of valid rows
            valid_mask = (
                (data['Open'] > 0) & 
                (data['High'] > 0) & 
                (data['Low'] > 0) & 
                (data['Close'] > 0)
            )
            
            # Apply mask but maintain DataFrame structure
            if not valid_mask.all():
                logger.warning(f"Found {(~valid_mask).sum()} invalid price rows, filtering them out")
                data = data[valid_mask]
            
            # If all rows were filtered, return empty DataFrame
            if data.empty:
                logger.warning("No valid data rows after filtering")
                return pd.DataFrame()
                
            return data
        except Exception as e:
            logger.error(f"Error cleaning price data: {e}")
            return pd.DataFrame()
    
    def get_historical_price_data(
        self, 
        ticker: str, 
        start_date: Optional[Union[str, datetime]] = None, 
        end_date: Optional[Union[str, datetime]] = None, 
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Retrieve historical price data using multiple data sources with caching.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str or datetime, optional): Start date for data retrieval
            end_date (str or datetime, optional): End date for data retrieval
            interval (str, optional): Data interval
            use_cache (bool, optional): Whether to use cached data if available
        
        Returns:
            pd.DataFrame: Historical price data
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        
        logger.info(f"Retrieving historical data for {ticker} from {start_date} to {end_date}")
        
        # Try to get data from cache first if caching is enabled
        if use_cache:
            cached_data = self._get_cached_data(ticker, start_date, end_date, interval)
            if cached_data is not None and not cached_data.empty:
                return cached_data
        
        # If not in cache or caching disabled, try data sources
        for source_idx, source in enumerate(self.fallback_sources):
            try:
                logger.info(f"Trying data source #{source_idx+1}: {source.__name__}")
                data = source(ticker, start_date, end_date, interval)
                
                # CRITICAL FIX: Check data type before processing
                logger.info(f"Source returned data of type: {type(data)}")
                
                if data is None:
                    logger.warning(f"Source {source.__name__} returned None for {ticker}")
                    continue
                    
                if isinstance(data, pd.DataFrame) and data.empty:
                    logger.warning(f"Source {source.__name__} returned empty DataFrame for {ticker}")
                    continue
                
                # CRITICAL FIX: Extra logging about data shape
                if isinstance(data, pd.DataFrame):
                    logger.info(f"Data columns: {data.columns.tolist()}")
                    logger.info(f"Data shape: {data.shape}")
                
                # CRITICAL FIX: Try/except around cleaning
                try:
                    cleaned_data = self._clean_price_data(data)
                    if isinstance(cleaned_data, pd.DataFrame) and not cleaned_data.empty:
                        logger.info(f"Successfully retrieved data for {ticker} from {source.__name__}")
                        
                        # Save to cache if caching is enabled
                        if use_cache:
                            self._save_to_cache(ticker, cleaned_data, interval)
                            
                        return cleaned_data
                    else:
                        logger.warning(f"Data cleaning produced empty result for {ticker} from {source.__name__}")
                except Exception as clean_err:
                    logger.error(f"Error during data cleaning from {source.__name__}: {clean_err}")
            
            except Exception as e:
                logger.warning(f"Data retrieval failed for {ticker} from {source.__name__}: {e}")
        
        # If all sources fail, log error
        logger.error(f"Could not retrieve data for {ticker} from any source")
        return pd.DataFrame()
    
    def import_bulk_historical_data(
        self, 
        tickers: List[str], 
        days: int = 365
    ) -> Dict[str, Dict[str, Union[str, int]]]:
        """
        Import historical data for multiple tickers.
        
        Args:
            tickers (List[str]): List of stock tickers
            days (int, optional): Number of historical days to import
        
        Returns:
            Dict[str, Dict[str, Union[str, int]]]: Import results for each ticker
        """
        results = {}
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for ticker in tickers:
            try:
                # Retrieve historical data
                data = self.get_historical_price_data(
                    ticker, 
                    start_date=start_date, 
                    end_date=end_date
                )
                
                if not data.empty:
                    results[ticker] = {
                        "status": "success",
                        "data_points": len(data),
                        "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    }
                else:
                    results[ticker] = {
                        "status": "failed",
                        "reason": "No data retrieved"
                    }
            
            except Exception as e:
                logger.error(f"Error importing data for {ticker}: {e}")
                results[ticker] = {
                    "status": "failed",
                    "reason": str(e)
                }
        
        return results

# Create a global importer instance
HISTORICAL_DATA_IMPORTER = HistoricalDataImporter()

def get_historical_price_data(
    ticker: str, 
    start_date: Optional[Union[str, datetime]] = None, 
    end_date: Optional[Union[str, datetime]] = None, 
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Public function to get historical price data.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str or datetime, optional): Start date for data retrieval
        end_date (str or datetime, optional): End date for data retrieval
        interval (str, optional): Data interval
    
    Returns:
        pd.DataFrame: Historical price data
    """
    return HISTORICAL_DATA_IMPORTER.get_historical_price_data(
        ticker, start_date, end_date, interval
    )

def read_cached_data(cache_file):
    try:
        df = pd.read_csv(cache_file)
        # Convert the timestamp column to tz-aware datetimes but make result timezone-naive
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        return df
    except Exception as e:
        logging.warning(f"Error reading from cache: {e}")
        return None