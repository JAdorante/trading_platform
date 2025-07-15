import sqlite3
import os
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import shutil
import threading
import secrets
import queue
import functools
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("database")

DB_PATH = "trading.db"
DB_LOCK = threading.Lock()

# Enhanced Connection Pool
class RobustConnectionPool:
    def __init__(self, max_connections=10, timeout=30):
        self.pool = queue.Queue(maxsize=max_connections)
        self.max_connections = max_connections
        self.timeout = timeout
        self.active_connections = 0
        self.lock = threading.Lock()
        
        # Pre-populate pool with connections
        for _ in range(max_connections // 2):  # Start with half the max connections
            conn = self._create_connection()
            if conn:
                self.pool.put(conn)
    
    def _create_connection(self):
        """Create a new database connection with robust settings."""
        try:
            conn = sqlite3.connect(
                DB_PATH, 
                timeout=self.timeout,  # Increase timeout
                isolation_level=None,  # Enable autocommit mode
                check_same_thread=False  # Allow cross-thread usage
            )
            
            # Optimize connection settings
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")  # Write-Ahead Logging
            cursor.execute("PRAGMA synchronous=NORMAL;")  # Balanced durability and performance
            cursor.execute("PRAGMA busy_timeout=30000;")  # 30 second busy wait
            cursor.execute("PRAGMA cache_size=-8000;")  # 8MB cache
            
            return conn
        except sqlite3.Error as e:
            logger.error(f"Connection creation error: {e}")
            return None
    
    def get_connection(self):
        """
        Get a database connection with robust error handling.
        
        Returns:
            sqlite3.Connection: Database connection
        """
        try:
            # Try to get a connection from the pool
            try:
                return self.pool.get_nowait()
            except queue.Empty:
                # If pool is empty, create a new connection if under max limit
                with self.lock:
                    if self.active_connections < self.max_connections:
                        self.active_connections += 1
                        return self._create_connection()
                    
            # If no connections available, wait with timeout
            try:
                return self.pool.get(timeout=self.timeout)
            except queue.Empty:
                logger.warning("Connection pool timeout - creating emergency connection")
                return self._create_connection()
        
        except Exception as e:
            logger.error(f"Unexpected error getting connection: {e}")
            return None
    
    def release_connection(self, conn):
        """
        Release a connection back to the pool.
        
        Args:
            conn (sqlite3.Connection): Connection to release
        """
        if conn is None:
            return
        
        try:
            # Attempt to put connection back in pool
            try:
                self.pool.put_nowait(conn)
            except queue.Full:
                # If pool is full, close the connection
                with self.lock:
                    self.active_connections -= 1
                conn.close()
        except Exception as e:
            logger.error(f"Error releasing connection: {e}")
            try:
                conn.close()
            except:
                pass

# Create global connection pool
CONNECTION_POOL = RobustConnectionPool()

# Decorator for database operations with improved locking
def with_db_lock(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 5
        retry_delay = 0.5  # seconds
        
        for attempt in range(max_retries):
            conn = None
            try:
                # Get connection from pool
                conn = CONNECTION_POOL.get_connection()
                
                if conn is None:
                    logger.error(f"Failed to get database connection for {func.__name__}")
                    return None
                
                # Execute function with connection
                result = func(conn, *args, **kwargs)
                return result
            
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    # Release connection back to pool before retry
                    if conn:
                        CONNECTION_POOL.release_connection(conn)
                        conn = None
                    
                    # Log retry attempt
                    logger.warning(f"Database locked, retrying in {retry_delay}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    logger.error(f"Database error in {func.__name__}: {e}")
                    return None
            
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                return None
            
            finally:
                # Always release connection back to pool
                if conn:
                    CONNECTION_POOL.release_connection(conn)
                    
        return None  # If we get here, all retries failed
    
    return wrapper

class DBConnection:
    """Context manager for database connections."""
    def __init__(self):
        """Initialize the context manager."""
        self.conn = None
    
    def __enter__(self):
        """
        Enter the runtime context related to this object.
        
        Returns:
            sqlite3.Connection: Database connection
        """
        self.conn = CONNECTION_POOL.get_connection()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context related to this object.
        
        Handles releasing the connection back to the pool.
        """
        if self.conn:
            CONNECTION_POOL.release_connection(self.conn)

def init_db():
    """Ensure the database is created and required tables exist."""
    try:
        with DBConnection() as conn:
            c = conn.cursor()
            # Increase timeout to 30 seconds
            c.execute("PRAGMA busy_timeout=30000;")
            # Use Write-Ahead Logging for better concurrency
            c.execute("PRAGMA journal_mode=WAL;")
            # Reduce fsync operations
            c.execute("PRAGMA synchronous=NORMAL;")
            # Set a reasonable cache size
            c.execute("PRAGMA cache_size=-8000;") # 8MB cache
            
            # Original tables
            c.execute('''CREATE TABLE IF NOT EXISTS raw_data (
                         id INTEGER PRIMARY KEY AUTOINCREMENT, 
                         timestamp TEXT, 
                         source TEXT, 
                         text TEXT, 
                         keyword_matched TEXT, 
                         stock_ticker TEXT,
                         url TEXT)''')
                         
            c.execute('''CREATE TABLE IF NOT EXISTS scores (
                         id INTEGER PRIMARY KEY AUTOINCREMENT, 
                         timestamp TEXT, 
                         stock_ticker TEXT, 
                         score REAL, 
                         explanation TEXT)''')
                         
            c.execute('''CREATE TABLE IF NOT EXISTS price_predictions (
                         id INTEGER PRIMARY KEY AUTOINCREMENT, 
                         timestamp TEXT, 
                         stock_ticker TEXT, 
                         predicted_change REAL)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS technical_indicators (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         timestamp TEXT,
                         stock_ticker TEXT,
                         indicator_name TEXT,
                         indicator_value REAL)''')
                         
            c.execute('''CREATE TABLE IF NOT EXISTS topics (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         timestamp TEXT,
                         document_id INTEGER,
                         topic_id INTEGER,
                         topic_description TEXT,
                         probability REAL,
                         FOREIGN KEY(document_id) REFERENCES raw_data(id))''')
                         
            c.execute('''CREATE TABLE IF NOT EXISTS entities (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         timestamp TEXT,
                         document_id INTEGER,
                         entity_type TEXT,
                         entity_text TEXT,
                         FOREIGN KEY(document_id) REFERENCES raw_data(id))''')
            
            # New tables for enhanced components
            
            # Source reliability for sentiment weighting
            c.execute('''CREATE TABLE IF NOT EXISTS source_reliability (
                         source TEXT PRIMARY KEY,
                         reliability_score REAL,
                         last_updated TEXT)''')
            
            # Historical price data
            c.execute('''CREATE TABLE IF NOT EXISTS price_history (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         ticker TEXT,
                         date TEXT,
                         open REAL,
                         high REAL,
                         low REAL,
                         close REAL,
                         volume REAL,
                         adj_close REAL,
                         source TEXT,
                         imported_at TEXT)''')
            
            # Content quality metrics
            c.execute('''CREATE TABLE IF NOT EXISTS content_quality (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         document_id INTEGER,
                         quality_score REAL,
                         is_spam INTEGER,
                         is_duplicate INTEGER,
                         filter_reason TEXT,
                         timestamp TEXT,
                         FOREIGN KEY(document_id) REFERENCES raw_data(id))''')
            
            # Backtest results
            c.execute('''CREATE TABLE IF NOT EXISTS backtest_results (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         ticker TEXT,
                         start_date TEXT,
                         end_date TEXT,
                         total_trades INTEGER,
                         win_rate REAL,
                         total_return REAL,
                         buy_hold_return REAL,
                         sharpe_ratio REAL,
                         timestamp TEXT)''')
                         
            # Create indexes for faster queries
            c.execute('''CREATE INDEX IF NOT EXISTS idx_raw_data_ticker ON raw_data(stock_ticker)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_raw_data_timestamp ON raw_data(timestamp)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_scores_ticker ON scores(stock_ticker)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_scores_timestamp ON scores(timestamp)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON price_predictions(stock_ticker)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_price_history_ticker_date ON price_history(ticker, date)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_topics_document ON topics(document_id)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_entities_document ON entities(document_id)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)''')
            
            conn.commit()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

@with_db_lock
def save_score(conn, ticker, score, explanation):
    """
    Save sentiment scores with robust error handling.
    
    Args:
        conn (sqlite3.Connection): Database connection
        ticker (str): Stock ticker
        score (float): Sentiment score
        explanation (str): Explanation for the score
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()
        timestamp = datetime.utcnow().isoformat()
        
        cursor.execute('''INSERT INTO scores 
                         (timestamp, stock_ticker, score, explanation) 
                         VALUES (?, ?, ?, ?)''',
                      (timestamp, ticker, score, explanation))
        
        conn.commit()
        logger.debug(f"Score saved: {ticker} - {score:.2f}")
        return True
    
    except sqlite3.Error as e:
        logger.error(f"Database Error (save_score): {e}")
        conn.rollback()
        return False
    except Exception as e:
        logger.error(f"Unexpected Error (save_score): {e}")
        conn.rollback()
        return False

@with_db_lock
def save_raw_data(conn, source, text, keyword, ticker, url=None):
    """
    Save raw data with robust connection handling.
    
    Args:
        conn (sqlite3.Connection): Database connection
        source (str): Data source
        text (str): Text content
        keyword (str): Keyword matched
        ticker (str): Stock ticker
        url (str, optional): Source URL
    
    Returns:
        int or None: Document ID if successful, None otherwise
    """
    try:
        cursor = conn.cursor()
        timestamp = datetime.utcnow().isoformat()
        
        cursor.execute('''INSERT INTO raw_data 
                         (timestamp, source, text, keyword_matched, stock_ticker, url) 
                         VALUES (?, ?, ?, ?, ?, ?)''', 
                      (timestamp, source, text, keyword, ticker, url))
        
        document_id = cursor.lastrowid
        conn.commit()
        
        logger.debug(f"Data inserted: {source} - {ticker}")
        return document_id
    
    except sqlite3.Error as e:
        logger.error(f"Database Error (save_raw_data): {e}")
        conn.rollback()
        return None
    except Exception as e:
        logger.error(f"Unexpected Error (save_raw_data): {e}")
        conn.rollback()
        return None

@with_db_lock
def save_price_prediction(conn, ticker, predicted_change):
    """
    Save predicted price change with robust error handling.
    
    Args:
        conn (sqlite3.Connection): Database connection
        ticker (str): Stock ticker
        predicted_change (float): Predicted price change
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()
        timestamp = datetime.utcnow().isoformat()
        
        cursor.execute('''INSERT INTO price_predictions 
                         (timestamp, stock_ticker, predicted_change) 
                         VALUES (?, ?, ?)''',
                      (timestamp, ticker, predicted_change))
        
        conn.commit()
        logger.info(f"Price prediction saved: {ticker} - Change: {predicted_change:.4f}")
        return True
    
    except sqlite3.Error as e:
        logger.error(f"Database Error (save_price_prediction): {e}")
        conn.rollback()
        return False
    except Exception as e:
        logger.error(f"Unexpected Error (save_price_prediction): {e}")
        conn.rollback()
        return False

@with_db_lock
def save_technical_indicators(conn, ticker, indicators_dict):
    """
    Save technical indicators with robust error handling and transaction support.
    
    Args:
        conn (sqlite3.Connection): Database connection
        ticker (str): Stock ticker
        indicators_dict (dict): Dictionary of technical indicators
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()
        timestamp = datetime.utcnow().isoformat()
        
        # Use a single transaction for all inserts
        cursor.execute("BEGIN TRANSACTION")
        
        try:
            for indicator_name, indicator_value in indicators_dict.items():
                cursor.execute('''INSERT INTO technical_indicators 
                                 (timestamp, stock_ticker, indicator_name, indicator_value) 
                                 VALUES (?, ?, ?, ?)''',
                              (timestamp, ticker, indicator_name, indicator_value))
            
            cursor.execute("COMMIT")
            logger.info(f"Technical indicators saved for {ticker}")
            return True
            
        except Exception as tx_error:
            cursor.execute("ROLLBACK")
            logger.error(f"Transaction error in save_technical_indicators: {tx_error}")
            return False
    
    except Exception as e:
        logger.error(f"Error in save_technical_indicators: {e}")
        try:
            cursor.execute("ROLLBACK")
        except:
            pass
        return False

# Additional functions like save_topic, save_entities, etc. would be similarly updated
# [Keep other existing functions from the original database.py]

def backup_database(conn):
    """
    Create a backup of the current database.
    
    Args:
        conn (sqlite3.Connection): Database connection
    
    Returns:
        bool: True if backup successful, False otherwise
    """
    try:
        backups_dir = "backups"
        os.makedirs(backups_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_filename = f"trading_backup_{timestamp}.db"
        backup_path = os.path.join(backups_dir, backup_filename)
        
        # Create a backup by copying the current database file
        shutil.copy2(DB_PATH, backup_path)
        
        # Also save a JSON summary of database stats
        cursor = conn.cursor()
        stats = {
            'timestamp': timestamp,
            'tables': {}
        }
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get count for each table
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            stats['tables'][table] = count
        
        # Save as JSON
        stats_path = os.path.join(backups_dir, f"db_stats_{timestamp}.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Database backup created at {backup_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating database backup: {e}")
        return False

def optimize_database():
    """
    Optimize the database for better performance.
    
    This function reclaims unused space, updates statistics, optimizes indexes,
    and performs an integrity check.
    
    Returns:
        bool: True if optimization succeeded, False otherwise.
    """
    try:
        with DBConnection() as conn:
            cursor = conn.cursor()
            # Vacuum to reclaim unused space and defragment the database
            cursor.execute("VACUUM")
            # Update statistics
            cursor.execute("ANALYZE")
            # Optimize indexes
            cursor.execute("PRAGMA optimize")
            # Integrity check to verify database health
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            if integrity_result != "ok":
                logger.error(f"Database integrity check failed: {integrity_result}")
                return False
            conn.commit()
        logger.info("Database optimization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error optimizing database: {e}")
        return False

@with_db_lock
def save_topic(conn, document_id, topic_id, topic_description, probability):
    """
    Save topic information with robust error handling.
    
    Args:
        conn (sqlite3.Connection): Database connection
        document_id (int): ID of the source document
        topic_id (int): Unique identifier for the topic
        topic_description (str): Description of the topic
        probability (float): Probability of the topic
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()
        timestamp = datetime.utcnow().isoformat()
        
        cursor.execute('''INSERT INTO topics 
                         (timestamp, document_id, topic_id, topic_description, probability) 
                         VALUES (?, ?, ?, ?, ?)''',
                      (timestamp, document_id, topic_id, topic_description, probability))
        
        conn.commit()
        logger.debug(f"Topic saved for document {document_id}: {topic_description}")
        return True
    
    except sqlite3.Error as e:
        logger.error(f"Database Error (save_topic): {e}")
        conn.rollback()
        return False
    except Exception as e:
        logger.error(f"Unexpected Error (save_topic): {e}")
        conn.rollback()
        return False

@with_db_lock
def save_entities(conn, document_id, entities_dict):
    """
    Save named entities with robust error handling and transaction support.
    
    Args:
        conn (sqlite3.Connection): Database connection
        document_id (int): ID of the source document
        entities_dict (dict): Dictionary of entities by type
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()
        timestamp = datetime.utcnow().isoformat()
        
        # Use a single transaction for all entity inserts
        cursor.execute("BEGIN TRANSACTION")
        
        try:
            for entity_type, entities_list in entities_dict.items():
                for entity_text in entities_list:
                    cursor.execute('''INSERT INTO entities 
                                    (timestamp, document_id, entity_type, entity_text) 
                                    VALUES (?, ?, ?, ?)''',
                                (timestamp, document_id, entity_type, entity_text))
            
            cursor.execute("COMMIT")
            logger.debug(f"Entities saved for document {document_id}")
            return True
            
        except Exception as tx_error:
            cursor.execute("ROLLBACK")
            logger.error(f"Transaction error in save_entities: {tx_error}")
            return False
    
    except Exception as e:
        logger.error(f"Error in save_entities: {e}")
        try:
            cursor.execute("ROLLBACK")
        except:
            pass
        return False

@with_db_lock
def save_content_quality(conn, document_id, quality_score, is_spam, is_duplicate, filter_reason=None):
    """
    Save content quality metrics with robust error handling.
    
    Args:
        conn (sqlite3.Connection): Database connection
        document_id (int): ID of the source document
        quality_score (float): Quality score of the content
        is_spam (bool): Whether the content is spam
        is_duplicate (bool): Whether the content is a duplicate
        filter_reason (str, optional): Reason for filtering
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()
        timestamp = datetime.utcnow().isoformat()
        
        cursor.execute('''INSERT INTO content_quality 
                         (document_id, quality_score, is_spam, is_duplicate, filter_reason, timestamp) 
                         VALUES (?, ?, ?, ?, ?, ?)''',
                      (document_id, quality_score, 1 if is_spam else 0, 
                       1 if is_duplicate else 0, filter_reason, timestamp))
        
        conn.commit()
        logger.debug(f"Content quality metrics saved for document {document_id}")
        return True
    
    except sqlite3.Error as e:
        logger.error(f"Database Error (save_content_quality): {e}")
        conn.rollback()
        return False
    except Exception as e:
        logger.error(f"Unexpected Error (save_content_quality): {e}")
        conn.rollback()
        return False

@with_db_lock
def save_source_reliability(conn, source, reliability_score):
    """
    Save or update source reliability score.
    
    Args:
        conn (sqlite3.Connection): Database connection
        source (str): Data source
        reliability_score (float): Reliability score of the source
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()
        timestamp = datetime.utcnow().isoformat()
        
        cursor.execute('''INSERT OR REPLACE INTO source_reliability 
                         (source, reliability_score, last_updated)
                         VALUES (?, ?, ?)''',
                      (source, reliability_score, timestamp))
        
        conn.commit()
        logger.info(f"Source reliability updated: {source} - {reliability_score:.2f}")
        return True
    
    except sqlite3.Error as e:
        logger.error(f"Database Error (save_source_reliability): {e}")
        conn.rollback()
        return False
    except Exception as e:
        logger.error(f"Unexpected Error (save_source_reliability): {e}")
        conn.rollback()
        return False

@with_db_lock
def save_backtest_results(conn, ticker, results):
    """
    Save model backtest results.
    
    Args:
        conn (sqlite3.Connection): Database connection
        ticker (str): Stock ticker
        results (dict): Backtest results dictionary
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()
        timestamp = datetime.utcnow().isoformat()
        
        cursor.execute('''INSERT INTO backtest_results 
                         (ticker, start_date, end_date, total_trades, win_rate, 
                          total_return, buy_hold_return, sharpe_ratio, timestamp) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (ticker, results['start_date'], results['end_date'], 
                       results['total_trades'], results['win_rate'],
                       results['total_return'], results['buy_hold_return'], 
                       results['sharpe_ratio'], timestamp))
        
        conn.commit()
        logger.info(f"Backtest results saved for {ticker}")
        return True
    
    except sqlite3.Error as e:
        logger.error(f"Database Error (save_backtest_results): {e}")
        conn.rollback()
        return False
    except Exception as e:
        logger.error(f"Unexpected Error (save_backtest_results): {e}")
        conn.rollback()
        return False

def get_recent_sentiment(ticker, days=7):
    """
    Get recent sentiment data for a ticker.
    
    Args:
        ticker (str): Stock ticker
        days (int, optional): Number of days to look back. Defaults to 7.
    
    Returns:
        pd.DataFrame: DataFrame with sentiment data
    """
    try:
        with DBConnection() as conn:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            query = """
                SELECT timestamp, score
                FROM scores
                WHERE stock_ticker = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            start_iso = start_date.isoformat()
            end_iso = end_date.isoformat()
            sentiment_df = pd.read_sql_query(query, conn, params=(ticker, start_iso, end_iso))
        return sentiment_df
    
    except Exception as e:
        logger.error(f"Error retrieving recent sentiment for {ticker}: {e}")
        return pd.DataFrame()

def get_stock_mentions(days=30, limit=10):
    """
    Get the most mentioned stock tickers.
    
    Args:
        days (int, optional): Number of days to look back. Defaults to 30.
        limit (int, optional): Maximum number of tickers to return. Defaults to 10.
    
    Returns:
        pd.DataFrame: DataFrame with stock mention counts
    """
    try:
        with DBConnection() as conn:
            start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            query = """
                SELECT stock_ticker, COUNT(*) as mention_count
                FROM raw_data
                WHERE timestamp > ?
                GROUP BY stock_ticker
                ORDER BY mention_count DESC
                LIMIT ?
            """
            mentions_df = pd.read_sql_query(query, conn, params=(start_date, limit))
        return mentions_df
    
    except Exception as e:
        logger.error(f"Error retrieving stock mentions: {e}")
        return pd.DataFrame()

def get_topic_trends(days=30, limit=10):
    """
    Get trending topics across all stocks.
    
    Args:
        days (int, optional): Number of days to look back. Defaults to 30.
        limit (int, optional): Maximum number of topics to return. Defaults to 10.
    
    Returns:
        pd.DataFrame: DataFrame with topic trends
    """
    try:
        with DBConnection() as conn:
            start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            query = """
                SELECT topic_description, COUNT(*) as topic_count, 
                       AVG(probability) as avg_probability
                FROM topics t
                JOIN raw_data r ON t.document_id = r.id
                WHERE r.timestamp > ?
                GROUP BY topic_description
                ORDER BY topic_count DESC
                LIMIT ?
            """
            topics_df = pd.read_sql_query(query, conn, params=(start_date, limit))
        return topics_df
    
    except Exception as e:
        logger.error(f"Error retrieving topic trends: {e}")
        return pd.DataFrame()

def get_entity_network(entity_type="ORG", days=30, min_occurrences=2):
    """
    Get network of co-occurring entities.
    
    Args:
        entity_type (str, optional): Type of entity to analyze. Defaults to "ORG".
        days (int, optional): Number of days to look back. Defaults to 30.
        min_occurrences (int, optional): Minimum number of occurrences. Defaults to 2.
    
    Returns:
        pd.DataFrame: DataFrame with entity network relationships
    """
    try:
        with DBConnection() as conn:
            start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            # First get entities that appear frequently enough
            query1 = """
                SELECT entity_text, COUNT(*) as count
                FROM entities e
                JOIN raw_data r ON e.document_id = r.id
                WHERE e.entity_type = ? AND r.timestamp > ?
                GROUP BY entity_text
                HAVING count >= ?
            """
            entities_df = pd.read_sql_query(query1, conn, params=(entity_type, start_date, min_occurrences))
            
            if entities_df.empty:
                return pd.DataFrame()
            
            # Then get co-occurrences
            entity_network = []
            for _, row in entities_df.iterrows():
                entity = row['entity_text']
                
                # Get documents with this entity
                query2 = """
                    SELECT document_id 
                    FROM entities
                    WHERE entity_type = ? AND entity_text = ?
                """
                docs_df = pd.read_sql_query(query2, conn, params=(entity_type, entity))
                
                if docs_df.empty:
                    continue
                    
                # Get other entities in these documents
                doc_ids = tuple(docs_df['document_id'].tolist())
                if len(doc_ids) == 1:
                    doc_ids = f"({doc_ids[0]})"  # Special case for single item
                
                query3 = f"""
                    SELECT entity_text, COUNT(*) as count
                    FROM entities
                    WHERE entity_type = ? AND document_id IN {doc_ids}
                    AND entity_text != ?
                    GROUP BY entity_text
                    HAVING count >= ?
                """
                related_df = pd.read_sql_query(query3, conn, params=(entity_type, entity, min_occurrences))
                
                for _, related_row in related_df.iterrows():
                    entity_network.append({
                        'source': entity,
                        'target': related_row['entity_text'],
                        'weight': related_row['count']
                    })
            
            network_df = pd.DataFrame(entity_network)
        return network_df
    
    except Exception as e:
        logger.error(f"Error retrieving entity network: {e}")
        return pd.DataFrame()

@with_db_lock
def prune_old_data(conn, days_to_keep=180):
    """
    Remove old data to manage database size.
    
    Args:
        conn (sqlite3.Connection): Database connection
        days_to_keep (int, optional): Number of days to retain data. Defaults to 180.
    
    Returns:
        int: Number of pruned documents
    """
    try:
        cursor = conn.cursor()
        cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat()
        
        # Get old document IDs
        cursor.execute("SELECT id FROM raw_data WHERE timestamp < ?", (cutoff_date,))
        old_doc_ids = [row[0] for row in cursor.fetchall()]
        
        if not old_doc_ids:
            logger.info("No old data to prune")
            return 0
        
        # Delete from related tables first
        for doc_id in old_doc_ids:
            cursor.execute("DELETE FROM entities WHERE document_id = ?", (doc_id,))
            cursor.execute("DELETE FROM topics WHERE document_id = ?", (doc_id,))
            cursor.execute("DELETE FROM content_quality WHERE document_id = ?", (doc_id,))
        
        # Delete raw data
        cursor.execute("DELETE FROM raw_data WHERE timestamp < ?", (cutoff_date,))
        
        # Delete other old data
        cursor.execute("DELETE FROM scores WHERE timestamp < ?", (cutoff_date,))
        cursor.execute("DELETE FROM price_predictions WHERE timestamp < ?", (cutoff_date,))
        cursor.execute("DELETE FROM technical_indicators WHERE timestamp < ?", (cutoff_date,))
        
        # Vacuum to reclaim space (this can take a while)
        cursor.execute("VACUUM")
        
        conn.commit()
        pruned_count = len(old_doc_ids)
        logger.info(f"Pruned {pruned_count} old documents")
        return pruned_count
    
    except Exception as e:
        logger.error(f"Error pruning old data: {e}")
        conn.rollback()
        return 0

@with_db_lock
def backup_database(conn):
    """
    Create a backup of the current database.
    
    Args:
        conn (sqlite3.Connection): Database connection
    
    Returns:
        bool: True if backup successful, False otherwise
    """
    try:
        backups_dir = "backups"
        os.makedirs(backups_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_filename = f"trading_backup_{timestamp}.db"
        backup_path = os.path.join(backups_dir, backup_filename)
        
        # Create a backup by copying the current database file
        shutil.copy2(DB_PATH, backup_path)
        
        # Also save a JSON summary of database stats
        cursor = conn.cursor()
        stats = {
            'timestamp': timestamp,
            'tables': {}
        }
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get count for each table
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            stats['tables'][table] = count
        
        # Save as JSON
        stats_path = os.path.join(backups_dir, f"db_stats_{timestamp}.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Database backup created at {backup_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating database backup: {e}")
        return False

def update_multiple_predictions(predictions):
    """
    Save multiple predictions in a single transaction.
    
    Args:
        predictions (dict): Dictionary of ticker to predicted change
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with DBConnection() as conn:
            cursor = conn.cursor()
            timestamp = datetime.utcnow().isoformat()
            
            # Use explicit transaction for multiple inserts
            cursor.execute("BEGIN TRANSACTION")
            
            try:
                for ticker, predicted_change in predictions.items():
                    cursor.execute('''INSERT INTO price_predictions 
                                    (timestamp, stock_ticker, predicted_change) 
                                    VALUES (?, ?, ?)''',
                                (timestamp, ticker, predicted_change))
                
                cursor.execute("COMMIT")
                logger.info(f"Batch saved {len(predictions)} predictions")
                return True
                
            except Exception as tx_error:
                cursor.execute("ROLLBACK")
                logger.error(f"Transaction error in update_multiple_predictions: {tx_error}")
                return False
    
    except sqlite3.Error as e:
        logger.error(f"Database Error (update_multiple_predictions): {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected Error (update_multiple_predictions): {e}")
        return False

def database_maintenance(days_to_keep=180):
    """
    Perform comprehensive database maintenance.
    
    Args:
        days_to_keep (int, optional): Number of days to retain data. Defaults to 180.
    
    Returns:
        dict: Maintenance operation results
    """
    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'prune_data': False,
        'backup': False,
        'optimize': False
    }
    
    try:
        # Prune old data
        pruned_count = prune_old_data(days_to_keep=days_to_keep)
        results['prune_data'] = pruned_count > 0
        results['pruned_count'] = pruned_count
        
        # Create backup
        results['backup'] = backup_database()
        
        # Optimize database
        results['optimize'] = optimize_database()
        
        logger.info("Database maintenance completed successfully")
        return results
    
    except Exception as e:
        logger.error(f"Comprehensive database maintenance error: {e}")
        return results

# Run database maintenance in a background thread
def run_db_maintenance():
    """Run periodic database maintenance to prevent locks."""
    while True:
        try:
            logger.info("Running database maintenance")
            database_maintenance(days_to_keep=180)
        except Exception as e:
            logger.error(f"Error in database maintenance: {e}")
        finally:
            # Run every hour
            time.sleep(3600)

# Start maintenance thread if this module is imported directly
if __name__ != "__main__":
    try:
        maintenance_thread = threading.Thread(target=run_db_maintenance, daemon=True)
        maintenance_thread.start()
        logger.info("Started background maintenance thread")
    except Exception as e:
        logger.error(f"Failed to start maintenance thread: {e}")

# Export key functions for easy import
__all__ = [
    'init_db',
    'DBConnection',
    'save_score',
    'save_raw_data',
    'save_price_prediction',
    'save_technical_indicators',
    'save_topic',
    'save_entities',
    'save_content_quality',
    'save_source_reliability',
    'save_backtest_results',
    'get_recent_sentiment',
    'get_stock_mentions',
    'get_topic_trends',
    'get_entity_network',
    'prune_old_data',
    'backup_database',
    'optimize_database',
    'database_maintenance',
    'update_multiple_predictions'
]