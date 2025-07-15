from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
import sqlite3
import pandas as pd
import numpy as np
import json
import logging
import threading
import time
import secrets
import os
from datetime import datetime, timedelta
from database import (
    DBConnection, with_db_lock, 
    get_stock_mentions, get_topic_trends, 
    get_entity_network, optimize_database
)

# Import additional modules for dashboard functionality
try:
    from model import StockAnalysisModel
    from historical_data_import import get_historical_price_data
except ImportError as e:
    logging.error(f"Error importing modules: {e}")
    StockAnalysisModel = None
    get_historical_price_data = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ui")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'trading_dashboard_secret')
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for component availability
model_available = False
topic_modeling_available = False
historical_data_available = False
model_instance = None

def check_component_availability():
    """Check which components are available and set global flags."""
    global model_available, topic_modeling_available, historical_data_available, model_instance
    
    # Check for model availability
    try:
        model_instance = StockAnalysisModel()
        model_available = True
        logger.info("StockAnalysisModel loaded successfully")
    except (ImportError, Exception) as e:
        model_available = False
        model_instance = None
        logger.warning(f"StockAnalysisModel not available: {e}")
    
    # Check for topic modeling
    try:
        from enhanced_topic_modeling import TOPIC_MODELER
        if TOPIC_MODELER and hasattr(TOPIC_MODELER, 'topic_model') and TOPIC_MODELER.topic_model:
            topic_modeling_available = True
            logger.info("Topic modeling module loaded successfully")
        else:
            topic_modeling_available = False
            logger.warning("Topic modeling module loaded but not initialized properly")
    except ImportError:
        topic_modeling_available = False
        logger.warning("Topic modeling module not available")
    
    # Check for historical data
    try:
        if get_historical_price_data:
            historical_data_available = True
            logger.info("Historical data module loaded successfully")
        else:
            historical_data_available = False
            logger.warning("Historical data module loaded but not initialized properly")
    except ImportError:
        historical_data_available = False
        logger.warning("Historical data module not available")

# Run availability check when module is imported
check_component_availability()

@socketio.on('request_update')
def handle_update_request(data):
    """
    Handle various update requests from the dashboard
    """
    update_type = data.get('type')
    ticker = data.get('ticker')
    
    try:
        if update_type == 'market':
            # Fetch overall market data
            market_data = fetch_market_data()
            emit('market_update', {'data': market_data})
        
        elif update_type == 'sentiment':
            # Fetch sentiment scores for stocks
            sentiment_data = fetch_sentiment_scores()
            emit('sentiment_update', {'data': sentiment_data})
        
        elif update_type == 'predictions':
            # Fetch price predictions
            predictions = fetch_price_predictions()
            emit('prediction_update', {'data': predictions})
        
        elif update_type == 'topics':
            # Fetch trending topics
            topics = fetch_trending_topics()
            emit('topics_update', {'data': topics})
        
        elif update_type == 'entity_network':
            # Fetch entity relationships
            entity_network = fetch_entity_relationships()
            emit('entity_update', {'data': entity_network})
        
        elif update_type == 'technical':
            # Fetch technical indicators for a specific ticker
            if ticker:
                technical_data = fetch_technical_indicators(ticker)
                emit('technical_update', {
                    'ticker': ticker, 
                    'data': technical_data
                })
        
        elif update_type == 'signals':
            # Fetch trading signals for a specific ticker
            if ticker:
                signals = fetch_trading_signals(ticker)
                emit('signals_update', {
                    'ticker': ticker, 
                    'data': signals
                })
        
        elif update_type == 'historical':
            # Fetch historical price data for a specific ticker
            if ticker:
                historical_data = fetch_historical_price_data(ticker)
                emit('historical_update', {
                    'ticker': ticker, 
                    'data': historical_data
                })
        
        elif update_type == 'backtest':
            # Run backtest for a specific ticker
            if ticker:
                days = data.get('days', 90)
                backtest_results = run_backtest(ticker, days)
                emit('backtest_update', {
                    'ticker': ticker, 
                    'data': backtest_results
                })
        
        elif update_type == 'feature_importance':
            # Fetch feature importance for a specific ticker
            if ticker:
                importance_data = get_feature_importance(ticker)
                emit('feature_importance_update', {
                    'ticker': ticker, 
                    'data': importance_data
                })
    
    except Exception as e:
        logger.error(f"Error handling update request: {e}")
        emit('error', {'message': str(e)})

def fetch_market_data():
    """
    Aggregate market-wide sentiment and trending data
    """
    try:
        # Fetch total mentions, overall sentiment, and top tickers
        with DBConnection() as conn:
            # Sentiment distribution
            cursor = conn.cursor()
            
            # Get overall sentiment
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            cursor.execute("""
                SELECT AVG(score) as avg_sentiment,
                       COUNT(*) as total_mentions
                FROM scores
                WHERE timestamp > ?
            """, (yesterday,))
            overall_result = cursor.fetchone()
            
            # Get sentiment distribution
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN score > 0.2 THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN score BETWEEN -0.2 AND 0.2 THEN 1 ELSE 0 END) as neutral,
                    SUM(CASE WHEN score < -0.2 THEN 1 ELSE 0 END) as negative
                FROM scores
                WHERE timestamp > ?
            """, (yesterday,))
            distribution_result = cursor.fetchone()
            
            # Get top tickers
            cursor.execute("""
                SELECT stock_ticker, COUNT(*) as mentions
                FROM scores
                WHERE timestamp > ?
                GROUP BY stock_ticker
                ORDER BY mentions DESC
                LIMIT 5
            """, (yesterday,))
            top_tickers_result = cursor.fetchall()
        
        return {
            'overall_sentiment': round(overall_result[0], 2) if overall_result[0] is not None else 0,
            'total_mentions': overall_result[1] if overall_result[1] is not None else 0,
            'sentiment_distribution': {
                'positive': distribution_result[0] if distribution_result[0] is not None else 0,
                'neutral': distribution_result[1] if distribution_result[1] is not None else 0,
                'negative': distribution_result[2] if distribution_result[2] is not None else 0
            },
            'top_tickers': [
                {'ticker': ticker, 'mentions': mentions} 
                for ticker, mentions in top_tickers_result
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return {
            'overall_sentiment': 0,
            'total_mentions': 0,
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'top_tickers': []
        }

def fetch_sentiment_scores():
    """
    Retrieve sentiment scores for all tracked stocks
    """
    try:
        with DBConnection() as conn:
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            query = """
                SELECT stock_ticker, timestamp, score, 
                       GROUP_CONCAT(explanation, '||') as news
                FROM scores
                WHERE timestamp > ?
                GROUP BY stock_ticker
            """
            
            sentiment_df = pd.read_sql_query(query, conn, params=(yesterday,))
            
            # Convert to dictionary format expected by frontend
            sentiment_data = {}
            for _, row in sentiment_df.iterrows():
                news = row['news'].split('||') if row['news'] else []
                sentiment_data[row['stock_ticker']] = {
                    'score': round(row['score'], 2),
                    'mentions': 1,  # This could be improved to count actual mentions
                    'news': news[-3:] if len(news) > 3 else news
                }
            
            return sentiment_data
    except Exception as e:
        logger.error(f"Error fetching sentiment scores: {e}")
        return {}

def fetch_price_predictions():
    """
    Retrieve price predictions for tracked stocks
    """
    try:
        # If model is not available, return empty predictions
        if not model_available or not model_instance:
            return {}
        
        # List of stocks to predict
        tickers = ['NVDA', 'AAPL', 'TSLA', 'META', 'MSFT', 'AMZN', 'GOOGL', 'AMD','QQQ','SPY','DIA']
        predictions = {}
        
        for ticker in tickers:
            try:
                # Use model to get prediction
                predicted_change = model_instance.predict(ticker)
                
                # Determine signal
                signal = 'buy' if predicted_change > 0.01 else 'sell' if predicted_change < -0.01 else 'hold'
                
                # Fetch technical indicators
                with DBConnection() as conn:
                    query = """
                        SELECT indicator_name, indicator_value
                        FROM technical_indicators
                        WHERE stock_ticker = ? AND timestamp > datetime('now', '-1 day')
                        AND indicator_name = 'RSI'
                    """
                    cursor = conn.cursor()
                    cursor.execute(query, (ticker,))
                    rsi_result = cursor.fetchone()
                
                rsi = float(rsi_result[1]) if rsi_result else None
                
                predictions[ticker] = {
                    'change': round(predicted_change * 100, 2),
                    'signal': signal,
                    'rsi': rsi
                }
            except Exception as pred_err:
                logger.error(f"Error predicting for {ticker}: {pred_err}")
        
        return predictions
    except Exception as e:
        logger.error(f"Error fetching price predictions: {e}")
        return {}

def fetch_trending_topics():
    """
    Retrieve trending topics
    """
    try:
        # Use topic trends function from database or topic modeling module
        topics_df = get_topic_trends(days=14, limit=15)
        
        if topics_df.empty:
            return []
        
        # Convert to list of dictionaries
        return [
            {
                'topic': row['topic_description'],
                'mentions': int(row['topic_count']),
                'sentiment': float(row.get('avg_sentiment', 0))
            }
            for _, row in topics_df.iterrows()
        ]
    except Exception as e:
        logger.error(f"Error fetching trending topics: {e}")
        return []

def fetch_entity_relationships():
    """
    Retrieve entity network relationships
    """
    try:
        # Use entity network function from database
        entity_df = get_entity_network(days=14)
        
        if entity_df.empty:
            return {"nodes": [], "links": []}
        
        # Convert to network format
        nodes = set()
        for _, row in entity_df.iterrows():
            nodes.add(row['source'])
            nodes.add(row['target'])
        
        nodes_list = [{"id": node, "group": 1} for node in nodes]
        links = [
            {
                "source": row['source'], 
                "target": row['target'], 
                "value": int(row['weight'])
            } 
            for _, row in entity_df.iterrows()
        ]
        
        return {"nodes": nodes_list, "links": links}
    except Exception as e:
        logger.error(f"Error fetching entity relationships: {e}")
        return {"nodes": [], "links": []}

def fetch_technical_indicators(ticker):
    """
    Retrieve technical indicators for a specific ticker
    """
    try:
        # If no model or historical data, return empty
        if not model_available or not get_historical_price_data:
            return []
        
        # Fetch historical data
        data = get_historical_price_data(ticker)
        
        # Calculate technical indicators
        if model_instance:
            indicators = model_instance.calculate_technical_indicators(data)
        
        # If no indicators, return empty
        if not indicators or indicators.empty:
            return []
        
        # Prepare indicators for transmission
        latest_indicators = indicators.iloc[-1]
        return [
            {
                'indicator_name': col,
                'indicator_value': float(latest_indicators[col])
            }
            for col in ['RSI', 'MACD', 'SMA20', 'Close']
            if col in latest_indicators.index
        ]
    except Exception as e:
        logger.error(f"Error fetching technical indicators for {ticker}: {e}")
        return []

def fetch_trading_signals(ticker):
    """
    Generate trading signals for a specific ticker
    """
    try:
        # If model is not available, return error
        if not model_available or not model_instance:
            return {"error": "Model not available"}
        
        # Generate trading signals
        return model_instance.generate_trading_signals(ticker)
    except Exception as e:
        logger.error(f"Error generating trading signals for {ticker}: {e}")
        return {"error": str(e)}

def fetch_historical_price_data(ticker):
    """
    Fetch historical price data for a specific ticker
    """
    try:
        # If historical data module is not available, return error
        if not historical_data_available or not get_historical_price_data:
            return {"error": "Historical data not available"}
        
        # Fetch data for the last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        data = get_historical_price_data(ticker, start_date, end_date)
        
        if data is None or data.empty:
            return {"error": "No data available"}
        
        return {
            "dates": data.index.strftime("%Y-%m-%d").tolist(),
            "opens": data["Open"].tolist(),
            "highs": data["High"].tolist(),
            "lows": data["Low"].tolist(),
            "closes": data["Close"].tolist(),
            "volumes": data["Volume"].tolist()
        }
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {e}")
        return {"error": str(e)}

def run_backtest(ticker, days=90):
    """
    Run backtest for a specific ticker
    """
    try:
        # If model is not available, return error
        if not model_available or not model_instance:
            return {"error": "Model not available"}
        
        # Run backtest
        return model_instance.backtest_model(ticker, days)
    except Exception as e:
        logger.error(f"Error running backtest for {ticker}: {e}")
        return {"error": str(e)}

def get_feature_importance(ticker):
    """
    Get feature importance data for a ticker
    """
    try:
        # Try to get from model first
        importance_data = None
        if model_available and model_instance and hasattr(model_instance, 'analyze_feature_importance'):
            importance_data = model_instance.analyze_feature_importance(ticker)
            
        # If model analysis didn't work, try reading from file
        if not importance_data:
            model_dir = getattr(model_instance, 'model_dir', 'models') if model_instance else 'models'
            importance_path = os.path.join(model_dir, f"{ticker}_feature_importance.json")
            
            if os.path.exists(importance_path):
                try:
                    with open(importance_path, 'r') as f:
                        importance_data = json.load(f)
                except Exception as file_err:
                    logger.error(f"Error reading importance file: {file_err}")
                    return {"error": "Could not read feature importance file"}
            else:
                return {"error": "Feature importance data not available"}
                
        return importance_data
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        return {"error": str(e)}

# Existing routes from the previous implementation
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/logs')
def view_logs():
    log_file_path = os.path.join(app.root_path, 'trading_system.log')
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except Exception as e:
        log_content = f"Failed to read log file: {e}"
    return render_template('logs.html', logs=log_content)

# New API routes for dashboard data
@app.route('/api/sentiment')
def sentiment_api():
    """API endpoint for sentiment data"""
    try:
        historical = request.args.get("historical", "false").lower() == "true"
        days = int(request.args.get("days", "7"))
        
        with DBConnection() as conn:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            query = """
                SELECT stock_ticker, timestamp, score
                FROM scores
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            
            sentiment_df = pd.read_sql_query(query, conn, params=(start_date.isoformat(), end_date.isoformat()))
            
            result = {}
            if not sentiment_df.empty:
                for ticker, group in sentiment_df.groupby("stock_ticker"):
                    result[ticker] = {
                        "dates": group['timestamp'].dt.strftime("%Y-%m-%d").tolist(),
                        "sentiment": group['score'].tolist()
                    }
            
            return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in sentiment API: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predictions/<ticker>')
def predictions_api(ticker):
    """API endpoint for price predictions"""
    try:
        # Validate ticker
        if not ticker.isalnum():
            return jsonify({"error": "Invalid ticker"}), 400
        
        with DBConnection() as conn:
            query = """
                SELECT timestamp, predicted_change
                FROM price_predictions
                WHERE stock_ticker = ?
                ORDER BY timestamp DESC
                LIMIT 30
            """
            
            predictions_df = pd.read_sql_query(query, conn, params=(ticker,))
            
            result = {
                "ticker": ticker,
                "timestamps": predictions_df['timestamp'].tolist(),
                "predictions": predictions_df['predicted_change'].tolist()
            }
            
            return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in predictions API: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/feature_importance/<ticker>')
def feature_importance_api(ticker):
    """API endpoint for feature importance analysis"""
    try:
        # Ensure valid model instance
        if not model_instance:
            # Try to get from app config if available
            if hasattr(app, 'config') and 'model_instance' in app.config:
                current_model = app.config['model_instance']
            else:
                current_model = None
            
            if not current_model:
                return jsonify({"error": "Model not available"}), 500
        else:
            current_model = model_instance
            
        # Check if model has the feature importance method
        if not hasattr(current_model, 'analyze_feature_importance'):
            return jsonify({"error": "Feature importance analysis not available"}), 500
        
        # Get feature importance data
        importance_data = get_feature_importance(ticker)
        
        if isinstance(importance_data, dict) and "error" in importance_data:
            return jsonify(importance_data), 404
        
        # Return the data
        return jsonify({
            "ticker": ticker,
            "importance": importance_data
        })
    except Exception as e:
        logger.error(f"Error in feature importance API: {e}")
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return render_template('500.html'), 500

# Background thread for periodic tasks
def background_thread():
    """
    Background thread to periodically update dashboard data
    """
    logger.info("Starting background thread for dashboard updates")
    
    while True:
        try:
            # Perform periodic tasks
            market_data = fetch_market_data()
            sentiment_data = fetch_sentiment_scores()
            predictions = fetch_price_predictions()
            topics = fetch_trending_topics()
            entity_network = fetch_entity_relationships()
            
            # Emit updates to all connected clients
            socketio.emit('market_update', {'data': market_data})
            socketio.emit('sentiment_update', {'data': sentiment_data})
            socketio.emit('prediction_update', {'data': predictions})
            socketio.emit('topics_update', {'data': topics})
            socketio.emit('entity_update', {'data': entity_network})
            
            # Sleep for a while before next update
            socketio.sleep(60)  # Update every minute
        
        except Exception as e:
            logger.error(f"Error in background thread: {e}")
            socketio.sleep(30)  # Wait a bit longer on error

@socketio.on('connect')
def handle_connect():
    """
    Handle client connection and start background thread if not already running
    """
    global thread
    with thread_lock:
        if thread is None or not thread.is_alive():
            thread = socketio.start_background_task(target=background_thread)
    
    # Send initial component availability status
    emit('connection_response', {
        'data': 'Connected and background task started.',
        'components': {
            'model_available': model_available,
            'topic_modeling_available': topic_modeling_available,
            'historical_data_available': historical_data_available
        }
    })

# Initialize the thread
thread = None
thread_lock = threading.Lock()

if __name__ == '__main__':
    # Start with component availability check
    check_component_availability()
    
    # Run the application
    socketio.run(app, debug=True)