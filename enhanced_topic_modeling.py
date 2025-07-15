import logging
import re
import numpy as np
import pandas as pd
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import os  # Ensure os is imported if used later

# Import NLP and Topic Modeling Libraries
try:
    import bertopic
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    import umap
    import hdbscan
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel
    from gensim.utils import simple_preprocess
    from gensim.parsing.preprocessing import STOPWORDS
    libraries_available = True
except ImportError:
    logging.warning("Optional topic modeling libraries not fully available")
    libraries_available = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_topic_modeling")

# Financial domain-specific stop words
FINANCIAL_STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    # Financial specific stop words
    'stock', 'market', 'trading', 'investment', 'price', 'share', 'company', 
    'data', 'report', 'analysis', 'financial', 'investor', 'business'
}

class AdvancedTopicModeler:
    def __init__(self):
        """
        Initialize advanced topic modeling capabilities.
        """
        try:
            if not libraries_available:
                logger.warning("Required libraries not available for topic modeling")
                self.topic_model = None
                self.embedding_model = None
                return
                
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize dimensionality reduction
            self.umap_model = umap.UMAP(
                n_neighbors=15, 
                n_components=5, 
                min_dist=0.0, 
                metric='cosine'
            )
            
            # Initialize clustering
            self.cluster_model = hdbscan.HDBSCAN(
                min_cluster_size=10, 
                metric='euclidean', 
                cluster_selection_method='eom'
            )
            
            # Initialize BERTopic model
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=self.umap_model,
                hdbscan_model=self.cluster_model,
                nr_topics='auto',
                low_memory=True,
                calculate_probabilities=True
            )
            
            # Initialize LDA model variables
            self.dictionary = None
            self.lda_model = None
            
            logger.info("Advanced Topic Modeler initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Topic Modeler: {e}")
            self.topic_model = None
            self.embedding_model = None

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for topic modeling.
        
        Args:
            text (str): Input text
        
        Returns:
            str: Preprocessed text
        """
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+', '', text)
            
            # Remove special characters
            text = re.sub(r'[^a-z0-9\s]', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text and remove stopwords.
        
        Args:
            text (str): Preprocessed text
            
        Returns:
            List[str]: List of tokens
        """
        try:
            # Tokenize and remove stopwords
            stop_words = STOPWORDS.union(FINANCIAL_STOP_WORDS)
            tokens = [
                word for word in simple_preprocess(text, deacc=True)
                if word not in stop_words and len(word) > 2
            ]
            return tokens
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return []
    
    def extract_topics_bert(self, documents: List[str]) -> Tuple[List[int], List[str], List[float]]:
        """
        Extract topics using BERTopic model.
        
        Args:
            documents (List[str]): List of preprocessed text documents
            
        Returns:
            Tuple[List[int], List[str], List[float]]: Tuple of topic IDs, topic descriptions, and probabilities
        """
        try:
            if not self.topic_model or not documents:
                return [], [], []
                
            # Fit the model on the documents
            topics, probs = self.topic_model.fit_transform(documents)
            
            # Get topic info
            topic_info = self.topic_model.get_topic_info()
            
            # Extract topic descriptions
            topic_descriptions = []
            for topic in topics:
                if topic == -1:  # -1 means outlier
                    topic_descriptions.append("Miscellaneous")
                else:
                    # Get the top words for this topic
                    top_words = " | ".join([word for word, _ in self.topic_model.get_topic(topic)][:5])
                    topic_descriptions.append(top_words)
            
            # Get max probability for each document
            max_probs = [max(prob) if len(prob) > 0 else 0.0 for prob in probs]
            
            return topics, topic_descriptions, max_probs
        except Exception as e:
            logger.error(f"Error extracting BERTopic topics: {e}")
            return [], [], []
    
    def extract_topics_lda(self, documents: List[str]) -> Tuple[List[int], List[str], List[float]]:
        """
        Extract topics using LDA model (fallback method).
        
        Args:
            documents (List[str]): List of preprocessed text documents
            
        Returns:
            Tuple[List[int], List[str], List[float]]: Tuple of topic IDs, topic descriptions, and probabilities
        """
        try:
            # Tokenize documents
            tokenized_docs = [self.tokenize_text(doc) for doc in documents]
            
            if not tokenized_docs or all(len(doc) == 0 for doc in tokenized_docs):
                return [], [], []
            
            # Create dictionary
            self.dictionary = corpora.Dictionary(tokenized_docs)
            
            # Create corpus
            corpus = [self.dictionary.doc2bow(doc) for doc in tokenized_docs]
            
            # Determine number of topics (between 2 and 10 based on corpus size)
            num_topics = min(max(2, len(documents) // 10), 10)
            
            # Train LDA model
            self.lda_model = LdaModel(
                corpus=corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                passes=10,
                alpha='auto',
                eta='auto'
            )
            
            # Get topics for each document
            topics = []
            topic_descriptions = []
            probabilities = []
            
            for doc_bow in corpus:
                # Get topic distribution for this document
                topic_dist = self.lda_model.get_document_topics(doc_bow)
                
                # Find topic with highest probability
                if topic_dist:
                    max_topic = max(topic_dist, key=lambda x: x[1])
                    topics.append(max_topic[0])
                    probabilities.append(max_topic[1])
                    
                    # Get topic description
                    topic_words = self.lda_model.show_topic(max_topic[0], topn=5)
                    description = " | ".join([word for word, _ in topic_words])
                    topic_descriptions.append(description)
                else:
                    topics.append(-1)
                    probabilities.append(0.0)
                    topic_descriptions.append("Miscellaneous")
            
            return topics, topic_descriptions, probabilities
        except Exception as e:
            logger.error(f"Error extracting LDA topics: {e}")
            return [], [], []
    
    def analyze_document_topics(self, document_id: int, text: str) -> bool:
        """
        Analyze topics in a document and save to database.
        
        Args:
            document_id (int): ID of the document
            text (str): Document text
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if we have necessary models
            if not self.topic_model and not libraries_available:
                logger.warning("Topic modeling not available - skipping analysis")
                return False
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Store document for batch processing later
            document_cache_path = os.path.join("data_cache", "pending_documents.json")
            pending_docs = {}
            
            # Load existing pending documents if available
            try:
                if os.path.exists(document_cache_path):
                    with open(document_cache_path, 'r') as f:
                        pending_docs = json.load(f)
            except Exception as e:
                logger.error(f"Error loading pending documents: {e}")
            
            # Add this document to pending analysis
            pending_docs[str(document_id)] = {
                "text": processed_text,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Save updated pending documents
            try:
                os.makedirs(os.path.dirname(document_cache_path), exist_ok=True)
                with open(document_cache_path, 'w') as f:
                    json.dump(pending_docs, f)
            except Exception as e:
                logger.error(f"Error saving pending documents: {e}")
            
            # For immediate feedback, use a simpler keyword-based approach
            # to extract potential topics
            keywords = self._extract_simple_keywords(processed_text)
            
            # Connect to database
            with sqlite3.connect("trading.db") as conn:
                c = conn.cursor()
                timestamp = datetime.utcnow().isoformat()
                
                # Save temporary topic information with keywords
                if keywords:
                    c.execute('''INSERT INTO topics 
                               (timestamp, document_id, topic_id, topic_description, probability) 
                               VALUES (?, ?, ?, ?, ?)''',
                              (timestamp, document_id, 0, " | ".join(keywords[:5]), 0.5))
                    conn.commit()
            
            logger.debug(f"Saved temporary keywords for document {document_id}, queued for batch topic analysis")
            return True
        
        except Exception as e:
            logger.error(f"Error analyzing document topics: {e}")
            return False

    def _extract_simple_keywords(self, text: str, top_n: int = 10) -> list:
        """
        Extract simple keywords from text as a fallback method.
        
        Args:
            text (str): Input text
            top_n (int): Number of top keywords to return
            
        Returns:
            list: List of extracted keywords
        """
        try:
            # Tokenize
            tokens = self.tokenize_text(text)
            
            # Count word frequencies
            word_counts = {}
            for token in tokens:
                if len(token) > 3:  # Only include words longer than 3 chars
                    word_counts[token] = word_counts.get(token, 0) + 1
            
            # Sort by frequency
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Return top N keywords
            return [word for word, count in sorted_words[:top_n]]
        
        except Exception as e:
            logger.error(f"Error extracting simple keywords: {e}")
            return []

    def process_pending_documents(self):
        """
        Process all pending documents for topic modeling in batch.
        This should be called periodically (e.g., every hour) by a scheduler.
        """
        try:
            document_cache_path = os.path.join("data_cache", "pending_documents.json")
            
            # Check if we have pending documents
            if not os.path.exists(document_cache_path):
                logger.info("No pending documents for topic analysis")
                return False
            
            # Load pending documents
            with open(document_cache_path, 'r') as f:
                pending_docs = json.load(f)
            
            if not pending_docs:
                logger.info("No pending documents for topic analysis")
                return False
                
            # Prepare batch processing
            doc_ids = []
            texts = []
            
            for doc_id, doc_info in pending_docs.items():
                doc_ids.append(int(doc_id))
                texts.append(doc_info["text"])
            
            # Only process if we have enough documents
            if len(texts) < 2:
                logger.info(f"Not enough documents for topic modeling ({len(texts)} available, need at least 2)")
                return False
            
            # Now we have multiple documents, we can run BERTopic
            if self.topic_model:
                try:
                    topics, probs = self.topic_model.fit_transform(texts)
                    
                    # Extract topic descriptions
                    topic_descriptions = []
                    for topic in topics:
                        if topic == -1:  # -1 means outlier
                            topic_descriptions.append("Miscellaneous")
                        else:
                            top_words = " | ".join([word for word, _ in self.topic_model.get_topic(topic)][:5])
                            topic_descriptions.append(top_words)
                    
                    max_probs = [max(prob) if len(prob) > 0 else 0.0 for prob in probs]
                    
                    # Save to database
                    with sqlite3.connect("trading.db") as conn:
                        cursor = conn.cursor()
                        timestamp = datetime.utcnow().isoformat()
                        
                        for i, doc_id in enumerate(doc_ids):
                            # Update existing topic entries with the real topics
                            cursor.execute('''UPDATE topics
                                             SET topic_id = ?, topic_description = ?, probability = ?
                                             WHERE document_id = ?''',
                                         (topics[i], topic_descriptions[i], max_probs[i], doc_id))
                        
                        conn.commit()
                    
                    # Clear processed documents
                    os.remove(document_cache_path)
                    
                    logger.info(f"Successfully processed {len(doc_ids)} documents for topic analysis")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error in batch topic processing: {e}")
                    return False
            else:
                logger.warning("Topic model not available for batch processing")
                return False
                
        except Exception as e:
            logger.error(f"Error processing pending documents: {e}")
            return False
    
    def get_document_text_from_db(self, days: int, min_documents: int) -> Dict[int, str]:
        """
        Get document text from database for topic modeling.
        
        Args:
            days (int): Number of days to look back
            min_documents (int): Minimum number of documents to retrieve
            
        Returns:
            Dict[int, str]: Dictionary of document IDs and text
        """
        try:
            with sqlite3.connect("trading.db") as conn:
                c = conn.cursor()
                start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
                
                # Query to get documents not yet analyzed for topics
                query = """
                    SELECT r.id, r.text
                    FROM raw_data r
                    LEFT JOIN topics t ON r.id = t.document_id
                    WHERE r.timestamp > ?
                    AND t.document_id IS NULL
                    LIMIT ?
                """
                c.execute(query, (start_date, min_documents))
                documents = {row[0]: row[1] for row in c.fetchall()}
                
                if len(documents) < min_documents:
                    # If not enough unanalyzed documents, get most recent ones
                    additional_query = """
                        SELECT id, text
                        FROM raw_data
                        WHERE timestamp > ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """
                    additional_limit = min_documents - len(documents)
                    c.execute(additional_query, (start_date, additional_limit))
                    additional_docs = {row[0]: row[1] for row in c.fetchall() if row[0] not in documents}
                    documents.update(additional_docs)
                
                logger.info(f"Retrieved {len(documents)} documents for topic modeling")
                return documents
        except Exception as e:
            logger.error(f"Error getting documents for topic modeling: {e}")
            return {}
    
    def run_periodic_topic_modeling(self, days: int = 7, min_documents: int = 25) -> bool:
        """
        Run topic modeling on recent documents periodically with improved robustness.
        
        Args:
            days (int): Number of days to look back
            min_documents (int): Minimum number of documents to consider
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get documents from database
            documents = self.get_document_text_from_db(days, min_documents)
            
            if not documents:
                logger.warning("No documents found for topic modeling")
                return False
            
            # Preprocess documents
            doc_ids = list(documents.keys())
            doc_texts = [self.preprocess_text(text) for text in documents.values()]
            
            # Filter out empty texts
            valid_docs = [(id, text) for id, text in zip(doc_ids, doc_texts) if text.strip()]
            
            if not valid_docs:
                logger.warning("No valid documents after preprocessing")
                return False
            
            doc_ids, doc_texts = zip(*valid_docs)
            
            # Adjust minimum documents for topic modeling
            if len(doc_texts) < 10:
                logger.warning(f"Not enough documents for comprehensive topic modeling. Found {len(doc_texts)}")
                # Fallback to basic topic extraction or skip
                return self._fallback_topic_extraction(doc_ids, doc_texts)
            
            # Ensure at least 2 documents for topic modeling
            if len(doc_texts) < 2:
                logger.warning("Need at least 2 documents for topic modeling")
                return self._fallback_topic_extraction(doc_ids, doc_texts)
            
            # Extract topics
            try:
                if self.topic_model:
                    # Retrain model with enough documents
                    if len(doc_texts) >= 2:
                        # Recreate the model with current documents
                        try:
                            # Reset and retrain the model
                            self.topic_model = BERTopic(
                                embedding_model=self.embedding_model,
                                umap_model=self.umap_model,
                                hdbscan_model=self.cluster_model,
                                nr_topics='auto',
                                low_memory=True,
                                calculate_probabilities=True
                            )
                            
                            # Fit and transform
                            topics, probs = self.topic_model.fit_transform(doc_texts)
                            
                            # Get topic info
                            topic_info = self.topic_model.get_topic_info()
                            
                            # Extract topic descriptions
                            topic_descriptions = []
                            for topic in topics:
                                if topic == -1:  # -1 means outlier
                                    topic_descriptions.append("Miscellaneous")
                                else:
                                    # Get the top words for this topic
                                    top_words = " | ".join([word for word, _ in self.topic_model.get_topic(topic)][:5])
                                    topic_descriptions.append(top_words)
                            
                            # Get max probability for each document
                            max_probs = [max(prob) if len(prob) > 0 else 0.0 for prob in probs]
                            
                            # Save topics to database
                            with sqlite3.connect("trading.db") as conn:
                                cursor = conn.cursor()
                                timestamp = datetime.utcnow().isoformat()
                                
                                for i, doc_id in enumerate(doc_ids):
                                    cursor.execute('''INSERT INTO topics 
                                                     (timestamp, document_id, topic_id, topic_description, probability) 
                                                     VALUES (?, ?, ?, ?, ?)''',
                                                  (timestamp, doc_id, topics[i], topic_descriptions[i], max_probs[i]))
                                
                                conn.commit()
                            
                            logger.info(f"Successfully extracted topics for {len(doc_ids)} documents")
                            return True
                        
                        except Exception as model_err:
                            logger.error(f"Topic model training error: {model_err}")
                            return self._fallback_topic_extraction(doc_ids, doc_texts)
                    else:
                        logger.warning("Not enough documents to train topic model")
                        return self._fallback_topic_extraction(doc_ids, doc_texts)
                else:
                    # Fallback to LDA if BERTopic is not available
                    return self.extract_topics_lda(list(doc_texts))
            
            except Exception as extraction_error:
                logger.error(f"Topic extraction failed: {extraction_error}")
                return self._fallback_topic_extraction(doc_ids, doc_texts)
        
        except Exception as e:
            logger.error(f"Error in periodic topic modeling: {e}")
            return False
    
    def _fallback_topic_extraction(self, doc_ids, doc_texts):
        """
        Fallback method for topic extraction when primary methods fail.
        
        Args:
            doc_ids (list): List of document IDs
            doc_texts (list): List of document texts
        
        Returns:
            bool: True if fallback extraction is successful, False otherwise
        """
        try:
            # Simple keyword-based topic extraction
            from collections import Counter
            import re
            
            # Tokenize and count keywords
            def extract_keywords(text):
                # Remove stop words and punctuation
                words = re.findall(r'\b\w+\b', text.lower())
                return [word for word in words if len(word) > 2 and word not in FINANCIAL_STOP_WORDS]
            
            all_keywords = []
            for text in doc_texts:
                all_keywords.extend(extract_keywords(text))
            
            # Get top keywords as topics
            keyword_counts = Counter(all_keywords)
            top_keywords = keyword_counts.most_common(5)
            
            # Save topics to database
            with sqlite3.connect("trading.db") as conn:
                c = conn.cursor()
                timestamp = datetime.utcnow().isoformat()
                
                for i, (keyword, count) in enumerate(top_keywords):
                    for doc_id in doc_ids[:min(5, len(doc_ids))]:
                        try:
                            c.execute('''INSERT INTO topics 
                                         (timestamp, document_id, topic_id, topic_description, probability) 
                                         VALUES (?, ?, ?, ?, ?)''',
                                      (timestamp, doc_id, i, keyword, count / sum(keyword_counts.values())))
                        except Exception as insert_error:
                            logger.error(f"Error inserting fallback topic: {insert_error}")
                
                conn.commit()
            
            logger.warning(f"Used fallback topic extraction with {len(top_keywords)} topics")
            return True
        except Exception as e:
            logger.error(f"Fallback topic extraction failed: {e}")
            return False

# Create a global topic modeler instance
TOPIC_MODELER = AdvancedTopicModeler()

def analyze_document_topics(document_id: int, text: str) -> bool:
    """
    Public function for document topic analysis.
    
    Args:
        document_id (int): ID of the document
        text (str): Document text
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        return TOPIC_MODELER.analyze_document_topics(document_id, text)
    except Exception as e:
        logger.error(f"Error in public topic analysis: {e}")
        return False

def get_topic_trends(days: int = 30, limit: int = 10) -> pd.DataFrame:
    """
    Public function to get trending topics.
    
    Args:
        days (int): Number of days for trend analysis
        limit (int): Maximum number of topics to return
        
    Returns:
        pd.DataFrame: DataFrame with trending topics
    """
    try:
        trends_df = TOPIC_MODELER.analyze_topic_trends(days)
        if trends_df.empty:
            return pd.DataFrame()
        return trends_df.head(limit)
    except Exception as e:
        logger.error(f"Error getting topic trends: {e}")
        return pd.DataFrame()