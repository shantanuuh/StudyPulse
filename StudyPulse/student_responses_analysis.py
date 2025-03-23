import streamlit as st
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from functools import lru_cache
from transformers import pipeline
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# In student_responses_analysis.py
from functools import wraps

def lru_cache_with_list_support(maxsize=None):
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert lists in args to tuples
            hashable_args = tuple(tuple(arg) if isinstance(arg, list) else arg for arg in args)
            
            # Convert lists in kwargs to tuples
            hashable_kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()}
            
            # Create a hashable key from the arguments
            key = (hashable_args, tuple(sorted(hashable_kwargs.items())))
            
            if key not in cache:
                result = func(*args, **kwargs)
                cache[key] = result
                # Manage cache size
                if maxsize and len(cache) > maxsize:
                    cache.pop(next(iter(cache)))
            return cache[key]
        
        return wrapper
    
    return decorator

# Replace @lru_cache with this custom decorator
@lru_cache_with_list_support(maxsize=1)
def analyze_student_responses(text_data, column_name, sentiment_analyzer=None):
    """Analyze a column of text responses with caching for performance"""
    try:
        if isinstance(column_name, list):
            column_name = str(column_name[0])  # Convert list to string

        if isinstance(text_data, list):
            text_data = [str(item) for sublist in text_data for item in (sublist if isinstance(sublist, list) else [sublist])]
        
        # Extract responses from the column
        responses = [r for r in text_data if isinstance(r, str) and len(r.strip()) > 0]

        if not responses:
            return {
                "error": "No valid responses found in this column."
            }

        result = {}
        # Sentiment analysis if model is available
        if sentiment_analyzer:
            try:
                # Batch process sentiments
                sentiment_data = sentiment_analyzer(responses[:100])  # Limit to first 100 for performance
                sentiment_counts = Counter([item['label'] for item in sentiment_data])
                sentiment_percentage = {key: sentiment_counts[key] / len(sentiment_data) * 100 for key in sentiment_counts}
                
                if sentiment_data:
                    average_score = sum([item['score'] for item in sentiment_data]) / len(sentiment_data)
                else:
                    average_score = 0
                
                result["sentiment_analysis"] = {
                    "counts": sentiment_percentage,
                    "average_score": average_score
                }
            except Exception as e:
                result["sentiment_analysis"] = {
                    "error": f"Sentiment analysis failed: {str(e)}"
                }
        
        # Extract keywords and common phrases
        try:
            # Combine all responses
            all_text = " ".join(responses)
            
            # Basic cleaning
            all_text = re.sub(r'[^\w\s]', '', all_text.lower())
            
            # Tokenize and remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(all_text)
            filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
            
            # Get word frequency
            word_freq = Counter(filtered_tokens)
            top_words = word_freq.most_common(15)
            
            result["keyword_analysis"] = {
                "top_words": top_words,
                "response_count": len(responses),
                "average_length": sum(len(r.split()) for r in responses) / len(responses)
            }
        except Exception as e:
            result["keyword_analysis"] = {
                "error": f"Keyword extraction failed: {str(e)}"
            }
        
        # Perform clustering to identify patterns
        if len(responses) >= 10:  # Need enough data for meaningful clustering
            try:
                # Vectorize text
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                X = vectorizer.fit_transform(responses)
                
                # Determine optimal number of clusters (2-5)
                num_clusters = min(5, max(2, len(responses) // 10))
                
                # Apply KMeans
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(X)
                
                # Get common patterns from each cluster
                common_patterns = {}
                for cluster_num in range(num_clusters):
                    cluster_indices = [i for i, label in enumerate(kmeans.labels_) if label == cluster_num]
                    cluster_size = len(cluster_indices)
                    cluster_percentage = (cluster_size / len(responses)) * 100
                    
                    if cluster_indices:
                        # Get representative responses (closest to centroid)
                        centroid = kmeans.cluster_centers_[cluster_num]
                        distances = []
                        for idx in cluster_indices:
                            dist = np.linalg.norm(X[idx].toarray() - centroid)
                            distances.append((idx, dist))
                        
                        # Sort by distance and get closest example
                        sorted_indices = sorted(distances, key=lambda x: x[1])
                        representative_idx = sorted_indices[0][0] if sorted_indices else cluster_indices[0]
                        representative_response = responses[representative_idx]
                        
                        # Extract keywords specific to this cluster
                        cluster_responses = [responses[i] for i in cluster_indices]
                        cluster_text = " ".join(cluster_responses)
                        cluster_text = re.sub(r'[^\w\s]', '', cluster_text.lower())
                        cluster_tokens = word_tokenize(cluster_text)
                        cluster_words = [w for w in cluster_tokens if w not in stop_words and len(w) > 2]
                        cluster_freq = Counter(cluster_words)
                        cluster_keywords = [word for word, count in cluster_freq.most_common(7)]
                        
                        common_patterns[f"Pattern {cluster_num+1}"] = {
                            "percentage": cluster_percentage,
                            "keywords": cluster_keywords,
                            "representative_response": representative_response
                        }
                
                result["pattern_analysis"] = {
                    "common_patterns": common_patterns
                }
                
                # Generate insights based on patterns
                insights = []
                for pattern, data in common_patterns.items():
                    keywords = ", ".join(data["keywords"][:3])
                    insights.append(f"{pattern} ({data['percentage']:.1f}%) focuses on {keywords}")
                
                result["insights"] = insights
                
            except Exception as e:
                result["pattern_analysis"] = {
                    "error": f"Pattern analysis failed: {str(e)}"
                }
        
        return result
    
    except Exception as e:
        logger.error(f"Error in analyze_student_responses: {e}")
        return {"error": str(e)}
