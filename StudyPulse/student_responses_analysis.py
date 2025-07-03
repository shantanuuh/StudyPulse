import streamlit as st
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import logging
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from functools import lru_cache, wraps
from transformers import pipeline

# ✅ Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ✅ NLTK fixes for Streamlit Cloud
nltk.data.path.append("/home/appuser/nltk_data")  # For Streamlit Cloud compatibility
nltk.download('punkt')
nltk.download('stopwords')


# ✅ Custom LRU cache wrapper for list support
def lru_cache_with_list_support(maxsize=None):
    def decorator(func):
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            hashable_args = tuple(tuple(arg) if isinstance(arg, list) else arg for arg in args)
            hashable_kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()}
            key = (hashable_args, tuple(sorted(hashable_kwargs.items())))

            if key not in cache:
                result = func(*args, **kwargs)
                cache[key] = result
                if maxsize and len(cache) > maxsize:
                    cache.pop(next(iter(cache)))
            return cache[key]

        return wrapper
    return decorator


@lru_cache_with_list_support(maxsize=1)
def analyze_student_responses(text_data, column_name, sentiment_analyzer=None):
    """Analyze a column of text responses with sentiment, keyword, and pattern analysis"""
    try:
        if isinstance(column_name, list):
            column_name = str(column_name[0])

        if isinstance(text_data, list):
            text_data = [str(item) for sublist in text_data for item in (sublist if isinstance(sublist, list) else [sublist])]

        responses = [r for r in text_data if isinstance(r, str) and len(r.strip()) > 0]
        if not responses:
            return {"error": "No valid responses found in this column."}

        result = {}

        # ✅ Sentiment Analysis
        if sentiment_analyzer:
            try:
                sentiment_data = sentiment_analyzer(responses[:100])
                sentiment_counts = Counter([item['label'] for item in sentiment_data])
                sentiment_percentage = {k: sentiment_counts[k] / len(sentiment_data) * 100 for k in sentiment_counts}
                average_score = sum(item['score'] for item in sentiment_data) / len(sentiment_data)
                result["sentiment_analysis"] = {
                    "counts": sentiment_percentage,
                    "average_score": average_score
                }
            except Exception as e:
                result["sentiment_analysis"] = {
                    "error": f"Sentiment analysis failed: {str(e)}"
                }

        # ✅ Keyword Extraction
        try:
            all_text = " ".join(responses)
            all_text = re.sub(r'[^\w\s]', '', all_text.lower())
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(all_text)
            filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
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

        # ✅ Pattern Clustering
        if len(responses) >= 10:
            try:
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                X = vectorizer.fit_transform(responses)
                num_clusters = min(5, max(2, len(responses) // 10))
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(X)

                common_patterns = {}
                for cluster_num in range(num_clusters):
                    cluster_indices = [i for i, label in enumerate(kmeans.labels_) if label == cluster_num]
                    cluster_size = len(cluster_indices)
                    cluster_percentage = (cluster_size / len(responses)) * 100

                    if cluster_indices:
                        centroid = kmeans.cluster_centers_[cluster_num]
                        distances = [(idx, np.linalg.norm(X[idx].toarray() - centroid)) for idx in cluster_indices]
                        sorted_indices = sorted(distances, key=lambda x: x[1])
                        representative_idx = sorted_indices[0][0]
                        representative_response = responses[representative_idx]

                        cluster_responses = [responses[i] for i in cluster_indices]
                        cluster_text = " ".join(cluster_responses)
                        cluster_text = re.sub(r'[^\w\s]', '', cluster_text.lower())
                        cluster_tokens = word_tokenize(cluster_text)
                        cluster_words = [w for w in cluster_tokens if w not in stop_words and len(w) > 2]
                        cluster_freq = Counter(cluster_words)
                        cluster_keywords = [word for word, _ in cluster_freq.most_common(7)]

                        common_patterns[f"Pattern {cluster_num+1}"] = {
                            "percentage": cluster_percentage,
                            "keywords": cluster_keywords,
                            "representative_response": representative_response
                        }

                result["pattern_analysis"] = {
                    "common_patterns": common_patterns
                }

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
