"""Sentiment Analysis and Topic Modeling on Amazon Reviews using Traditional NLP."""

import html
import re
import time
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datasets import load_dataset
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import NMF
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake


# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK resources."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)


download_nltk_data()


# Idiomatic expressions and context-dependent phrases
# These are phrases where individual words might have different sentiment than the phrase as a whole
POSITIVE_IDIOMS = {
    # "Crazy" idioms (positive when used with "for", "about", "love")
    r'\b(went|go|goes|going|gone)\s+(crazy|wild|nuts|bonkers)\s+(for|about|over)\b': 'absolutely loved',
    r'\b(crazy|wild|nuts|bonkers)\s+(about|for|over)\b': 'enthusiastic about',
    r'\b(love|loved|loving|loves)\s+it\s+to\s+death\b': 'love it extremely',

    # "Die" idioms (positive)
    r'\b(die|dying|died)\s+(for|to\s+have)\b': 'really want',
    r'\bto\s+die\s+for\b': 'absolutely wonderful',

    # "Kill" idioms (positive in context)
    r'\b(killing|killed)\s+it\b': 'did excellently',

    # "Sick" idioms (can be positive in slang)
    r'\b(so|really|pretty)\s+sick\b': 'really cool',

    # "Insane" idioms (positive when describing quality/value)
    r'\b(insane|insanely)\s+(good|great|amazing|awesome|value|quality|deal|price)\b': 'extremely good',

    # "Blow away" idioms
    r'\b(blow|blew|blown)\s+(me|us|them)\s+away\b': 'extremely impressed',
    r'\b(mind|brain)(-|\s+)(blowing|blown)\b': 'amazingly impressive',

    # "Destroy" in gaming/performance context
    r'\b(destroy|destroys|destroyed|crushing|crushed|kills|killed)\s+(it|the\s+competition)\b': 'performs excellently',
}

NEGATIVE_IDIOMS = {
    # "Crazy" in negative contexts
    r'\b(driving|drives|drove)\s+(me|us)\s+(crazy|insane|nuts|mad)\b': 'very frustrating',
    r'\b(go|goes|went)\s+crazy\s+(and\s+)?(broke|stopped|died)\b': 'malfunctioned badly',

    # "Die" in negative contexts
    r'\b(died|dies|dying)\s+(on|after|within)\b': 'stopped working',
    r'\b(dead|died)\s+(on\s+arrival|immediately)\b': 'completely broken',

    # "Sick" in literal health context
    r'\b(made|makes|making)\s+(me|us|them)\s+sick\b': 'caused illness',
    r'\b(feel|felt|feeling)\s+sick\b': 'felt ill',
}


def preprocess_idioms(text: str) -> str:
    """
    Preprocess text to handle idiomatic expressions by replacing them with
    sentiment-equivalent phrases. This helps VADER and other models correctly
    identify sentiment in context-dependent phrases.
    """
    import re

    processed = text

    # Apply positive idiom replacements first (more specific patterns)
    for pattern, replacement in POSITIVE_IDIOMS.items():
        processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)

    # Then apply negative idiom replacements
    for pattern, replacement in NEGATIVE_IDIOMS.items():
        processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)

    return processed


@st.cache_data
def load_amazon_reviews(num_samples: int = 50, seed: Optional[int] = None) -> List[Dict]:
    """
    Load real Amazon reviews from Hugging Face dataset.
    Uses SetFit/amazon_reviews_multi_en which contains English reviews.
    """
    # Load the dataset (validation split for diverse reviews)
    dataset = load_dataset('SetFit/amazon_reviews_multi_en', split='validation')

    # Convert to pandas for easier manipulation
    df = pd.DataFrame(dataset)

    random_state = seed if seed is not None else None
    final_df = df.sample(n=min(num_samples, len(df)), random_state=random_state)

    # Convert to our format
    reviews = []
    for idx, row in final_df.iterrows():
        # Extract fields from the dataset
        review_text = row.get('text', row.get('review_body', ''))
        rating = int(row.get('label', 0)) + 1  # Convert 0-4 to 1-5

        # Get product title if available
        product = row.get('review_title', row.get('product_title', f'Product #{idx + 1}'))
        if not product or len(product.strip()) == 0:
            product = f'Product #{idx + 1}'

        # Only include reviews with actual text
        if review_text and len(review_text.strip()) > 10:
            reviews.append({
                'id': idx + 1,
                'product': product[:100],  # Limit product name length
                'rating': rating,
                'review': review_text[:500],  # Limit review length for display
            })

    # Return up to num_samples reviews
    return reviews[:num_samples]


@st.cache_resource
def get_vader_analyzer():
    """Initialize and cache VADER sentiment analyzer."""
    return SentimentIntensityAnalyzer()


def build_sentiment_labels_from_df(reviews: List[Dict]) -> Tuple[List[str], List[int]]:
    """
    Build sentiment labels from reviews based on ratings.
    Rating >= 4 → positive (1), rating <= 2 → negative (0), drop neutrals.
    """
    texts = []
    labels = []

    for review in reviews:
        rating = review.get('rating', 3)
        # Only include clearly positive or negative reviews
        if rating >= 4:
            texts.append(review['review'])
            labels.append(1)  # positive
        elif rating <= 2:
            texts.append(review['review'])
            labels.append(0)  # negative

    return texts, labels


@st.cache_resource
def train_ml_sentiment_model(texts: Tuple[str, ...], labels: Tuple[int, ...], use_idiom_preprocessing: bool = True) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """
    Train a TF-IDF + Logistic Regression sentiment model in runtime.
    Uses caching to train only once per session.

    Args:
        texts: Training texts
        labels: Training labels
        use_idiom_preprocessing: If True, preprocess idioms in training data
    """
    # Convert tuples back to lists for processing
    texts_list = list(texts)
    labels_list = list(labels)

    # Validate we have both classes
    from collections import Counter
    label_counts = Counter(labels_list)
    print(f"Training samples - Positive (1): {label_counts.get(1, 0)}, Negative (0): {label_counts.get(0, 0)}")

    if len(label_counts) < 2:
        raise ValueError(f"Need both positive and negative samples to train. Got only: {label_counts}")

    if label_counts.get(0, 0) < 2 or label_counts.get(1, 0) < 2:
        raise ValueError(f"Need at least 2 samples of each class. Got: {label_counts}")

    # Preprocess idioms in training data if requested
    if use_idiom_preprocessing:
        texts_list = [preprocess_idioms(text) for text in texts_list]

    # Optional: sample to keep training snappy
    max_samples = 2000
    if len(texts_list) > max_samples:
        texts_list = texts_list[:max_samples]
        labels_list = labels_list[:max_samples]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=20000,
        min_df=1
    )
    X = vectorizer.fit_transform(texts_list)

    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        C=1.0  # Regularization parameter
    )
    clf.fit(X, labels_list)

    # Verify the model can predict both classes
    train_preds = clf.predict(X)
    train_pred_counts = Counter(train_preds)
    print(f"Training predictions - Positive (1): {train_pred_counts.get(1, 0)}, Negative (0): {train_pred_counts.get(0, 0)}")

    return vectorizer, clf


def analyze_sentiment_ml(review_text: str, vectorizer: TfidfVectorizer, clf: LogisticRegression, use_idiom_preprocessing: bool = True) -> Dict[str, Any]:
    """
    Analyze sentiment using the trained ML model (TF-IDF + Logistic Regression).

    Args:
        review_text: The text to analyze
        vectorizer: Trained TF-IDF vectorizer
        clf: Trained classifier
        use_idiom_preprocessing: If True, preprocess idioms before analysis
    """
    start_time = time.time()

    # Preprocess idioms if requested
    processed_text = preprocess_idioms(review_text) if use_idiom_preprocessing else review_text

    # Detect if any idioms were preprocessed
    idioms_detected = []
    if use_idiom_preprocessing and processed_text != review_text:
        for pattern in POSITIVE_IDIOMS.keys():
            if re.search(pattern, review_text, re.IGNORECASE):
                idioms_detected.append("positive idiom")
                break
        for pattern in NEGATIVE_IDIOMS.keys():
            if re.search(pattern, review_text, re.IGNORECASE):
                idioms_detected.append("negative idiom")
                break

    X = vectorizer.transform([processed_text])
    pred = clf.predict(X)[0]

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        confidence = float(proba.max())
        # Get probability for the predicted class
        pos_prob = float(proba[1]) if len(proba) > 1 else 0.5
        neg_prob = float(proba[0]) if len(proba) > 0 else 0.5
    else:
        confidence = 1.0
        pos_prob = 1.0 if pred == 1 else 0.0
        neg_prob = 1.0 if pred == 0 else 0.0

    sentiment = "positive" if pred == 1 else "negative"
    elapsed_ms = (time.time() - start_time) * 1000

    # Generate reasoning
    idiom_note = ""
    if idioms_detected:
        idiom_note = f" [Context-aware: detected {', '.join(set(idioms_detected))}]"

    reasoning = f"ML model prediction based on TF-IDF features. Confidence: {confidence:.1%} (pos: {pos_prob:.1%}, neg: {neg_prob:.1%}){idiom_note}"

    # Create compound score that represents the confidence and direction
    compound = (pos_prob - neg_prob)  # Range: -1 to 1

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "reasoning": reasoning,
        "key_phrases": [],  # ML model doesn't extract phrases
        "scores": {"compound": compound, "pos": pos_prob, "neg": neg_prob, "neu": 0.0},
        "processing_time_ms": elapsed_ms,
        "elapsed_ms": elapsed_ms,
        "idioms_detected": bool(idioms_detected)
    }


def analyze_sentiment_vader(review_text: str, use_idiom_preprocessing: bool = True) -> Dict[str, Any]:
    """
    Use VADER sentiment analysis - a rule-based model optimized for social media text.
    VADER is extremely fast (microseconds) compared to LLMs (seconds).

    Args:
        review_text: The text to analyze
        use_idiom_preprocessing: If True, preprocess idioms before analysis (recommended)
    """
    start_time = time.time()

    # Preprocess idioms to handle context-dependent phrases
    processed_text = preprocess_idioms(review_text) if use_idiom_preprocessing else review_text

    analyzer = get_vader_analyzer()
    scores = analyzer.polarity_scores(processed_text)

    # Determine overall sentiment
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Detect if any idioms were preprocessed
    idioms_detected = []
    if use_idiom_preprocessing and processed_text != review_text:
        # Find which idioms were matched
        for pattern in POSITIVE_IDIOMS.keys():
            if re.search(pattern, review_text, re.IGNORECASE):
                idioms_detected.append("positive idiom")
                break
        for pattern in NEGATIVE_IDIOMS.keys():
            if re.search(pattern, review_text, re.IGNORECASE):
                idioms_detected.append("negative idiom")
                break

    # Extract key phrases by finding highly polar sentences
    sentences = re.split(r'[.!?]+', processed_text)
    key_phrases = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            sent_scores = analyzer.polarity_scores(sentence)
            if abs(sent_scores['compound']) > 0.3:  # Significant sentiment
                # Take first few words as key phrase
                words = sentence.split()[:6]
                if len(words) > 0:
                    key_phrases.append(' '.join(words))

    key_phrases = key_phrases[:3]  # Limit to top 3

    # Calculate confidence based on compound score magnitude
    confidence = min(abs(compound), 1.0)

    # Generate reasoning
    idiom_note = ""
    if idioms_detected:
        idiom_note = f" [Context-aware: detected {', '.join(set(idioms_detected))}]"

    if sentiment == "positive":
        reasoning = f"High positive sentiment detected (compound: {compound:.3f}). Positive indicators: {scores['pos']:.2f}, Negative: {scores['neg']:.2f}{idiom_note}"
    elif sentiment == "negative":
        reasoning = f"High negative sentiment detected (compound: {compound:.3f}). Negative indicators: {scores['neg']:.2f}, Positive: {scores['pos']:.2f}{idiom_note}"
    else:
        reasoning = f"Neutral or mixed sentiment (compound: {compound:.3f}). Balanced positive ({scores['pos']:.2f}) and negative ({scores['neg']:.2f}) language{idiom_note}"

    elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "reasoning": reasoning,
        "key_phrases": key_phrases,
        "scores": scores,
        "processing_time_ms": elapsed_time,
        "idioms_detected": bool(idioms_detected)
    }


def preprocess_for_topics(text: str) -> List[str]:
    """Preprocess text for topic modeling."""
    # Tokenize
    tokens = word_tokenize(text.lower())

    # Remove stopwords and short words
    stop_words = set(stopwords.words('english'))
    tokens = [
        word for word in tokens
        if word.isalnum() and word not in stop_words and len(word) > 3
    ]

    return tokens


@st.cache_data
def extract_topics_lda(reviews: List[Dict], num_topics: int = 5, words_per_topic: int = 5) -> Dict[str, Any]:
    """
    Use Latent Dirichlet Allocation (LDA) for topic modeling.
    LDA is a traditional statistical model that's much faster than LLM-based approaches.
    """
    start_time = time.time()

    # Preprocess all reviews
    processed_docs = [preprocess_for_topics(r['review']) for r in reviews]

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # Train LDA model
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto'
    )

    # Extract topics
    topics = []
    for idx in range(num_topics):
        topic_words = lda_model.show_topic(idx, topn=words_per_topic)
        topics.append({
            'id': idx,
            'words': [word for word, _ in topic_words],
            'weights': [float(weight) for _, weight in topic_words]
        })

    elapsed_time = (time.time() - start_time) * 1000

    return {
        'topics': topics,
        'model': lda_model,
        'dictionary': dictionary,
        'corpus': corpus,
        'processing_time_ms': elapsed_time,
        'elapsed_ms': elapsed_time
    }


@st.cache_data
def extract_topics_nmf(reviews: List[Dict], num_topics: int = 5, words_per_topic: int = 5) -> Dict[str, Any]:
    """
    Use Non-negative Matrix Factorization (NMF) for topic modeling.
    NMF is a matrix factorization approach that's trained in runtime on the current dataset.
    """
    start_time = time.time()

    docs = [r["review"] for r in reviews]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000
    )
    X = vectorizer.fit_transform(docs)

    nmf = NMF(n_components=num_topics, random_state=42, max_iter=300)
    W = nmf.fit_transform(X)
    H = nmf.components_
    terms = vectorizer.get_feature_names_out()

    topics = []
    for topic_idx, topic_vec in enumerate(H):
        top_indices = topic_vec.argsort()[:-words_per_topic - 1:-1]
        words = [terms[i] for i in top_indices]
        weights = [float(topic_vec[i]) for i in top_indices]

        topics.append({
            "id": topic_idx,
            "words": words,
            "weights": weights,
        })

    elapsed_time = (time.time() - start_time) * 1000
    return {
        "topics": topics,
        "processing_time_ms": elapsed_time,
        "elapsed_ms": elapsed_time
    }


def is_valid_phrase(phrase: str) -> bool:
    """
    Check if a phrase is valid (not just punctuation, not too short, etc.).
    """
    if not phrase or len(phrase) < 2:
        return False

    # Remove common punctuation
    cleaned = phrase.strip("'\",.!?;:-()[]{}").strip()

    if not cleaned or len(cleaned) < 2:
        return False

    # Must contain at least one letter
    if not any(c.isalpha() for c in cleaned):
        return False

    # Filter out common noise words that RAKE might miss
    noise_words = {'however', 'therefore', 'moreover', 'furthermore', 'thus', 'hence'}
    if cleaned.lower() in noise_words:
        return False

    return True


def clean_phrase(phrase: str) -> str:
    """Clean a phrase by removing extra punctuation, whitespace, and trailing stopwords."""
    # Strip common punctuation from ends
    cleaned = phrase.strip("'\",.!?;:-()[]{}").strip()

    # Normalize whitespace
    cleaned = ' '.join(cleaned.split())

    # Remove trailing/leading stopwords and transition words
    noise_words = {'however', 'therefore', 'moreover', 'furthermore', 'thus', 'hence',
                   'although', 'though', 'but', 'yet', 'still', 'even'}

    words = cleaned.split()
    # Remove leading noise words
    while words and words[0].lower() in noise_words:
        words.pop(0)
    # Remove trailing noise words
    while words and words[-1].lower() in noise_words:
        words.pop()

    cleaned = ' '.join(words)
    return cleaned


def extract_topics_from_single_review(review_text: str) -> Dict[str, Any]:
    """
    Extract topics from a single review using RAKE (Rapid Automatic Keyword Extraction).
    RAKE is a traditional NLP algorithm that extracts keyphrases based on word co-occurrence.
    """
    start_time = time.time()

    # Use RAKE for keyphrase extraction with better configuration
    rake = Rake(
        min_length=1,  # Minimum words per phrase
        max_length=4,  # Maximum words per phrase (avoid very long phrases)
    )
    rake.extract_keywords_from_text(review_text)
    ranked_phrases = rake.get_ranked_phrases()  # already sorted by importance

    # Filter and clean the phrases
    valid_phrases = []
    for phrase in ranked_phrases:
        if is_valid_phrase(phrase):
            cleaned = clean_phrase(phrase)
            # Re-validate after cleaning (in case all words were noise)
            if cleaned and len(cleaned) >= 2 and len(cleaned.split()) <= 4:  # Prefer shorter, focused phrases
                valid_phrases.append(cleaned)

        # Stop when we have enough good phrases
        if len(valid_phrases) >= 5:
            break

    top_phrases = valid_phrases[:5] if valid_phrases else []

    # If RAKE didn't produce good results, fall back to frequency-based
    if len(top_phrases) < 3:
        tokens = preprocess_for_topics(review_text)
        word_freq = Counter(tokens)
        top_words = word_freq.most_common(5)
        # Add single words as fallback
        for word, _ in top_words:
            if word not in top_phrases and len(top_phrases) < 5:
                top_phrases.append(word)

    # Also get individual words for aspect mapping
    tokens = preprocess_for_topics(review_text)
    word_freq = Counter(tokens)
    top_words = word_freq.most_common(10)

    # Map to common aspect categories
    aspect_keywords = {
        'quality': ['quality', 'excellent', 'premium', 'perfect', 'great', 'amazing', 'incredible', 'good', 'solid', 'wonderful', 'superb', 'outstanding'],
        'price': ['price', 'expensive', 'cheap', 'worth', 'value', 'money', 'cost', 'affordable', 'overpriced'],
        'comfort': ['comfortable', 'comfort', 'ergonomic', 'soft', 'cozy'],
        'durability': ['durable', 'sturdy', 'strong', 'broke', 'broken', 'stopped', 'working', 'lasted', 'reliable'],
        'design': ['design', 'looks', 'style', 'appearance', 'beautiful', 'gorgeous', 'attractive', 'ugly'],
        'performance': ['performance', 'works', 'working', 'responsive', 'fast', 'slow', 'effective'],
        'support': ['support', 'service', 'help', 'customer', 'warranty'],
        'shipping': ['shipping', 'delivery', 'arrived', 'package', 'packaging'],
        'size': ['size', 'large', 'small', 'fits', 'fitting', 'dimensions'],
    }

    # Identify aspects mentioned - check both phrases and individual words
    aspects = {}
    for aspect, keywords in aspect_keywords.items():
        # Check if any keyword appears in our extracted phrases or words
        found_keyword = None

        # First check phrases (better context)
        for phrase in top_phrases:
            phrase_lower = phrase.lower()
            for keyword in keywords:
                if keyword in phrase_lower:
                    found_keyword = keyword
                    break
            if found_keyword:
                break

        # If not found in phrases, check individual words
        if not found_keyword:
            for word, _ in top_words:
                if word in keywords:
                    found_keyword = word
                    break

        if found_keyword:
            # Determine sentiment of this aspect using VADER on sentences containing the keyword
            analyzer = get_vader_analyzer()
            sentences_with_word = [s for s in re.split(r'[.!?]+', review_text) if found_keyword in s.lower()]
            if sentences_with_word:
                # Preprocess the sentences for idioms before sentiment analysis
                combined_text = ' '.join(sentences_with_word)
                processed_text = preprocess_idioms(combined_text)
                aspect_sentiment = analyzer.polarity_scores(processed_text)['compound']
                if aspect_sentiment >= 0.05:
                    aspects[aspect] = 'positive'
                elif aspect_sentiment <= -0.05:
                    aspects[aspect] = 'negative'
                else:
                    aspects[aspect] = 'neutral'

    # Generate summary using RAKE phrases
    if top_phrases:
        summary = f"Key topics: {', '.join(top_phrases[:3])}"
    else:
        topics_list = [word for word, _ in top_words]
        summary = f"Review focuses on: {', '.join(topics_list[:3])}"

    elapsed_time = (time.time() - start_time) * 1000

    return {
        "topics": top_phrases if top_phrases else [word for word, _ in top_words[:5]],
        "top_phrases": top_phrases,
        "aspects": aspects,
        "summary": summary,
        "processing_time_ms": elapsed_time,
        "elapsed_ms": elapsed_time
    }


def get_sentiment_color(sentiment: str) -> str:
    """Return color based on sentiment."""
    colors = {
        "positive": "#10b981",  # green
        "negative": "#ef4444",  # red
        "neutral": "#f59e0b",   # yellow
    }
    return colors.get(sentiment.lower(), "#6b7280")


def get_sentiment_emoji(sentiment: str) -> str:
    """Return emoji based on sentiment."""
    emojis = {
        "positive": "😊",
        "negative": "😞",
        "neutral": "😐",
    }
    return emojis.get(sentiment.lower(), "🤔")


def build_review_card_html(review: Dict[str, Any], sentiment_result: Dict[str, Any], topic_result: Dict[str, Any]) -> str:
    """Return HTML for a single review card without leading indentation."""
    sentiment = sentiment_result["sentiment"]
    sentiment_color = get_sentiment_color(sentiment)
    sentiment_emoji = get_sentiment_emoji(sentiment)
    stars = "⭐" * review["rating"] + "☆" * (5 - review["rating"])

    aspects = topic_result.get("aspects", {})
    topics = topic_result.get("topics", [])[:3]
    idioms_detected = sentiment_result.get("idioms_detected", False)

    lines = [
        "<div class='review-card'>",
        f"<div class='review-header' style='border-left-color: {sentiment_color};'>",
        "<div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;'>",
        "<div style='flex: 1;'>",
        f"<strong style='color: #1f2937; font-size: 1rem;'>{html.escape(review['product'])}</strong>",
        "</div>",
        f"<div style='font-size: 1rem;'>{stars}</div>",
        "</div>",
        "<div style='color: #4b5563; line-height: 1.5; font-size: 0.95rem; font-style: italic; margin-bottom: 0.8rem;'>",
        f"\"{html.escape(review['review'])}\"",
        "</div>",
        "<div style='margin-bottom: 0.8rem;'>",
        f"<span class='sentiment-badge' style='background: {sentiment_color};'>",
        f"{sentiment_emoji} {sentiment.upper()}",
        "</span>",
        f"<span class='performance-badge'>⚡ {sentiment_result['processing_time_ms']:.1f}ms</span>",
    ]

    # Add idiom detection badge if applicable
    if idioms_detected:
        lines.append("<span class='idiom-badge' title='Context-aware idiom detected'>🧠 Context-Aware</span>")

    lines.append("</div>")  # Close badges div

    if topics:
        lines.append("<div style='margin-top: 0.3rem;'>")
        for topic in topics:
            lines.append(f"<span class='topic-chip'>🏷️ {html.escape(topic)}</span>")
        lines.append("</div>")

    if aspects:
        lines.append("<div style='margin-top: 0.5rem;'>")
        for aspect, aspect_sentiment in aspects.items():
            aspect_color = get_sentiment_color(aspect_sentiment)
            lines.append(
                f"<span class='aspect-badge' style='background: {aspect_color}; color: white;'>{html.escape(aspect.title())}</span>"
            )
        lines.append("</div>")

    lines.append(build_analysis_details_html(review, sentiment_result, topic_result))

    lines.extend(
        [
            "</div>",  # Close review-header
            "</div>",  # Close review-card
        ]
    )

    return "\n".join(lines)


def build_analysis_details_html(review: Dict[str, Any], sentiment_result: Dict[str, Any], topic_result: Dict[str, Any]) -> str:
    """Return expandable HTML with sentiment/topic details."""
    scores = sentiment_result.get('scores', {})
    key_phrases = sentiment_result.get('key_phrases') or []
    confidence_pct = sentiment_result.get('confidence', 0) * 100
    reasoning = html.escape(sentiment_result.get('reasoning', ''))

    key_phrase_items = "".join(f"<li>{html.escape(phrase)}</li>" for phrase in key_phrases)
    key_phrases_block = (
        f"<div><strong>Key Phrases:</strong><ul class='analysis-list'>{key_phrase_items}</ul></div>"
        if key_phrase_items else ""
    )

    aspects = topic_result.get("aspects", {})
    aspect_lines = []
    if aspects:
        for aspect, aspect_sentiment in aspects.items():
            aspect_color = get_sentiment_color(aspect_sentiment)
            aspect_emoji = get_sentiment_emoji(aspect_sentiment)
            aspect_lines.append(
                f"<div class='aspect-detail' style='border-left-color: {aspect_color};'>"
                f"{html.escape(aspect.title())}: {aspect_emoji} {aspect_sentiment.title()}"
                "</div>"
            )
    else:
        aspect_lines.append("<p style='color: #6b7280;'>No specific product aspects detected.</p>")
    aspect_items = "".join(aspect_lines)

    topic_summary = html.escape(topic_result.get("summary", ""))
    topic_summary_block = f"<p><strong>Summary:</strong> {topic_summary}</p>" if topic_summary else ""

    processing_ms = topic_result.get("processing_time_ms", 0.0)

    # Tooltip helper function
    def tooltip(text):
        return f'<span class="metric-tooltip">?<span class="tooltip-text">{text}</span></span>'

    # Get scores for interpretation
    compound = scores.get('compound', 0)
    pos_score = scores.get('pos', 0)
    neu_score = scores.get('neu', 0)
    neg_score = scores.get('neg', 0)

    sentiment_details = "".join([
        "<div class='analysis-column'>",
        "<h4>Sentiment Details</h4>",
        f"<p><strong>Confidence:</strong> {confidence_pct:.1f}%"
        f"{tooltip('How certain the model is about its prediction. Higher is more confident. <strong>Good:</strong> >70%, <strong>Excellent:</strong> >85%')}</p>",
        f"<p><strong>Compound Score:</strong> {compound:.3f}"
        f"{tooltip('Overall sentiment score combining positive, negative, and neutral. Range: -1 (most negative) to +1 (most positive). <strong>Positive:</strong> ≥0.05, <strong>Negative:</strong> ≤-0.05, <strong>Neutral:</strong> between -0.05 and 0.05')}</p>",
        "<ul class='analysis-list'>",
        f"<li><strong>Positive:</strong> {pos_score:.2f}"
        f"{tooltip('Proportion of text with positive sentiment. Range: 0-1. <strong>High positive:</strong> >0.5 indicates strong positive language')}</li>",
        f"<li><strong>Neutral:</strong> {neu_score:.2f}"
        f"{tooltip('Proportion of text that is neutral. Range: 0-1. <strong>High neutral:</strong> >0.5 indicates objective or factual language')}</li>",
        f"<li><strong>Negative:</strong> {neg_score:.2f}"
        f"{tooltip('Proportion of text with negative sentiment. Range: 0-1. <strong>High negative:</strong> >0.5 indicates strong negative language')}</li>",
        "</ul>",
        f"<div class='analysis-note'>{reasoning}</div>",
        key_phrases_block,
        "</div>",
    ])

    topic_details = "".join([
        "<div class='analysis-column'>",
        "<h4>Topic Analysis</h4>",
        f"<p><strong>Processing Time:</strong> {processing_ms:.2f}ms"
        f"{tooltip('How long it took to extract topics and keywords from this review using traditional NLP (RAKE algorithm). <strong>Typical:</strong> 5-20ms, which is 100-1000x faster than LLM-based topic extraction')}</p>",
        f"<div><strong>Aspect-Based Sentiment:</strong>"
        f"{tooltip('Specific product aspects (quality, price, etc.) mentioned in this review and their associated sentiment. Helps identify what the reviewer liked or disliked.')}"
        f"{aspect_items}</div>",
        topic_summary_block,
        "</div>",
    ])

    return "".join([
        "<details class='analysis-details'>",
        "<summary>📊 View Full Analysis</summary>",
        "<div class='analysis-columns'>",
        sentiment_details,
        topic_details,
        "</div>",
        "</details>",
    ])


@st.cache_data
def calculate_aggregate_stats(reviews: List[Dict]) -> Dict[str, Any]:
    """Calculate aggregate statistics from all reviews."""
    total = len(reviews)
    avg_rating = sum(r["rating"] for r in reviews) / total

    rating_distribution = Counter(r["rating"] for r in reviews)

    return {
        "total_reviews": total,
        "average_rating": avg_rating,
        "rating_distribution": dict(rating_distribution),
    }


def analyze_topic_sentiment_across_reviews(reviews: List[Dict], topic_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate aspect-based sentiment across all reviews to identify which topics
    are consistently viewed positively or negatively.

    Returns a dictionary with topic statistics including:
    - Total mentions
    - Sentiment breakdown (positive/negative/neutral counts)
    - Percentage positive/negative
    - Overall sentiment trend
    """
    # Aggregate aspects from all reviews
    topic_sentiments = {}

    for review in reviews:
        review_id = review['id']
        if review_id not in topic_results:
            continue

        aspects = topic_results[review_id].get('aspects', {})

        for aspect, sentiment in aspects.items():
            if aspect not in topic_sentiments:
                topic_sentiments[aspect] = {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0,
                    'total': 0,
                    'reviews': []
                }

            topic_sentiments[aspect][sentiment] += 1
            topic_sentiments[aspect]['total'] += 1
            topic_sentiments[aspect]['reviews'].append({
                'id': review_id,
                'rating': review['rating'],
                'sentiment': sentiment
            })

    # Calculate statistics for each topic
    topic_stats = []
    for topic, data in topic_sentiments.items():
        total = data['total']
        if total == 0:
            continue

        positive_pct = (data['positive'] / total) * 100
        negative_pct = (data['negative'] / total) * 100
        neutral_pct = (data['neutral'] / total) * 100

        # Determine overall trend
        if positive_pct >= 60:
            trend = 'positive'
        elif negative_pct >= 60:
            trend = 'negative'
        else:
            trend = 'mixed'

        topic_stats.append({
            'topic': topic.title(),
            'total_mentions': total,
            'positive_count': data['positive'],
            'negative_count': data['negative'],
            'neutral_count': data['neutral'],
            'positive_pct': positive_pct,
            'negative_pct': negative_pct,
            'neutral_pct': neutral_pct,
            'trend': trend,
            'net_sentiment': positive_pct - negative_pct  # Net positive sentiment
        })

    # Sort by total mentions (most discussed topics first)
    topic_stats.sort(key=lambda x: x['total_mentions'], reverse=True)

    return {
        'topics': topic_stats,
        'total_topics': len(topic_stats)
    }


def extract_keywords_tfidf(reviews: List[Dict], top_n: int = 15) -> List[Tuple[str, float]]:
    """Extract important keywords using TF-IDF."""
    texts = [r["review"] for r in reviews]

    # Remove common stop words and use TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=top_n,
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams
        min_df=1,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # Get average TF-IDF score for each term
        avg_scores = tfidf_matrix.mean(axis=0).A1
        keywords = [(feature_names[i], avg_scores[i]) for i in range(len(feature_names))]
        keywords.sort(key=lambda x: x[1], reverse=True)

        return keywords
    except Exception:
        return []


@st.cache_data
def batch_analyze_sentiments(reviews: List[Dict]) -> Dict[int, Dict[str, Any]]:
    """Batch analyze all reviews to demonstrate speed."""
    start_time = time.time()
    results = {}

    for review in reviews:
        results[review['id']] = analyze_sentiment_vader(review['review'])

    total_time = (time.time() - start_time) * 1000
    avg_time = total_time / len(reviews)

    return {
        'results': results,
        'total_time_ms': total_time,
        'avg_time_ms': avg_time
    }


def sentiment_analysis_page() -> None:
    """Render the sentiment analysis and topic modeling page."""

    # Custom CSS for presentation-friendly layout
    st.markdown(
        """
        <style>
        .review-card {
            background: white;
            padding: 0;
            border-radius: 12px;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: all 0.3s ease;
        }
        .review-card:hover {
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }
        .review-header {
            padding: 1.2rem;
            border-left: 5px solid;
        }
        .review-content {
            padding: 0 1.2rem 1.2rem 1.2rem;
        }
        .sentiment-badge {
            display: inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            color: white;
        }
        .topic-chip {
            display: inline-block;
            background: #e0e7ff;
            color: #4338ca;
            padding: 0.25rem 0.6rem;
            border-radius: 5px;
            margin: 0.2rem;
            font-size: 0.8rem;
            font-weight: 500;
        }
        .performance-badge {
            display: inline-block;
            background: #10b981;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
            margin-left: 0.3rem;
        }
        .idiom-badge {
            display: inline-block;
            background: #8b5cf6;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
            margin-left: 0.3rem;
        }
        .reviews-container {
            max-height: 600px;
            overflow-y: auto;
            padding-right: 0.5rem;
        }
        .reviews-container::-webkit-scrollbar {
            width: 8px;
        }
        .reviews-container::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        .reviews-container::-webkit-scrollbar-thumb {
            background: #3b82f6;
            border-radius: 4px;
        }
        .reviews-container::-webkit-scrollbar-thumb:hover {
            background: #4f46e5;
        }
        .aspect-badge {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin: 0.1rem;
        }
        .analysis-details {
            background: #f9fafb;
            border-radius: 8px;
            margin-top: 0.8rem;
            border: 1px solid #e5e7eb;
        }
        .analysis-details summary {
            cursor: pointer;
            padding: 0.8rem 1rem;
            font-weight: 600;
            color: #1f2937;
            list-style: none;
            position: relative;
        }
        .analysis-details summary::after {
            content: "▾";
            position: absolute;
            right: 1rem;
            transition: transform 0.2s ease;
        }
        .analysis-details summary::-webkit-details-marker {
            display: none;
        }
        .analysis-details[open] summary {
            border-bottom: 1px solid #e5e7eb;
            background: #eef2ff;
        }
        .analysis-details[open] summary::after {
            transform: rotate(180deg);
        }
        .analysis-columns {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            padding: 1rem;
        }
        .analysis-column {
            flex: 1;
            min-width: 240px;
            background: white;
            border-radius: 6px;
            padding: 0.8rem;
            border: 1px solid #e5e7eb;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.02);
        }
        .analysis-column h4 {
            margin-top: 0;
            margin-bottom: 0.4rem;
            font-size: 1rem;
            color: #111827;
        }
        .analysis-list {
            margin: 0.3rem 0 0 1rem;
            padding: 0;
        }
        .analysis-note {
            background: #eef2ff;
            padding: 0.6rem;
            border-radius: 6px;
            margin-top: 0.6rem;
            border-left: 4px solid #6366f1;
            color: #3730a3;
            font-size: 0.9rem;
        }
        .aspect-detail {
            padding: 0.35rem 0.5rem;
            margin: 0.2rem 0;
            background: #f3f4f6;
            border-left: 4px solid #10b981;
            border-radius: 4px;
            font-size: 0.9rem;
            color: #1f2937;
        }
        .metric-tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
            color: #6366f1;
            font-weight: 600;
            margin-left: 0.2rem;
            font-size: 0.85rem;
        }
        .metric-tooltip .tooltip-text {
            visibility: hidden;
            width: 280px;
            background-color: #1f2937;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 0.8rem;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -140px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.85rem;
            font-weight: 400;
            line-height: 1.4;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .metric-tooltip .tooltip-text::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #1f2937 transparent transparent transparent;
        }
        .metric-tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown("# 💭 Sentiment Analysis Lab")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3b82f6; margin-bottom: 2rem;'>
            <p style='margin: 0; color: #4b5563; font-size: 1.05rem;'>
                <strong>What you'll learn:</strong> How traditional NLP models can analyze sentiment and extract topics
                <em>without expensive LLM API calls</em> - delivering results in <strong>milliseconds instead of seconds</strong>.
            </p>
            <p style='margin: 0.8rem 0 0 0; color: #4b5563;'>
                <strong>Key advantage:</strong> Traditional models like VADER & TF-IDF+LogReg (sentiment) and LDA & NMF (topics) are
                <strong>100-1000x faster</strong> than LLMs, cost nothing to run, and work offline. All models train and run
                in realtime within this app!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Sentiment analysis classifies the emotional tone of text—positive, negative, or neutral—"
        "and can extend to fine-grained emotions. Use the panels below to see how traditional models "
        "infer attitude from reviews in milliseconds."
    )

    # Performance comparison callout
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        padding: 1.2rem; border-radius: 10px; margin-bottom: 1.5rem; border-left: 4px solid #10b981;'>
            <strong style='color: #065f46; font-size: 1.1rem;'>⚡ Traditional NLP vs. LLM Performance</strong>
            <div style='margin-top: 0.5rem; color: #065f46;'>
                <div>• <strong>VADER Sentiment:</strong> ~1-5ms per review vs. ~500-2000ms with GPT-4</div>
                <div>• <strong>ML Sentiment (TF-IDF+LogReg):</strong> ~5-15ms per review (trained live in-app!)</div>
                <div>• <strong>LDA/NMF Topic Modeling:</strong> ~100-500ms for entire dataset vs. ~10-30 seconds with LLMs</div>
                <div>• <strong>RAKE Keyphrase Extraction:</strong> ~5-20ms per review vs. ~500ms+ with LLMs</div>
                <div>• <strong>Cost:</strong> $0 (runs locally) vs. $0.01-0.10 per review with API calls</div>
                <div>• <strong>Scalability:</strong> Process millions of reviews per hour on a single machine</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Context-aware idiom handling callout
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%);
        padding: 1.2rem; border-radius: 10px; margin-bottom: 1.5rem; border-left: 4px solid #8b5cf6;'>
            <strong style='color: #5b21b6; font-size: 1.1rem;'>🧠 Smart Idiom Detection</strong>
            <div style='margin-top: 0.5rem; color: #5b21b6;'>
                <div>Our models now handle context-dependent phrases intelligently:</div>
                <div style='margin-top: 0.5rem;'>
                    <div>✅ <strong>"went crazy for it"</strong> → Recognized as POSITIVE (loved it)</div>
                    <div>✅ <strong>"driving me crazy"</strong> → Recognized as NEGATIVE (frustrating)</div>
                    <div>✅ <strong>"to die for"</strong> → Recognized as POSITIVE (wonderful)</div>
                    <div>✅ <strong>"died on arrival"</strong> → Recognized as NEGATIVE (broken)</div>
                </div>
                <div style='margin-top: 0.5rem; font-size: 0.9rem;'>
                    Look for the <span style='background: #8b5cf6; color: white; padding: 0.1rem 0.4rem; border-radius: 3px;'>🧠 Context-Aware</span> badge on reviews!
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "sentiment_results" not in st.session_state:
        st.session_state["sentiment_results"] = {}
    if "topic_results" not in st.session_state:
        st.session_state["topic_results"] = {}
    if "reviews_loaded" not in st.session_state:
        st.session_state["reviews_loaded"] = False
    if "sample_reviews" not in st.session_state:
        st.session_state["sample_reviews"] = []
    if "expanded_reviews" not in st.session_state:
        st.session_state["expanded_reviews"] = set()
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = 1

    # Model Selection Controls
    st.markdown("## ⚙️ Model Configuration")

    # Model comparison overview
    st.markdown(
        """
        <div style='background: #fef3c7; padding: 1rem; border-radius: 8px; border-left: 4px solid #f59e0b; margin-bottom: 1.5rem;'>
            <strong>Choose Your Analysis Models</strong>
            <p style='margin: 0.5rem 0; color: #78350f;'>
                Select which traditional NLP models to use for sentiment analysis and topic extraction.
                Each model has different strengths - experiment to see which works best for your data!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        sentiment_model_choice = st.radio(
            "**Sentiment Analysis Model** (Traditional NLP)",
            ["VADER (lexicon-based)", "TF-IDF + Linear Model (trained live)"],
            help="Choose between rule-based VADER or a machine learning model trained on the dataset"
        )

        # Add detailed comparison for sentiment models
        if sentiment_model_choice == "VADER (lexicon-based)":
            st.markdown(
                """
                <div style='background: #f0fdf4; padding: 0.8rem; border-radius: 6px; font-size: 0.9rem; margin-top: 0.5rem;'>
                    <strong style='color: #15803d;'>📖 VADER (Rule-Based)</strong>
                    <ul style='margin: 0.3rem 0; padding-left: 1.2rem; color: #166534;'>
                        <li><strong>How it works:</strong> Uses a pre-built dictionary of words with sentiment scores</li>
                        <li><strong>Speed:</strong> Extremely fast (1-5ms per review)</li>
                        <li><strong>Training:</strong> None required - ready to use</li>
                        <li><strong>Best for:</strong> General sentiment, social media text, quick analysis</li>
                        <li><strong>Accuracy:</strong> ~80-85% on product reviews</li>
                        <li><strong>Advantage:</strong> Works immediately, no training data needed, very interpretable</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style='background: #eff6ff; padding: 0.8rem; border-radius: 6px; font-size: 0.9rem; margin-top: 0.5rem;'>
                    <strong style='color: #1e40af;'>🤖 TF-IDF + Logistic Regression (ML)</strong>
                    <ul style='margin: 0.3rem 0; padding-left: 1.2rem; color: #1e3a8a;'>
                        <li><strong>How it works:</strong> Learns patterns from your data using machine learning</li>
                        <li><strong>Speed:</strong> Fast (5-15ms per review after training)</li>
                        <li><strong>Training:</strong> Trains live on your dataset (1-2 seconds for 2000 reviews)</li>
                        <li><strong>Best for:</strong> Dataset-specific patterns, domain-specific language</li>
                        <li><strong>Accuracy:</strong> ~85-90% on product reviews (often outperforms VADER)</li>
                        <li><strong>Advantage:</strong> Adapts to your specific data, learns domain vocabulary</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        topic_model_choice = st.radio(
            "**Topic Modeling Method** (Traditional NLP)",
            ["LDA (probabilistic)", "NMF (matrix factorization)"],
            help="Choose between probabilistic LDA or matrix factorization NMF"
        )

        # Add detailed comparison for topic models
        if topic_model_choice == "LDA (probabilistic)":
            st.markdown(
                """
                <div style='background: #fdf4ff; padding: 0.8rem; border-radius: 6px; font-size: 0.9rem; margin-top: 0.5rem;'>
                    <strong style='color: #86198f;'>🎲 LDA (Probabilistic Model)</strong>
                    <ul style='margin: 0.3rem 0; padding-left: 1.2rem; color: #701a75;'>
                        <li><strong>How it works:</strong> Uses probability distributions to find topics</li>
                        <li><strong>Speed:</strong> Fast (100-500ms for entire dataset)</li>
                        <li><strong>Approach:</strong> Documents are mixtures of topics, topics are mixtures of words</li>
                        <li><strong>Best for:</strong> Discovering overlapping themes, interpretable results</li>
                        <li><strong>Output:</strong> Each topic shows probability distribution over words</li>
                        <li><strong>Advantage:</strong> Handles documents with multiple themes naturally</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style='background: #fef2f2; padding: 0.8rem; border-radius: 6px; font-size: 0.9rem; margin-top: 0.5rem;'>
                    <strong style='color: #991b1b;'>🔢 NMF (Matrix Factorization)</strong>
                    <ul style='margin: 0.3rem 0; padding-left: 1.2rem; color: #7f1d1d;'>
                        <li><strong>How it works:</strong> Factorizes document-term matrix into topics</li>
                        <li><strong>Speed:</strong> Fast (100-500ms for entire dataset)</li>
                        <li><strong>Approach:</strong> Linear algebra decomposition with non-negative constraints</li>
                        <li><strong>Best for:</strong> Clear, distinct topics with sharp separation</li>
                        <li><strong>Output:</strong> Topic weights that are easier to interpret (all positive)</li>
                        <li><strong>Advantage:</strong> Often produces more coherent topics than LDA</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Section 1: Sample Data Display
    st.markdown("## 📋 Section 1: Real Amazon Review Data with Integrated Analysis")
    st.markdown("Loading real customer reviews from the **SetFit/amazon_reviews_multi_en** dataset on Hugging Face.")

    # Load data button
    if not st.session_state["reviews_loaded"]:
        col1, col2 = st.columns([3, 1])
        with col1:
            num_reviews = st.slider("Number of reviews to load:", min_value=10, max_value=100, value=30, step=10)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📥 Load Reviews from Dataset", type="primary", use_container_width=True):
                with st.spinner("Loading real Amazon reviews from Hugging Face..."):
                    try:
                        random_seed = random.randint(0, 2**32 - 1)
                        reviews = load_amazon_reviews(num_samples=num_reviews, seed=random_seed)
                        st.session_state["sample_reviews"] = reviews
                        st.session_state["reviews_loaded"] = True
                        st.success(f"✅ Loaded {len(reviews)} real Amazon reviews!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading dataset: {e}")
                        st.info("Make sure you have internet connection to download from Hugging Face.")
    else:
        reviews = st.session_state["sample_reviews"]

        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"✅ {len(reviews)} real Amazon reviews loaded from Hugging Face")
        with col2:
            if st.button("🔄 Reload Different Reviews", type="secondary", use_container_width=True):
                st.session_state["reviews_loaded"] = False
                st.session_state["sample_reviews"] = []
                st.session_state["sentiment_results"] = {}
                st.session_state["topic_results"] = {}
                st.session_state["expanded_reviews"] = set()
                st.session_state["current_page"] = 1
                if "ml_model" in st.session_state:
                    del st.session_state["ml_model"]
                st.rerun()

    if st.session_state["reviews_loaded"] and st.session_state["sample_reviews"]:
        reviews = st.session_state["sample_reviews"]

        # Train ML sentiment model if needed
        vectorizer = None
        clf = None
        if sentiment_model_choice == "TF-IDF + Linear Model (trained live)":
            if "ml_model" not in st.session_state:
                with st.spinner("Training ML sentiment model on dataset..."):
                    texts, labels = build_sentiment_labels_from_df(reviews)
                    if len(texts) > 0:
                        try:
                            # Convert to tuples for caching
                            vectorizer, clf = train_ml_sentiment_model(tuple(texts), tuple(labels))
                            st.session_state["ml_model"] = (vectorizer, clf)

                            # Show training data distribution
                            from collections import Counter
                            label_counts = Counter(labels)
                            pos_count = label_counts.get(1, 0)
                            neg_count = label_counts.get(0, 0)
                            st.success(f"✅ ML model trained on {len(texts)} reviews! (Positive: {pos_count}, Negative: {neg_count})")

                            if pos_count < 3 or neg_count < 3:
                                st.warning(f"⚠️ Limited training data detected. For best results, try loading more reviews or use VADER model instead.")
                        except Exception as e:
                            st.error(f"Failed to train ML model: {e}")
                            st.info("Falling back to VADER model. Try loading more reviews with diverse ratings.")
                            # Don't set the model, will fall back to VADER below
            else:
                vectorizer, clf = st.session_state["ml_model"]

        # Precompute analysis for all reviews (fast, enables dashboard metrics)
        for review in reviews:
            review_id = review['id']
            # Use cache key that includes model choice
            cache_key = f"{review_id}_{sentiment_model_choice}"
            if cache_key not in st.session_state["sentiment_results"]:
                if sentiment_model_choice == "VADER (lexicon-based)":
                    st.session_state["sentiment_results"][cache_key] = analyze_sentiment_vader(review['review'])
                else:
                    if vectorizer and clf:
                        st.session_state["sentiment_results"][cache_key] = analyze_sentiment_ml(review['review'], vectorizer, clf)
                    else:
                        st.session_state["sentiment_results"][cache_key] = analyze_sentiment_vader(review['review'])

            if review_id not in st.session_state["topic_results"]:
                st.session_state["topic_results"][review_id] = extract_topics_from_single_review(review['review'])

        # Cost & performance savings dashboard
        st.markdown("### 💰 Cost & Time Savings Dashboard")
        total_reviews = len(reviews)
        nlp_total_time_ms = sum(
            st.session_state["sentiment_results"][f"{r['id']}_{sentiment_model_choice}"]["processing_time_ms"]
            for r in reviews
        )
        avg_nlp_time_ms = nlp_total_time_ms / total_reviews if total_reviews else 0

        # Estimated LLM costs/times (assumptions based on GPT-4 style API)
        llm_time_per_review_ms = 1500  # 1.5s average latency
        llm_cost_per_review = 0.05     # $0.05 per review on average
        nlp_cost_per_review = 0.0

        total_llm_time_ms = total_reviews * llm_time_per_review_ms
        total_llm_cost = total_reviews * llm_cost_per_review
        total_nlp_cost = total_reviews * nlp_cost_per_review

        time_saved_ms = total_llm_time_ms - nlp_total_time_ms
        cost_saved = total_llm_cost - total_nlp_cost
        speedup_factor = total_llm_time_ms / nlp_total_time_ms if nlp_total_time_ms > 0 else 0

        # Convert times to appropriate units
        nlp_total_time_display = f"{nlp_total_time_ms:.0f} ms" if nlp_total_time_ms < 1000 else f"{nlp_total_time_ms/1000:.2f} sec"
        llm_total_time_display = f"{total_llm_time_ms/1000:.1f} sec" if total_llm_time_ms < 60000 else f"{total_llm_time_ms/60000:.1f} min"
        time_saved_display = f"{time_saved_ms/1000:.1f} sec" if time_saved_ms < 60000 else f"{time_saved_ms/60000:.1f} min"

        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #ecfccb 0%, #d9f99d 100%);
            padding: 1.2rem; border-radius: 12px; border-left: 4px solid #65a30d; margin: 1rem 0;'>
                <div style='display: flex; flex-wrap: wrap; gap: 1.5rem;'>
                    <div style='flex: 1; min-width: 200px;'>
                        <div style='font-size: 0.85rem; color: #4d7c0f;'>Traditional NLP Total Time</div>
                        <div style='font-size: 1.6rem; font-weight: 700; color: #1a2e05;'>{nlp_total_time_display}</div>
                        <div style='color: #4d7c0f;'>For {total_reviews} reviews</div>
                    </div>
                    <div style='flex: 1; min-width: 200px;'>
                        <div style='font-size: 0.85rem; color: #4d7c0f;'>Estimated LLM Total Time</div>
                        <div style='font-size: 1.6rem; font-weight: 700; color: #1a2e05;'>{llm_total_time_display}</div>
                        <div style='color: #4d7c0f;'>Same {total_reviews} reviews</div>
                    </div>
                    <div style='flex: 1; min-width: 200px;'>
                        <div style='font-size: 0.85rem; color: #4d7c0f;'>Time Saved</div>
                        <div style='font-size: 1.6rem; font-weight: 700; color: #1a2e05;'>{time_saved_display}</div>
                        <div style='color: #4d7c0f;'>{speedup_factor:.0f}x faster!</div>
                    </div>
                    <div style='flex: 1; min-width: 200px;'>
                        <div style='font-size: 0.85rem; color: #4d7c0f;'>Cost Saved</div>
                        <div style='font-size: 1.6rem; font-weight: 700; color: #1a2e05;'>${cost_saved:,.2f}</div>
                        <div style='color: #4d7c0f;'>vs. ${total_llm_cost:,.2f} in LLM fees</div>
                    </div>
                </div>
                <div style='margin-top: 0.8rem; font-size: 0.95rem; color: #365314;'>
                    ✅ <strong>At scale:</strong> Processing 1 million reviews would take ~{(nlp_total_time_ms / total_reviews * 1000000) / 3600000:.1f} hours with NLP vs. ~{(llm_time_per_review_ms * 1000000) / 3600000:.0f} hours with LLMs, saving ${(llm_cost_per_review * 1000000):,.0f} in API costs.
                </div>
                <div style='margin-top: 0.3rem; font-size: 0.8rem; color: #4d7c0f;'>
                    Assumptions: NLP average = {avg_nlp_time_ms:.2f}ms/review, LLM average = {llm_time_per_review_ms/1000:.1f}s/review, LLM cost = ${llm_cost_per_review:.02f}/review.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Scaling visualization
        st.markdown("#### 📈 How Traditional NLP Costs Scale vs. LLMs")

        # Generate data for scaling chart
        review_counts = [10, 50, 100, 500, 1000, 5000, 10000]
        nlp_costs = [count * nlp_cost_per_review for count in review_counts]
        llm_costs = [count * llm_cost_per_review for count in review_counts]

        # Cost scaling chart (full width)
        cost_scaling_df = pd.DataFrame({
            'Review Count': review_counts + review_counts,
            'Cost ($)': nlp_costs + llm_costs,
            'Method': ['Traditional NLP'] * len(review_counts) + ['LLM API'] * len(review_counts)
        })

        fig_cost = px.line(
            cost_scaling_df,
            x='Review Count',
            y='Cost ($)',
            color='Method',
            markers=True,
            title='Cost Scaling Comparison: Traditional NLP vs. LLM API',
            color_discrete_map={'Traditional NLP': '#10b981', 'LLM API': '#ef4444'}
        )
        fig_cost.update_layout(
            height=400,
            margin=dict(t=40, b=40, l=40, r=40),
            hovermode='x unified',
            xaxis=dict(
                title='Number of Reviews',
                tickmode='array',
                tickvals=review_counts,
                ticktext=[f'{x:,}' for x in review_counts]
            ),
            yaxis=dict(
                title='Cost ($)'
            )
        )
        # Add annotation for current dataset
        fig_cost.add_vline(
            x=total_reviews,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Your data ({total_reviews} reviews)",
            annotation_position="top"
        )
        st.plotly_chart(fig_cost, use_container_width=True)

        # Summary insights
        st.markdown(
            f"""
            <div style='background: #f0fdf4; padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981; margin: 1rem 0;'>
                <strong>📊 Key Insights at Scale:</strong>
                <ul style='margin: 0.5rem 0; color: #065f46;'>
                    <li><strong>10,000 reviews:</strong> NLP = {(avg_nlp_time_ms * 10000) / 1000:.1f}s (${nlp_cost_per_review * 10000:.2f}), LLM = {(llm_time_per_review_ms * 10000) / 3600000:.1f} hours (${llm_cost_per_review * 10000:,.0f})</li>
                    <li><strong>100,000 reviews:</strong> NLP = {(avg_nlp_time_ms * 100000) / 60000:.1f} min (${nlp_cost_per_review * 100000:.2f}), LLM = {(llm_time_per_review_ms * 100000) / 3600000:.1f} hours (${llm_cost_per_review * 100000:,.0f})</li>
                    <li><strong>1,000,000 reviews:</strong> NLP = {(avg_nlp_time_ms * 1000000) / 3600000:.1f} hours (${nlp_cost_per_review * 1000000:.2f}), LLM = {(llm_time_per_review_ms * 1000000) / 3600000:.0f} hours (${llm_cost_per_review * 1000000:,.0f})</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Display reviews with integrated analysis
        st.markdown("### 📝 Reviews (Click to expand full analysis)")

        # Pagination controls - fixed at 10 reviews per page
        total_reviews_count = len(reviews)
        reviews_per_page = 10

        # Calculate pagination
        total_pages = max(1, (total_reviews_count + reviews_per_page - 1) // reviews_per_page)
        current_page = min(st.session_state["current_page"], total_pages)  # Ensure current page is valid

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("◀ Prev", disabled=(current_page <= 1), use_container_width=True):
                st.session_state["current_page"] = max(1, current_page - 1)
                st.rerun()
        with col2:
            st.markdown(f"<div style='text-align: center; padding-top: 0.5rem;'>Page {current_page} of {total_pages} ({total_reviews_count} total reviews)</div>", unsafe_allow_html=True)
        with col3:
            if st.button("Next ▶", disabled=(current_page >= total_pages), use_container_width=True):
                st.session_state["current_page"] = min(total_pages, current_page + 1)
                st.rerun()

        # Calculate slice indices for current page
        start_idx = (current_page - 1) * reviews_per_page
        end_idx = min(start_idx + reviews_per_page, total_reviews_count)
        page_reviews = reviews[start_idx:end_idx]

        st.markdown("<div class='reviews-container'>", unsafe_allow_html=True)

        for idx, review in enumerate(page_reviews):
            review_id = review['id']
            cache_key = f"{review_id}_{sentiment_model_choice}"

            sentiment_result = st.session_state["sentiment_results"][cache_key]
            topic_result = st.session_state["topic_results"][review_id]

            review_card_html = build_review_card_html(review, sentiment_result, topic_result)

            st.markdown(review_card_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Bottom pagination controls
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 3, 2])
        with col1:
            st.markdown(f"Showing reviews {start_idx + 1}-{end_idx} of {total_reviews_count}")
        with col2:
            st.markdown(f"<div style='text-align: center;'>Page {current_page} of {total_pages}</div>", unsafe_allow_html=True)
        with col3:
            page_col1, page_col2 = st.columns(2)
            with page_col1:
                if st.button("◀ Previous", disabled=(current_page <= 1), use_container_width=True, key="prev_bottom"):
                    st.session_state["current_page"] = max(1, current_page - 1)
                    st.rerun()
            with page_col2:
                if st.button("Next ▶", disabled=(current_page >= total_pages), use_container_width=True, key="next_bottom"):
                    st.session_state["current_page"] = min(total_pages, current_page + 1)
                    st.rerun()

        st.divider()

        # Section 2: Aggregate Dashboards
        st.markdown("## 📊 Section 2: Aggregate Analysis Dashboard")
        st.markdown("Patterns and insights across all reviews in the dataset")

        # Calculate aggregate stats
        stats = calculate_aggregate_stats(reviews)

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Reviews", stats["total_reviews"])
        with col2:
            st.metric("Average Rating", f"{stats['average_rating']:.2f} ⭐")
        with col3:
            positive_count = sum(1 for r in reviews if r["rating"] >= 4)
            st.metric("Positive Reviews", f"{positive_count} ({positive_count/stats['total_reviews']*100:.0f}%)")
        with col4:
            negative_count = sum(1 for r in reviews if r["rating"] <= 2)
            st.metric("Negative Reviews", f"{negative_count} ({negative_count/stats['total_reviews']*100:.0f}%)")

        st.markdown("---")

        # Visualizations
        st.markdown("### Rating Distribution")
        rating_df = pd.DataFrame([
            {"Rating": f"{rating} Stars", "Count": count}
            for rating, count in sorted(stats["rating_distribution"].items(), reverse=True)
        ])
        fig1 = px.bar(
            rating_df,
            x="Rating",
            y="Count",
            color="Count",
            color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
        )
        fig1.update_layout(
            showlegend=False,
            height=350,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Topic Sentiment Analysis
        st.markdown("### 🎯 Topic Sentiment Analysis - What Do Customers Love/Hate?")
        st.markdown("See which product aspects are consistently viewed positively or negatively across all reviews.")

        # Analyze topic sentiment across all reviews
        topic_analysis = analyze_topic_sentiment_across_reviews(reviews, st.session_state["topic_results"])

        if topic_analysis['total_topics'] > 0:
            topics_data = topic_analysis['topics']

            # Create a summary metrics row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Topics Discussed", topic_analysis['total_topics'])
            with col2:
                # Find most positive topic
                most_positive = max(topics_data, key=lambda x: x['positive_pct'])
                st.metric(label="Most Positive", value=most_positive['topic'], delta=f"{most_positive['positive_pct']:.0f}% Positive", delta_color="normal")
            with col3:
                # Find most negative topic
                most_negative = max(topics_data, key=lambda x: x['negative_pct'])
                st.metric(label="Most Negative", value=most_negative['topic'], delta=f"{most_negative['negative_pct']:.0f}% Negative", delta_color="inverse")

            # Create visualization - horizontal stacked bar chart
            st.markdown("#### Sentiment Breakdown by Topic")

            # Prepare data for stacked bar chart
            topic_names = [t['topic'] for t in topics_data]
            positive_pcts = [t['positive_pct'] for t in topics_data]
            neutral_pcts = [t['neutral_pct'] for t in topics_data]
            negative_pcts = [t['negative_pct'] for t in topics_data]

            fig_topics = go.Figure()

            # Add positive bars
            fig_topics.add_trace(go.Bar(
                y=topic_names,
                x=positive_pcts,
                name='Positive',
                orientation='h',
                marker=dict(color='#10b981'),
                text=[f"{p:.0f}%" for p in positive_pcts],
                textposition='inside',
                hovertemplate='<b>%{y}</b><br>Positive: %{x:.1f}%<extra></extra>'
            ))

            # Add neutral bars
            fig_topics.add_trace(go.Bar(
                y=topic_names,
                x=neutral_pcts,
                name='Neutral',
                orientation='h',
                marker=dict(color='#f59e0b'),
                text=[f"{n:.0f}%" if n > 5 else "" for n in neutral_pcts],
                textposition='inside',
                hovertemplate='<b>%{y}</b><br>Neutral: %{x:.1f}%<extra></extra>'
            ))

            # Add negative bars
            fig_topics.add_trace(go.Bar(
                y=topic_names,
                x=negative_pcts,
                name='Negative',
                orientation='h',
                marker=dict(color='#ef4444'),
                text=[f"{n:.0f}%" for n in negative_pcts],
                textposition='inside',
                hovertemplate='<b>%{y}</b><br>Negative: %{x:.1f}%<extra></extra>'
            ))

            fig_topics.update_layout(
                barmode='stack',
                height=max(300, len(topics_data) * 50),
                margin=dict(t=20, b=20, l=120, r=20),
                xaxis=dict(title='Percentage (%)', range=[0, 100]),
                yaxis=dict(title=''),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                hovermode='closest'
            )

            st.plotly_chart(fig_topics, use_container_width=True)

            # Detailed table
            with st.expander("📊 Detailed Topic Statistics"):
                # Create DataFrame for display
                table_data = []
                for topic in topics_data:
                    trend_emoji = "😊" if topic['trend'] == 'positive' else ("😞" if topic['trend'] == 'negative' else "😐")
                    table_data.append({
                        'Topic': topic['topic'],
                        'Mentions': topic['total_mentions'],
                        'Positive': f"{topic['positive_count']} ({topic['positive_pct']:.0f}%)",
                        'Neutral': f"{topic['neutral_count']} ({topic['neutral_pct']:.0f}%)",
                        'Negative': f"{topic['negative_count']} ({topic['negative_pct']:.0f}%)",
                        'Net Sentiment': f"{topic['net_sentiment']:+.0f}%",
                        'Trend': f"{trend_emoji} {topic['trend'].title()}"
                    })

                df_topics = pd.DataFrame(table_data)
                st.dataframe(df_topics, use_container_width=True, hide_index=True)

                st.markdown("""
                **How to read this:**
                - **Net Sentiment** = (Positive % - Negative %). Higher is better!
                - **Trend**: Positive (≥60% positive), Negative (≥60% negative), Mixed (everything else)
                - Topics are sorted by frequency of mentions
                """)
        else:
            st.info("No aspects detected in reviews. Try loading more reviews or reviews with more detailed feedback.")

        st.markdown("---")

        # Topic modeling on entire dataset
        model_display_name = "LDA" if topic_model_choice == "LDA (probabilistic)" else "NMF"
        st.markdown(f"### 🔍 Topic Modeling ({model_display_name}) - Entire Dataset")

        # Add overview description
        st.markdown(
            """
            <div style='background: #f0f9ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #0284c7; margin: 1rem 0;'>
                <strong>What is Topic Modeling?</strong>
                <p style='margin: 0.5rem 0; color: #0c4a6e;'>
                    Topic modeling is an unsupervised machine learning technique that automatically discovers abstract "topics"
                    that occur in a collection of documents. Instead of analyzing individual reviews, it looks across your
                    entire dataset to find common themes and patterns.
                </p>
                <p style='margin: 0.5rem 0; color: #0c4a6e;'>
                    <strong>What's being extracted:</strong>
                </p>
                <ul style='margin: 0.5rem 0; color: #0c4a6e;'>
                    <li><strong>Hidden themes:</strong> Groups of words that frequently appear together (e.g., "battery", "charge", "life" might form a topic about battery performance)</li>
                    <li><strong>Topic distributions:</strong> Each topic is represented by its most important words with weights showing their relevance</li>
                    <li><strong>5 main topics:</strong> The model will identify 5 distinct themes that summarize what customers are talking about</li>
                    <li><strong>Top 6 words per topic:</strong> The most representative words that define each theme</li>
                </ul>
                <p style='margin: 0.5rem 0; color: #0c4a6e;'>
                    <strong>Use cases:</strong> Understanding broad customer concerns, identifying product features that matter most,
                    discovering unexpected patterns in feedback, and segmenting reviews by common themes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        topic_cache_key = f"topic_results_{topic_model_choice}"

        if st.button("🎯 Extract Topics from All Reviews", type="secondary"):
            if topic_model_choice == "LDA (probabilistic)":
                with st.spinner("Running LDA topic modeling..."):
                    topic_results = extract_topics_lda(reviews, num_topics=5, words_per_topic=6)
                    st.session_state[topic_cache_key] = topic_results
            else:
                with st.spinner("Running NMF topic modeling..."):
                    topic_results = extract_topics_nmf(reviews, num_topics=5, words_per_topic=6)
                    st.session_state[topic_cache_key] = topic_results

        if topic_cache_key in st.session_state:
            topic_results = st.session_state[topic_cache_key]
            proc_time = topic_results['processing_time_ms']

            st.markdown(f"**{model_display_name} Analysis completed in {proc_time:.0f}ms** ⚡")

            # Display topics
            for topic in topic_results['topics']:
                topic_id = topic['id']
                words = topic['words']
                weights = topic['weights']

                st.markdown(f"**Topic {topic_id + 1}:** {', '.join(words[:5])}")

                # Create horizontal bar for word weights
                topic_df = pd.DataFrame({
                    'Word': words,
                    'Weight': weights
                })

                fig = px.bar(
                    topic_df,
                    x='Weight',
                    y='Word',
                    orientation='h',
                    color='Weight',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    showlegend=False,
                    height=200,
                    margin=dict(t=10, b=10, l=10, r=10),
                    yaxis={'categoryorder': 'total ascending'},
                )
                st.plotly_chart(fig, use_container_width=True)

        # Keyword extraction
        st.markdown("### 🔑 Most Important Keywords (TF-IDF)")
        keywords = extract_keywords_tfidf(reviews, top_n=15)

        if keywords:
            keyword_df = pd.DataFrame(keywords, columns=["Keyword", "Importance"])
            fig3 = px.bar(
                keyword_df,
                x="Importance",
                y="Keyword",
                orientation='h',
                color="Importance",
                color_continuous_scale="Blues",
            )
            fig3.update_layout(
                showlegend=False,
                height=400,
                margin=dict(t=20, b=20, l=20, r=20),
                yaxis={'categoryorder': 'total ascending'},
            )
            st.plotly_chart(fig3, use_container_width=True)

    # Information expander
    with st.expander("ℹ️ How Traditional NLP Works"):
        st.markdown(
            """
            ### 🧠 Context-Aware Idiom Detection (NEW!)

            **The Problem:** Words can have different meanings in different contexts:
            - "went crazy **for** it" = POSITIVE (loved it) 😊
            - "driving me crazy" = NEGATIVE (frustrating) 😞
            - "to die for" = POSITIVE (wonderful) 😊
            - "died on arrival" = NEGATIVE (broken) 😞

            **Our Solution:** Pattern-based preprocessing that recognizes idiomatic expressions:

            **How it works:**
            1. **Pattern Matching:** Uses regex patterns to detect common idioms before sentiment analysis
            2. **Context Replacement:** Replaces idioms with sentiment-equivalent phrases
               - "went crazy for it" → "absolutely loved"
               - "driving me crazy" → "very frustrating"
            3. **Sentiment Analysis:** VADER/ML models then analyze the preprocessed text
            4. **Visual Indicator:** Reviews with detected idioms show a 🧠 Context-Aware badge

            **Examples of handled idioms:**
            - **Positive:** "crazy about", "to die for", "blew me away", "mind blowing", "insanely good"
            - **Negative:** "driving me crazy", "died on me", "made me sick"

            **Benefits:**
            - **Accuracy:** Significantly improves sentiment detection on colloquial text
            - **Speed:** Pattern matching adds <1ms overhead
            - **Transparency:** You can see which reviews triggered idiom detection
            - **Extensible:** Easy to add new patterns as you discover them

            **For ML Models:** The idiom preprocessing is applied to both training and inference data,
            helping the model learn better patterns from the start.

            ---

            ### VADER Sentiment Analysis

            **What it is:** Valence Aware Dictionary and sEntiment Reasoner - a rule-based model specifically tuned for social media text.

            **How it works:**
            - Uses a lexicon of words with pre-assigned sentiment scores
            - Accounts for capitalization, punctuation, degree modifiers ("very good" vs "good")
            - Handles negations, contrasts, and emoticons
            - No neural network or API calls - pure algorithmic analysis

            **Advantages:**
            - **Speed:** Processes text in 1-5 milliseconds
            - **Cost:** Completely free, runs locally
            - **Reliability:** Deterministic results, no hallucinations
            - **Privacy:** No data leaves your infrastructure
            - **Accuracy:** ~80-85% on product reviews (vs ~85-90% for fine-tuned LLMs)

            ---

            ### TF-IDF + Logistic Regression (ML Sentiment)

            **What it is:** A supervised machine learning model trained in realtime on the loaded dataset.

            **How it works:**
            - Converts text to TF-IDF features (term frequency-inverse document frequency)
            - Trains a Logistic Regression classifier on reviews with clear sentiment (1-2 stars = negative, 4-5 stars = positive)
            - Training happens once per session using Streamlit caching
            - Typically trains on 2000 samples in <1 second

            **Advantages:**
            - **Accuracy:** Often 85-90% on product reviews, can outperform VADER
            - **Speed:** Still very fast at 5-15ms per review after training
            - **Adaptability:** Learns patterns specific to your dataset
            - **Transparency:** Model trained live in the app, not a black box

            ---

            ### RAKE Keyphrase Extraction

            **What it is:** Rapid Automatic Keyword Extraction - extracts meaningful phrases from single reviews.

            **How it works:**
            - Splits text into candidate phrases using stop words as delimiters
            - Scores phrases based on word frequency and co-occurrence
            - Returns the most important multi-word phrases, not just single words

            **Advantages:**
            - **Better context:** Extracts "excellent customer service" vs just "excellent" and "service"
            - **Speed:** Processes in 5-20ms per review
            - **No training:** Works immediately on any text

            ---

            ### Aspect-Based Sentiment

            **What it is:** Identifying specific product aspects (quality, price, comfort) and their sentiment.

            **How it works:**
            - Extracts key terms from each review using RAKE
            - Maps terms to common product aspects (quality, price, performance, etc.)
            - Analyzes sentiment of sentences mentioning each aspect
            - Provides granular understanding beyond overall sentiment

            **NEW - Topic Sentiment Analysis Across Reviews:**
            - Aggregates aspect mentions across all reviews
            - Calculates sentiment distribution for each topic (% positive/negative/neutral)
            - Identifies which topics are consistently loved or hated
            - Visualizes trends with stacked bar charts

            **Business Value:**
            - Understand what customers like/dislike specifically
            - **Identify product strengths to emphasize in marketing** (e.g., "85% say Quality is excellent!")
            - **Prioritize improvements** based on topics with high negative sentiment
            - Track aspect sentiment over time
            - Make data-driven product decisions

            **Example Insights:**
            - "Performance: 75% positive" → Highlight in marketing
            - "Price: 60% negative" → Consider pricing strategy
            - "Quality: 90% positive, Shipping: 40% negative" → Product is great, logistics need work

            ---

            ### LDA Topic Modeling

            **What it is:** Latent Dirichlet Allocation - a probabilistic model for discovering abstract topics in documents.

            **How it works:**
            - Treats documents as mixtures of topics
            - Topics are distributions over words
            - Uses statistical inference to discover topic structures
            - No training data needed - unsupervised learning

            **Advantages:**
            - **Speed:** Processes entire datasets in milliseconds to seconds
            - **Interpretability:** Clear word-to-topic mappings
            - **No training required:** Works out of the box
            - **Scalability:** Handles millions of documents

            ---

            ### NMF Topic Modeling

            **What it is:** Non-negative Matrix Factorization - a linear algebra approach to topic discovery.

            **How it works:**
            - Converts documents to TF-IDF matrix
            - Factorizes the matrix into document-topic and topic-word matrices
            - All values are constrained to be non-negative (interpretable as strengths)
            - Trained in realtime on the current dataset

            **Advantages:**
            - **Speed:** Similar to LDA, processes datasets in 100-500ms
            - **Clarity:** Often produces more coherent topics than LDA
            - **Interpretability:** Non-negative weights are easier to understand
            - **Flexibility:** Works well with TF-IDF features
            """
        )
