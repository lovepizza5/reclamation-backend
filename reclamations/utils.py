# reclamations/utils.py - FINAL VERSION
import re
from collections import Counter
from .ml_utils import sentiment_analyzer  # ✅ Use the new unified analyzer

# Keep POS/NEG words for emergency fallback
POS = {"good", "great", "happy", "satisfied", "excellent", "fast", "ok", "fine",
       "perfect", "awesome", "love", "best", "wonderful", "amazing"}
NEG = {"bad", "slow", "late", "angry", "terrible", "unhappy", "disappointed",
       "problem", "issue", "fail", "failed", "worst", "poor", "horrible", "broken"}


def estimate_sentiment(text: str) -> str:
    """Estimate sentiment using unified ML model"""
    if not text or not isinstance(text, str):
        return "neutral"

    try:
        result = sentiment_analyzer.predict(text)
        return result['sentiment']
    except Exception as e:
        print(f"ML sentiment error: {e}")
        return rule_based_sentiment(text)


def rule_based_sentiment(text: str) -> str:
    """Rule-based sentiment analysis as fallback"""
    t = (text or "").lower()
    score = 0
    words = re.findall(r"\w+", t)

    for w in words:
        if w in POS: score += 1
        if w in NEG: score -= 1

    if '!' in t:
        if any(word in POS for word in words):
            score += 2
        elif any(word in NEG for word in words):
            score -= 2

    if score > 1: return "positive"
    if score < -1: return "negative"
    return "neutral"


def word_frequencies(texts, max_words=40):
    """Extract word frequencies from texts"""
    words = []
    for t in texts:
        if not t:
            continue
        t = re.sub(r"http\S+", "", t)
        t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
        words.extend([w.lower() for w in t.split() if len(w) > 2])
    return Counter(words).most_common(max_words)


def get_sentiment_analysis(text):
    """Get detailed sentiment analysis using unified model"""
    try:
        return sentiment_analyzer.predict(text)
    except Exception as e:
        print(f"Error in detailed analysis: {e}")
        sentiment = rule_based_sentiment(text)
        confidence = 0.6 if sentiment != 'neutral' else 0.8
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'negative': 0.3 if sentiment == 'negative' else 0.1,
                'neutral': 0.3 if sentiment == 'neutral' else 0.1,
                'positive': 0.3 if sentiment == 'positive' else 0.1
            }
        }