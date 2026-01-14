# reclamations/reliable_sentiment.py
import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
import os


class ReliableSentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.classes = ['negative', 'neutral', 'positive']
        self.model_path = Path(__file__).parent / 'sentiment_model.pkl'
        self.vectorizer_path = Path(__file__).parent / 'sentiment_vectorizer.pkl'
        self._load_or_train()

    def _load_or_train(self):
        """Load existing model or train new one"""
        if self.model_path.exists() and self.vectorizer_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                print("✅ Loaded trained sentiment model")
            except:
                print("⚠️ Failed to load model, training new one...")
                self._train_model()
        else:
            print("🆕 No sentiment model found, training new one...")
            self._train_model()

    def _train_model(self):
        """Train sentiment model with comprehensive dataset"""
        # Comprehensive training data
        training_data = {
            'text': [
                # NEGATIVE EXAMPLES - Security & Anger (30 examples)
                "my software got hacked i'm very disappointed from your system security it's furious",
                "system hacked need help urgent this is terrible",
                "security breach detected immediately i'm angry",
                "data stolen from database urgent worst experience ever",
                "malware infection critical hate this so much",
                "ransomware attack help now your system is garbage",
                "unauthorized access to system asap terrible security",
                "cyber attack in progress emergency furious right now",
                "help system compromised hacked awful service",
                "emergency security issue hacked disappointed and angry",
                "hacked account please help very unhappy",
                "data breach urgent response needed horrible situation",
                "critical security vulnerability pissed off",
                "system intrusion detected urgent worst day",
                "password stolen hacked account terrible",
                "phishing attack compromised data angry customer",
                "your security is awful i want refund",
                "system crashed lost all data furious",
                "privacy violated data leaked very angry",
                "hacked and no support terrible company",
                "security failure disappointed customer",
                "breached again this is ridiculous",
                "attack on system need help now angry",
                "data loss due to hack unacceptable",
                "system down emergency frustrated",
                "can't believe you got hacked worst",
                "security issue makes me furious",
                "hacked account refund now angry",
                "privacy breach very disappointed",
                "system failure terrible experience",

                # NEGATIVE EXAMPLES - General Complaints (20 examples)
                "software not working properly very frustrating",
                "camera showing error message annoying",
                "application crashing sometimes hate it",
                "system running slow this is bad",
                "bug in the interface frustrating",
                "feature not working correctly unhappy",
                "having trouble logging in irritating",
                "password reset not working annoying",
                "can't access certain features disappointed",
                "getting error code 404 frustrating",
                "service is down again terrible",
                "slow response time unacceptable",
                "bad customer service unhappy",
                "poor quality disappointed",
                "not worth the money waste",
                "worst purchase ever regret",
                "broken on arrival angry",
                "doesn't work as advertised misleading",
                "technical issues frustrating",
                "constant problems annoying",

                # POSITIVE EXAMPLES (25 examples)
                "love your software amazing work",
                "system working perfectly great job",
                "excellent security features love it",
                "best security system ever fantastic",
                "perfect solution thank you so much",
                "fantastic support team awesome",
                "great product very happy",
                "working smoothly excellent",
                "fixed my issue quickly helpful",
                "thanks for the help appreciate",
                "thank you for great service",
                "appreciate your quick response",
                "awesome product love it",
                "wonderful experience best ever",
                "brilliant solution perfect",
                "outstanding service amazing",
                "superb quality fantastic",
                "marvelous work excellent",
                "splendid job well done",
                "terrific system love it",
                "satisfied customer happy",
                "pleased with results great",
                "delighted with service awesome",
                "joy to use perfect",
                "ecstatic about features best",

                # NEUTRAL EXAMPLES (25 examples)
                "software needs update version 2.0",
                "camera installed yesterday working",
                "application running on windows 10",
                "system using version 3.5.1",
                "bug reported to development team",
                "feature requested for next release",
                "login issue with chrome browser",
                "password reset link received",
                "accessing features from dashboard",
                "error code 404 on page load",
                "service maintenance scheduled",
                "response time measured at 200ms",
                "customer service contacted today",
                "quality check completed",
                "purchase made last week",
                "product delivered yesterday",
                "arrived in packaging intact",
                "advertisement seen online",
                "technical details provided",
                "problem reported to support",
                "information requested about features",
                "question about configuration",
                "inquiry about pricing plans",
                "request for documentation",
                "need instructions for setup"
            ],
            'sentiment': [
                # Negative sentiments (30 + 20 = 50)
                'negative', 'negative', 'negative', 'negative', 'negative',
                'negative', 'negative', 'negative', 'negative', 'negative',
                'negative', 'negative', 'negative', 'negative', 'negative',
                'negative', 'negative', 'negative', 'negative', 'negative',
                'negative', 'negative', 'negative', 'negative', 'negative',
                'negative', 'negative', 'negative', 'negative', 'negative',
                'negative', 'negative', 'negative', 'negative', 'negative',
                'negative', 'negative', 'negative', 'negative', 'negative',
                'negative', 'negative', 'negative', 'negative', 'negative',
                'negative', 'negative', 'negative', 'negative', 'negative',

                # Positive sentiments (25)
                'positive', 'positive', 'positive', 'positive', 'positive',
                'positive', 'positive', 'positive', 'positive', 'positive',
                'positive', 'positive', 'positive', 'positive', 'positive',
                'positive', 'positive', 'positive', 'positive', 'positive',
                'positive', 'positive', 'positive', 'positive', 'positive',

                # Neutral sentiments (25)
                'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
                'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
                'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
                'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
                'neutral', 'neutral', 'neutral', 'neutral', 'neutral'
            ]
        }

        # Convert to arrays
        texts = training_data['text']
        sentiments = training_data['sentiment']

        # Clean texts
        cleaned_texts = [self._clean_text(t) for t in texts]

        # Create vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),  # Capture phrases
            stop_words='english',
            min_df=1,
            max_df=0.9
        )

        # Vectorize
        X = self.vectorizer.fit_transform(cleaned_texts)

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X, sentiments)

        # Save model
        self._save_model()

        # Test on training data
        predictions = self.model.predict(X)
        accuracy = np.mean(predictions == sentiments)
        print(f"✅ Sentiment model trained with accuracy: {accuracy:.1%}")

        # Test specific cases
        test_cases = [
            ("my software got hacked i'm very disappointed from your system security it's furious", "negative"),
            ("love your security system amazing", "positive"),
            ("need help with installation", "neutral"),
            ("hacked account emergency help", "negative"),
            ("great job on the update", "positive")
        ]

        print("\n🧪 Model test results:")
        for text, expected in test_cases:
            result = self.analyze(text)
            correct = result['sentiment'] == expected
            symbol = "✅" if correct else "❌"
            print(f"{symbol} '{text[:30]}...' -> {result['sentiment']} (expected: {expected})")

    def _clean_text(self, text):
        """Clean text for analysis"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        return text.strip()

    def _save_model(self):
        """Save model and vectorizer"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"💾 Sentiment model saved")

    def analyze(self, text):
        """Analyze sentiment of text"""
        # Clean text
        cleaned = self._clean_text(text)

        # Vectorize
        X = self.vectorizer.transform([cleaned])

        # Predict
        probabilities = self.model.predict_proba(X)[0]
        predicted_idx = np.argmax(probabilities)
        predicted = self.classes[predicted_idx]
        confidence = probabilities[predicted_idx]

        # Adjust confidence for short texts
        if len(cleaned.split()) < 3:
            confidence *= 0.8

        # Override rules for strong indicators
        text_lower = text.lower()

        # Strong negative overrides
        strong_negative = ['hacked', 'furious', 'terrible', 'worst', 'hate', 'angry', 'disappointed']
        if any(word in text_lower for word in strong_negative) and 'hacked' in text_lower:
            if predicted != 'negative':
                predicted = 'negative'
                confidence = max(confidence, 0.95)

        # Strong positive overrides
        strong_positive = ['love', 'amazing', 'best', 'perfect', 'fantastic', 'excellent']
        if any(word in text_lower for word in strong_positive):
            if predicted != 'positive':
                predicted = 'positive'
                confidence = max(confidence, 0.90)

        return {
            'sentiment': predicted,
            'confidence': float(confidence),
            'text_length': len(text_lower.split())
        }


# Create global instance
reliable_analyzer = ReliableSentimentAnalyzer()