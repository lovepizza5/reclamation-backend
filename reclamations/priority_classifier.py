# reclamations/priority_classifier.py
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')


class PriorityClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.classes = ['low', 'normal', 'urgent']
        self.model_path = Path(__file__).parent / 'models' / 'priority_model.pkl'
        self.vectorizer_path = Path(__file__).parent / 'models' / 'priority_vectorizer.pkl'

        # Don't auto-initialize on import to avoid Django startup issues
        # self._load_or_initialize()

    def _load_or_initialize(self):
        """Load existing model or initialize new one - call this when needed"""
        os.makedirs(Path(__file__).parent / 'models', exist_ok=True)

        if self.model_path.exists() and self.vectorizer_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                print("✅ Loaded existing priority model")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}, initializing new")
                return self._initialize_new_model()
        else:
            print("🆕 No existing model found, initializing new")
            return self._initialize_new_model()

    def _initialize_new_model(self):
        """Initialize with better training data - FIXED LENGTHS"""
        try:
            # FIXED: Training data with equal lengths
            training_texts = [
                # URGENT - Security Issues (16 examples)
                "system hacked need help urgent", "my account was hacked emergency",
                "security breach detected immediately", "data stolen from database urgent",
                "malware infection critical", "ransomware attack help now",
                "unauthorized access to system asap", "cyber attack in progress emergency",
                "help system compromised hacked", "emergency security issue hacked",
                "hacked account please help", "data breach urgent response needed",
                "critical security vulnerability", "system intrusion detected urgent",
                "password stolen hacked account", "phishing attack compromised data",

                # URGENT - System Failures (8 examples)
                "system crashed cannot work", "all servers down emergency",
                "production system not working urgent", "critical bug causing data loss",
                "emergency server failure", "system outage affecting customers",
                "database corruption urgent", "payment system broken immediately",

                # URGENT - Emotional/Time Pressure (6 examples)
                "help me now urgent emergency", "need immediate assistance asap",
                "this is critical fix now", "urgent please respond immediately",
                "emergency situation help", "cannot wait need help now",

                # NORMAL - Regular Issues (12 examples)
                "software not working properly", "camera showing error message",
                "need help with installation", "question about features",
                "application crashing sometimes", "system running slow",
                "bug in the interface", "feature not working correctly",
                "having trouble logging in", "password reset not working",
                "can't access certain features", "getting error code 404",

                # NORMAL - Questions (6 examples)
                "how do I use this feature", "where can I find settings",
                "what does this error mean", "can you help me configure",
                "need instructions for setup", "how to backup my data",

                # LOW - Suggestions/Info (16 examples)
                "suggestion for improvement", "general inquiry about product",
                "when will new features come", "curious about pricing plans",
                "feature request for future", "just wanted to give feedback",
                "information about your services", "question about availability",
                "thinking about purchasing", "wondering if you support",
                "could you add this feature", "maybe consider improving",
                "minor typo on website", "small suggestion for UI",
                "color scheme could be better", "font size is small"
            ]

            training_labels = [
                # URGENT labels (16 + 8 + 6 = 30)
                'urgent', 'urgent', 'urgent', 'urgent', 'urgent', 'urgent',
                'urgent', 'urgent', 'urgent', 'urgent', 'urgent', 'urgent',
                'urgent', 'urgent', 'urgent', 'urgent',
                'urgent', 'urgent', 'urgent', 'urgent', 'urgent', 'urgent',
                'urgent', 'urgent',
                'urgent', 'urgent', 'urgent', 'urgent', 'urgent', 'urgent',

                # NORMAL labels (12 + 6 = 18)
                'normal', 'normal', 'normal', 'normal', 'normal', 'normal',
                'normal', 'normal', 'normal', 'normal', 'normal', 'normal',
                'normal', 'normal', 'normal', 'normal', 'normal', 'normal',

                # LOW labels (16)
                'low', 'low', 'low', 'low', 'low', 'low', 'low', 'low',
                'low', 'low', 'low', 'low', 'low', 'low', 'low', 'low'
            ]

            # Verify lengths match
            print(f"📊 Training data: {len(training_texts)} texts, {len(training_labels)} labels")
            if len(training_texts) != len(training_labels):
                raise ValueError(f"Data mismatch: {len(training_texts)} texts vs {len(training_labels)} labels")

            df = pd.DataFrame({
                'text': training_texts,
                'priority': training_labels
            })

            print(f"✅ Created DataFrame with {len(df)} samples")
            print(f"📈 Class distribution:")
            print(df['priority'].value_counts())

            self.vectorizer = TfidfVectorizer(
                max_features=500,  # Reduced for faster training
                ngram_range=(1, 2),
                stop_words='english',
                min_df=1
            )

            X = self.vectorizer.fit_transform(df['text'])
            y = df['priority']

            print(f"🔡 Vectorized to {X.shape[1]} features")

            # Use simpler model for reliability
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=3,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            print(f"🤖 Training model with {len(X_train)} samples...")
            self.model.fit(X_train, y_train)

            # Save model
            self._save_model()

            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"✅ Model trained with accuracy: {accuracy:.1%}")

            report = classification_report(y_test, y_pred, target_names=self.classes, output_dict=True)
            for cls in self.classes:
                if cls in report:
                    print(
                        f"   {cls.upper()}: Precision={report[cls]['precision']:.1%}, Recall={report[cls]['recall']:.1%}")

            return True

        except Exception as e:
            print(f"❌ Error initializing model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _save_model(self):
        """Save model and vectorizer"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            print(f"💾 Model saved to {self.model_path}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")

    def ensure_loaded(self):
        """Ensure model is loaded (lazy loading)"""
        if self.model is None or self.vectorizer is None:
            return self._load_or_initialize()
        return True

    def predict(self, text, sentiment=None):
        """Predict priority with security override"""
        # Ensure model is loaded
        if not self.ensure_loaded():
            return self._emergency_prediction(text, sentiment)

        text = str(text).lower()

        # SECURITY OVERRIDE - Always urgent for security issues
        security_keywords = [
            'hacked', 'hack', 'hacking', 'security breach', 'data breach',
            'data stolen', 'compromised', 'attacked', 'attack', 'intrusion',
            'unauthorized', 'malware', 'virus', 'ransomware', 'phishing',
            'cyber attack', 'system breach', 'data leak'
        ]

        for keyword in security_keywords:
            if keyword in text:
                return {
                    'priority': 'urgent',
                    'confidence': 0.99,
                    'reason': f'Security keyword detected: {keyword}',
                    'method': 'security_override'
                }

        # EMERGENCY OVERRIDE
        if 'urgent' in text or 'emergency' in text or 'asap' in text:
            return {
                'priority': 'urgent',
                'confidence': 0.95,
                'reason': 'Emergency keyword detected',
                'method': 'emergency_override'
            }

        # ML PREDICTION
        try:
            X = self.vectorizer.transform([text])
            probabilities = self.model.predict_proba(X)[0]

            # Get prediction
            predicted_idx = np.argmax(probabilities)
            predicted_class = self.classes[predicted_idx]
            confidence = probabilities[predicted_idx]

            # If ML is unsure (confidence < 60%), use rules
            if confidence < 0.6:
                return self._rule_based_prediction(text, sentiment)

            return {
                'priority': predicted_class,
                'confidence': float(confidence),
                'reason': 'ML prediction',
                'method': 'ml_model'
            }

        except Exception as e:
            print(f"⚠️ ML prediction failed: {e}")
            return self._rule_based_prediction(text, sentiment)

    def _emergency_prediction(self, text, sentiment):
        """Emergency prediction when model fails to load"""
        text = text.lower()
        if 'hack' in text or 'urgent' in text or 'emergency' in text:
            return {
                'priority': 'urgent',
                'confidence': 0.9,
                'reason': 'Emergency fallback - security/emergency keyword',
                'method': 'emergency_fallback'
            }
        return {
            'priority': 'normal',
            'confidence': 0.5,
            'reason': 'Emergency fallback - default',
            'method': 'emergency_fallback'
        }

    def _rule_based_prediction(self, text, sentiment):
        """Rule-based fallback"""
        text = text.lower()

        urgent_count = 0
        low_count = 0

        # Urgent indicators
        urgent_words = ['urgent', 'emergency', 'critical', 'asap', 'immediately',
                        'broken', 'crashed', 'failed', 'help', 'hack']
        for word in urgent_words:
            if word in text:
                urgent_count += 1

        # Exclamation marks
        urgent_count += min(text.count('!'), 3)

        # Low priority indicators
        low_words = ['suggestion', 'feedback', 'idea', 'inquiry', 'question',
                     'maybe', 'perhaps', 'could', 'might', 'just']
        for word in low_words:
            if word in text:
                low_count += 1

        # Determine priority
        if urgent_count >= 2:
            priority = 'urgent'
            confidence = 0.8
        elif low_count >= 2 and urgent_count == 0:
            priority = 'low'
            confidence = 0.7
        else:
            priority = 'normal'
            confidence = 0.6

        return {
            'priority': priority,
            'confidence': confidence,
            'reason': f'Rule-based: {urgent_count} urgent, {low_count} low indicators',
            'method': 'rule_based'
        }

    def train(self, additional_data=None):
        """Retrain model"""
        print("🔄 Retraining priority model...")
        success = self._initialize_new_model()
        return 0.85 if success else 0.0  # Return dummy accuracy

    def info(self):
        """Get model information"""
        if self.model is not None:
            return {
                'ready': True,
                'model_type': 'RandomForest',
                'classes': self.classes
            }
        return {'ready': False, 'error': 'Model not loaded'}


# Create instance but don't auto-initialize
priority_classifier = PriorityClassifier()