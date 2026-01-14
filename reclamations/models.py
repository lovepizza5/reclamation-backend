# reclamations/models.py - UPDATED WITH FIXES
from django.db import models
class Reclamation(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("reviewed", "Reviewed"),
        ("resolved", "Resolved"),
    ]

    PRIORITY_CHOICES = [
        ("low", "Low"),
        ("normal", "Normal"),
        ("urgent", "Urgent"),
    ]

    username = models.CharField(max_length=150, blank=True, default="Anonymous")
    # ✅ ADD THESE TWO LINES - Email field
    email = models.EmailField(max_length=254, blank=True, null=True)

    # ✅ ADD THIS LINE - Phone field
    phone = models.CharField(max_length=20, blank=True, null=True)

    message = models.TextField()
    date = models.DateTimeField(auto_now_add=True)
    category = models.CharField(max_length=100, blank=True)
    sentiment = models.CharField(max_length=20, blank=True)
    sentiment_confidence = models.FloatField(default=0.0)
    priority = models.CharField(max_length=20, choices=PRIORITY_CHOICES, default="")  # CHANGED: Empty default
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    admin_notes = models.TextField(blank=True)
    location = models.CharField(max_length=150, blank=True)
    length = models.PositiveIntegerField(default=0)

    def analyze_sentiment(self):
        """Analyze sentiment using reliable ML model"""
        print(f"\n🎯 ANALYZING SENTIMENT for: '{self.message[:50]}...'")

        if not self.message or not isinstance(self.message, str):
            print("⚠️ No message to analyze")
            self.sentiment = "neutral"
            self.sentiment_confidence = 0.5
            return

        try:
            from reclamations.reliable_sentiment import reliable_analyzer
            result = reliable_analyzer.analyze(self.message)

            self.sentiment = result['sentiment']
            self.sentiment_confidence = result['confidence']

            print(f"✅ Sentiment (ML): {self.sentiment} ({self.sentiment_confidence:.0%})")

            # Force negative if security + emotion
            text_lower = self.message.lower()
            security_emotional = ('hacked' in text_lower or 'security' in text_lower) and \
                                 ('furious' in text_lower or 'angry' in text_lower or 'disappointed' in text_lower)

            if security_emotional and self.sentiment != 'negative':
                print(f"🚨 SECURITY + EMOTION DETECTED - FORCING NEGATIVE")
                self.sentiment = 'negative'
                self.sentiment_confidence = 0.98

        except Exception as e:
            print(f"❌ Error in sentiment analysis: {e}")
            # Simple emergency fallback
            text = self.message.lower()
            if 'hacked' in text and ('furious' in text or 'angry' in text or 'disappointed' in text):
                self.sentiment = "negative"
                self.sentiment_confidence = 0.95
                print(f"⚠️ Emergency fallback: NEGATIVE (security + emotion)")
            elif 'hacked' in text or 'furious' in text or 'angry' in text:
                self.sentiment = "negative"
                self.sentiment_confidence = 0.85
                print(f"⚠️ Emergency fallback: NEGATIVE")
            elif 'love' in text or 'amazing' in text or 'great' in text:
                self.sentiment = "positive"
                self.sentiment_confidence = 0.80
                print(f"⚠️ Emergency fallback: POSITIVE")
            else:
                self.sentiment = "neutral"
                self.sentiment_confidence = 0.60
                print(f"⚠️ Emergency fallback: NEUTRAL")

    def _fallback_sentiment_analysis(self):
        """Enhanced fallback sentiment analysis"""
        text = self.message.lower()

        # Enhanced keyword matching
        positive_keywords = [
            'love', 'amazing', 'excellent', 'great', 'perfect', 'fantastic',
            'best', 'good', 'working', 'fixed', 'solved', 'helpful', 'thanks',
            'thank you', 'appreciate', 'awesome', 'wonderful'
        ]

        negative_keywords = [
            'hate', 'terrible', 'worst', 'horrible', 'awful', 'bad', 'broken',
            'hacked', 'crash', 'crashed', 'urgent', 'emergency', 'critical',
            'failed', 'error', 'problem', 'issue', 'angry', 'furious', 'mad',
            'complaint', 'disappointed', 'frustrated', 'stuck', 'not working',
            'useless', 'waste', 'refund', 'cancel', 'stop'
        ]

        delivery_keywords = [
            'arrived', 'delivered', 'received', 'package', 'order', 'on time',
            'shipped', 'delivery', 'arrival'
        ]

        positive_count = sum(1 for word in positive_keywords if word in text)
        negative_count = sum(1 for word in negative_keywords if word in text)
        delivery_count = sum(1 for word in delivery_keywords if word in text)

        # Check for emotional indicators
        exclamation_count = text.count('!')
        question_count = text.count('?')

        # Determine sentiment
        if negative_count > 2 or (negative_count > 0 and exclamation_count > 1):
            self.sentiment = "negative"
            self.sentiment_confidence = 0.85 + (min(negative_count, 5) * 0.03)
        elif positive_count > 2:
            self.sentiment = "positive"
            self.sentiment_confidence = 0.80 + (min(positive_count, 5) * 0.04)
        elif delivery_count > 0:
            self.sentiment = "neutral"
            self.sentiment_confidence = 0.75
        elif exclamation_count > 2:
            self.sentiment = "negative"
            self.sentiment_confidence = 0.70
        elif question_count > 2:
            self.sentiment = "neutral"
            self.sentiment_confidence = 0.65
        else:
            self.sentiment = "neutral"
            self.sentiment_confidence = 0.60

        print(f"⚠️ Used enhanced fallback sentiment: {self.sentiment} ({self.sentiment_confidence:.0%})")

    def analyze_priority(self):
        """Analyze priority using enhanced ML classifier"""
        print(f"\n🚨 ANALYZING PRIORITY for: '{self.message[:50]}...'")

        if not self.message or not isinstance(self.message, str):
            print("⚠️ No message to analyze")
            self.priority = "normal"
            return

        try:
            from reclamations.priority_classifier import priority_classifier
            result = priority_classifier.predict(self.message, self.sentiment)

            self.priority = result['priority']

            print(f"📊 Priority prediction: {result['priority'].upper()} ({result['confidence']:.0%})")
            print(f"   Method: {result['method']}")
            if 'reason' in result:
                print(f"   Reason: {result['reason']}")

            # Log override if security detected
            if result.get('method') == 'security_override':
                print(f"🚨 SECURITY OVERRIDE APPLIED!")

        except Exception as e:
            print(f"❌ Error in priority analysis: {e}")
            # Simple fallback
            text = self.message.lower()
            if 'hack' in text or 'urgent' in text or 'emergency' in text:
                self.priority = "urgent"
                print(f"⚠️ Fallback: URGENT (keywords detected)")
            else:
                self.priority = "normal"
                print(f"⚠️ Fallback: NORMAL (default)")

    def _enhanced_priority_analysis(self):
        """Enhanced rule-based priority analysis with better keyword detection"""
        text = self.message.lower()

        # EXPANDED URGENT KEYWORDS
        urgent_keywords = [
            'urgent', 'emergency', 'asap', 'immediately', 'now', 'critical',
            'broken', 'not working', 'crashed', 'failed', 'error', 'problem',
            'help', 'support', 'assistance', 'issue', 'terrible', 'worst',
            'refund', 'cancel', 'stop', 'angry', 'furious', 'complaint',
            # ADDED CRITICAL SECURITY KEYWORDS
            'hacked', 'hack', 'hacking', 'security', 'breach', 'attacked',
            'attack', 'danger', 'dangerous', 'leak', 'data leak', 'compromised',
            'stolen', 'theft', 'malware', 'virus', 'ransomware', 'phishing',
            'exploit', 'vulnerability', 'intrusion', 'unauthorized', 'invasion',
            'spyware', 'trojan', 'worm', 'cyber', 'hijack', 'takeover',
            # ADDED EMOTIONAL URGENCY
            'panic', 'desperate', 'please help', 'need help now', 'as soon as possible',
            'right away', 'right now', 'this instant', 'cannot wait',
            # ADDED SYSTEM FAILURE
            'down', 'offline', 'unresponsive', 'freeze', 'frozen', 'locked',
            'blocked', 'denied', 'rejected', 'corrupted', 'damaged', 'destroyed'
        ]

        # SUPER URGENT KEYWORDS (automatic urgent if found)
        super_urgent = [
            'hacked', 'security breach', 'data stolen', 'emergency', 'critical',
            'system down', 'cannot access', 'locked out', 'data loss'
        ]

        # LOW PRIORITY KEYWORDS
        low_keywords = [
            'suggestion', 'feedback', 'idea', 'improvement', 'feature',
            'question', 'inquiry', 'curious', 'wonder', 'think',
            'maybe', 'perhaps', 'possibly', 'could', 'might',
            'when', 'where', 'how', 'what', 'why', 'information',
            'general', 'just asking', 'out of curiosity', 'wanted to know'
        ]

        # Check for SUPER URGENT first (instant urgent)
        for keyword in super_urgent:
            if keyword in text:
                self.priority = "urgent"
                print(f"🚨 SUPER URGENT detected: '{keyword}'")
                return

        # Count urgent indicators
        urgent_count = 0
        for keyword in urgent_keywords:
            if keyword in text:
                urgent_count += 1
                print(f"   ⚠️ Found urgent keyword: '{keyword}'")

        # Count exclamation marks (adds urgency)
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            urgent_count += exclamation_count
            print(f"   ‼️ Found {exclamation_count} exclamation marks")

        # Count question marks with urgency
        if '?' in text and ('urgent' in text or 'help' in text or 'emergency' in text):
            urgent_count += 2

        # Count low priority indicators
        low_count = 0
        for keyword in low_keywords:
            if keyword in text:
                low_count += 1

        # Check sentiment influence
        if self.sentiment == "negative":
            urgent_count += 1
            print(f"   😠 Negative sentiment adds urgency")

        # DETERMINE PRIORITY
        print(f"   📊 Urgent indicators: {urgent_count}, Low indicators: {low_count}")

        if urgent_count >= 3:
            self.priority = "urgent"
            print(f"   🔥 HIGH URGENCY DETECTED ({urgent_count} indicators)")
        elif urgent_count >= 1 and low_count == 0:
            self.priority = "normal"
            print(f"   ⚠️ Some urgency detected ({urgent_count} indicators)")
        elif low_count >= 2 and urgent_count == 0:
            self.priority = "low"
            print(f"   👍 Low priority ({low_count} indicators)")
        else:
            self.priority = "normal"
            print(f"   📄 Normal priority (default)")

        print(f"   ✅ Final priority: {self.priority.upper()}")

    def save(self, *args, **kwargs):
        # Calculate message length
        self.length = len(self.message or "")

        print(f"\n💾 SAVING Reclamation...")
        print(f"   Message: '{self.message[:50]}...'")
        # ✅ ADD THESE TWO LINES
        if self.email:
            print(f"   Email: {self.email}")
        if self.phone:
            print(f"   Phone: {self.phone}")
        print(f"   Current priority: {self.priority}")



        # FIXED: Always analyze sentiment for new or updated messages
        print("   🔄 Analyzing sentiment...")
        self.analyze_sentiment()

        # FIXED: Always analyze priority (force re-analysis)
        print("   🔄 Analyzing priority...")
        self.analyze_priority()

        print(f"   📝 Final values - Sentiment: {self.sentiment}, Priority: {self.priority}")

        # Call parent save
        super().save(*args, **kwargs)

        print(f"   ✅ Saved successfully! ID: {self.id}\n")

    def __str__(self):
        return f"{self.username} - {self.priority.upper()} - {self.sentiment}"

    class Meta:
        ordering = ['-date']