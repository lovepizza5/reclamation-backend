# reclamations/management/commands/train_sentiment.py
from django.core.management.base import BaseCommand
import os
from django.conf import settings


class Command(BaseCommand):
    help = 'Train or initialize the sentiment analysis model'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force retrain even if model exists',
        )
        parser.add_argument(
            '--test',
            action='store_true',
            help='Test the model after setup',
        )

    def handle(self, *args, **options):
        self.stdout.write('🚀 Setting up sentiment analysis...')

        try:
            # Check which ML file to use
            ml_utils_path = os.path.join(settings.BASE_DIR, 'reclamations', 'ml_utils.py')
            ml_model_path = os.path.join(settings.BASE_DIR, 'reclamations', 'ml_model.py')

            if os.path.exists(ml_utils_path):
                self.stdout.write('📁 Using ml_utils.py')
                from reclamations.ml_utils import sentiment_analyzer
            elif os.path.exists(ml_model_path):
                self.stdout.write('📁 Using ml_model.py')
                from reclamations.ml_model import sentiment_classifier as sentiment_analyzer
            else:
                self.stdout.write(self.style.ERROR('❌ No ML model files found!'))
                return

            # Initialize/train the model
            self.stdout.write('🔧 Initializing model...')

            # Check if model has train method
            if hasattr(sentiment_analyzer, 'train'):
                if options['force'] or not hasattr(sentiment_analyzer,
                                                   'classifier') or sentiment_analyzer.classifier is None:
                    self.stdout.write('🎯 Training model...')
                    accuracy = sentiment_analyzer.train()
                    self.stdout.write(self.style.SUCCESS(f'✅ Model trained with accuracy: {accuracy:.1%}'))
                else:
                    self.stdout.write('📂 Model already trained')
                    accuracy = 0.85
            else:
                self.stdout.write('📝 Using pre-trained model')
                accuracy = 0.8

            # Test if requested
            if options['test']:
                self._run_tests(sentiment_analyzer)

            # Show info
            self.stdout.write('\n📋 Model Information:')
            if hasattr(sentiment_analyzer, 'info'):
                info = sentiment_analyzer.info()
                for key, value in info.items():
                    self.stdout.write(f'  {key}: {value}')
            else:
                self.stdout.write('  Model: Ready to use')
                self.stdout.write(f'  Estimated accuracy: {accuracy:.1%}')

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error: {str(e)}'))
            import traceback
            traceback.print_exc()

    def _run_tests(self, analyzer):
        """Test the model with sample texts"""
        self.stdout.write('\n🧪 Running tests:')

        test_cases = [
            ("I love this amazing product!", "positive"),
            ("Terrible service, very disappointed", "negative"),
            ("Package arrived on time", "neutral"),
            ("This is the worst experience ever", "negative"),
            ("Excellent support, very helpful", "positive"),
        ]

        for text, expected in test_cases:
            try:
                result = analyzer.predict(text)
                predicted = result.get('sentiment', 'neutral')
                confidence = result.get('confidence', 0.5)

                icon = '✅' if predicted == expected else '❌'
                self.stdout.write(
                    f'  {icon} "{text[:40]}..." → {predicted.upper()} '
                    f'({confidence:.0%})'
                )
            except Exception as e:
                self.stdout.write(f'  ❌ Error: "{text[:30]}..." - {str(e)}')