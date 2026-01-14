# reclamations/management/commands/train_priority.py
from django.core.management.base import BaseCommand
from django.conf import settings
import sys

sys.path.append(str(settings.BASE_DIR))


class Command(BaseCommand):
    help = 'Train the priority classification ML model'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force retrain even if model exists',
        )
        parser.add_argument(
            '--test',
            action='store_true',
            help='Test the model after training',
        )

    def handle(self, *args, **options):
        self.stdout.write('🚀 Training Priority Classification Model...')

        try:
            # Import priority classifier
            from reclamations.priority_classifier import priority_classifier

            # Check if model already exists
            if priority_classifier.model_path.exists() and not options['force']:
                self.stdout.write('⚠️ Priority model already exists. Use --force to retrain.')
                priority_classifier.load()
            else:
                # Train model
                accuracy = priority_classifier.train()

                self.stdout.write(
                    self.style.SUCCESS(f'✅ Priority model trained successfully!')
                )
                self.stdout.write(f'📊 Accuracy: {accuracy:.1%}')

            # Test if requested
            if options['test']:
                self._test_model(priority_classifier)

            # Show model info
            self.stdout.write('\n📋 Priority Model Information:')
            info = priority_classifier.info()
            for key, value in info.items():
                self.stdout.write(f'  {key}: {value}')

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error: {e}')
            )
            import traceback
            traceback.print_exc()

    def _test_model(self, classifier):
        """Test the model with sample texts"""
        self.stdout.write('\n🧪 Testing priority classification:')

        test_cases = [
            ("URGENT! System crashed, need immediate help!", "urgent"),
            ("Emergency: Product not working, broken!", "urgent"),
            ("Question about delivery status", "normal"),
            ("Suggestion for improvement", "low"),
            ("I'm angry! This product is terrible!", "urgent"),
            ("Could you add this feature maybe?", "low"),
            ("Normal inquiry about service", "normal"),
            ("CRITICAL ISSUE! NEED HELP NOW!", "urgent"),
            ("Just curious about a feature", "low"),
            ("Feedback on recent update", "normal"),
        ]

        correct = 0
        total = len(test_cases)

        for text, expected in test_cases:
            result = classifier.predict(text)
            predicted = result['priority']
            confidence = result['confidence']

            if predicted == expected:
                correct += 1
                icon = '✅'
            else:
                icon = '❌'

            self.stdout.write(
                f'  {icon} "{text[:40]}..." → {predicted.upper()} '
                f'(expected: {expected.upper()}) '
                f'[{confidence:.1%}]'
            )

        accuracy = correct / total
        self.stdout.write(f'\n📈 Test Accuracy: {accuracy:.1%} ({correct}/{total})')