# reclamations/views.py
from rest_framework import generics, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAdminUser
from django.http import HttpResponse
from django.db.models import Count, Q
from django.utils.dateparse import parse_date
from django.db.models.functions import TruncDate
import csv
import datetime

from .models import Reclamation
from .serializers import ReclamationSerializer
from .utils import word_frequencies, get_sentiment_analysis
from reclamations.priority_classifier import priority_classifier

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator


# ===== RECLAMATION VIEWS =====

class ReclamationListView(generics.ListAPIView):
    serializer_class = ReclamationSerializer

    def get_queryset(self):
        qs = Reclamation.objects.all().order_by("-date")

        # Search query
        q = self.request.GET.get("q")
        if q:
            qs = qs.filter(Q(message__icontains=q) | Q(username__icontains=q))

        # Status filter
        status_q = self.request.GET.get("status")
        if status_q:
            qs = qs.filter(status=status_q)

        # Priority filter
        priority_q = self.request.GET.get("priority")
        if priority_q:
            qs = qs.filter(priority=priority_q)

        # Sentiment filter
        sentiment_q = self.request.GET.get("sentiment")
        if sentiment_q:
            qs = qs.filter(sentiment=sentiment_q)

        # Date range filters
        start = self.request.GET.get("start")
        end = self.request.GET.get("end")
        if start:
            qs = qs.filter(date__date__gte=parse_date(start))
        if end:
            qs = qs.filter(date__date__lte=parse_date(end))

        # Category filter
        category_q = self.request.GET.get("category")
        if category_q:
            qs = qs.filter(category=category_q)

        return qs


# views.py - Update the ReclamationCreateView
@method_decorator(csrf_exempt, name='dispatch')
class ReclamationCreateView(generics.CreateAPIView):
    queryset = Reclamation.objects.all()
    serializer_class = ReclamationSerializer
    authentication_classes = []  # Disable authentication for testing
    permission_classes = []  # Allow all

    def create(self, request, *args, **kwargs):
        print("📨 Received POST request from Rasa")
        print(f"📦 Request data: {request.data}")
        print(f"📦 Headers: {dict(request.headers)}")

        # Accept ANY data for now
        data = request.data.copy()

        # Ensure required fields
        if not data.get('username'):
            data['username'] = 'Anonymous_Rasa_User'

        if not data.get('message'):
            return Response(
                {'error': 'Message is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Set default category if not provided
        if not data.get('category'):
            data['category'] = 'Rasa Bot'

        if not data.get('location'):
            data['location'] = 'Rasa Chat Interface'

        print(f"✅ Processed data: {data}")

        # Create serializer with data
        serializer = self.get_serializer(data=data)

        if serializer.is_valid():
            print("✅ Serializer is valid")

            try:
                # Save the instance
                instance = serializer.save()
                print(f"✅ Instance saved with ID: {instance.id}")
                print(f"✅ Sentiment: {instance.sentiment}")
                print(f"✅ Priority: {instance.priority}")

                # Build response
                response_data = serializer.data
                response_data['analysis_summary'] = {
                    'sentiment': instance.sentiment,
                    'sentiment_confidence': instance.sentiment_confidence,
                    'priority': instance.priority,
                    'message': 'Auto-analyzed successfully'
                }

                return Response(response_data, status=status.HTTP_201_CREATED)

            except Exception as e:
                print(f"❌ Error saving instance: {e}")
                return Response(
                    {'error': f'Failed to save: {str(e)}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        else:
            print(f"❌ Serializer errors: {serializer.errors}")
            return Response(
                {'error': 'Invalid data', 'details': serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )

class ReclamationDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Reclamation.objects.all()
    serializer_class = ReclamationSerializer

    def update(self, request, *args, **kwargs):
        instance = self.get_object()
        data = request.data.copy()

        # If message is being updated, re-analyze sentiment and priority
        if 'message' in data and data['message'] != instance.message:
            # The model's save() method will handle re-analysis
            pass

        serializer = self.get_serializer(instance, data=data, partial=True)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        return Response(serializer.data)


# ===== STATISTICS ENDPOINT =====

# In views.py, update the stats function to ensure it returns all required data:

@api_view(["GET"])
def stats(request):
    total = Reclamation.objects.count()
    today = Reclamation.objects.filter(date__date=datetime.date.today()).count()
    week_ago = datetime.datetime.now() - datetime.timedelta(days=7)
    last7 = Reclamation.objects.filter(date__gte=week_ago).count()
    unresolved = Reclamation.objects.filter(status="pending").count()

    per_day_qs = (Reclamation.objects.annotate(day=TruncDate("date"))
                  .values("day")
                  .annotate(count=Count("id"))
                  .order_by("day"))
    per_day = [{"day": str(p["day"]), "count": p["count"]} for p in per_day_qs]

    top_users_qs = Reclamation.objects.values("username").annotate(count=Count("id")).order_by("-count")[:10]
    top_users = [{"username": u["username"], "count": u["count"]} for u in top_users_qs]

    categories_qs = Reclamation.objects.values("category").annotate(count=Count("id")).order_by("-count")
    categories = [{"category": c["category"] or "Uncategorized", "count": c["count"]} for c in categories_qs]

    texts = Reclamation.objects.values_list("message", flat=True)
    freqs = word_frequencies(texts, max_words=40)

    return Response({
        "total": total,
        "today": today,
        "last7": last7,
        "unresolved": unresolved,
        "per_day": per_day,
        "top_users": top_users,
        "categories": categories,
        "word_freqs": freqs,
    })

@api_view(["GET"])
def export_csv(request):
    """Export reclamations to CSV with all filters"""
    qs = Reclamation.objects.all().order_by("-date")

    # Apply filters
    q = request.GET.get("q")
    if q:
        qs = qs.filter(Q(message__icontains=q) | Q(username__icontains=q))

    start = request.GET.get("start")
    end = request.GET.get("end")
    if start:
        qs = qs.filter(date__date__gte=parse_date(start))
    if end:
        qs = qs.filter(date__date__lte=parse_date(end))

    status_q = request.GET.get("status")
    if status_q:
        qs = qs.filter(status=status_q)

    priority_q = request.GET.get("priority")
    if priority_q:
        qs = qs.filter(priority=priority_q)

    sentiment_q = request.GET.get("sentiment")
    if sentiment_q:
        qs = qs.filter(sentiment=sentiment_q)

    # Create CSV response
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = "attachment; filename=reclamations_export.csv"

    writer = csv.writer(response)
    writer.writerow([
        "id", "username", "message", "date", "category",
        "sentiment", "sentiment_confidence", "priority",
        "status", "admin_notes", "location", "length"
    ])

    for r in qs:
        writer.writerow([
            r.id, r.username, r.message, r.date.isoformat(),
            r.category, r.sentiment, r.sentiment_confidence,
            r.priority, r.status, r.admin_notes, r.location, r.length
        ])

    return response


# ===== ML ENDPOINTS =====

@api_view(['POST'])
@permission_classes([IsAdminUser])
def train_model(request):
    """Train or retrain the sentiment model (legacy - keep for compatibility)"""
    try:
        # Try to import and train sentiment model
        from .ml_utils import sentiment_analyzer
        accuracy = sentiment_analyzer.train()
        return Response({
            'success': True,
            'message': f'Sentiment model trained successfully with accuracy: {accuracy:.1%}',
            'accuracy': accuracy
        })
    except Exception as e:
        return Response({
            'success': False,
            'message': f'Error training model: {str(e)}'
        }, status=500)


@api_view(['POST'])
@permission_classes([IsAdminUser])
def train_priority_model(request):
    """Train or retrain the priority classification model"""
    try:
        accuracy = priority_classifier.train()
        return Response({
            'success': True,
            'message': f'Priority model trained successfully with accuracy: {accuracy:.1%}',
            'accuracy': accuracy
        })
    except Exception as e:
        return Response({
            'success': False,
            'message': f'Error training priority model: {str(e)}'
        }, status=500)


@api_view(['POST'])
def analyze_sentiment(request):
    """Analyze sentiment for given text"""
    text = request.data.get('text', '')

    if not text:
        return Response({
            'success': False,
            'message': 'No text provided'
        }, status=400)

    try:
        result = get_sentiment_analysis(text)
        return Response({
            'success': True,
            'text': text,
            'analysis': result
        })
    except Exception as e:
        return Response({
            'success': False,
            'message': f'Error analyzing sentiment: {str(e)}'
        }, status=500)


@api_view(['POST'])
def analyze_priority(request):
    from reclamations.priority_classifier import priority_classifier
    """Analyze priority for given text"""
    text = request.data.get('text', '')
    sentiment = request.data.get('sentiment', None)

    if not text:
        return Response({
            'success': False,
            'message': 'No text provided'
        }, status=400)

    try:
        result = priority_classifier.predict(text, sentiment)
        return Response({
            'success': True,
            'text': text,
            'sentiment': sentiment,
            'analysis': result
        })
    except Exception as e:
        return Response({
            'success': False,
            'message': f'Error analyzing priority: {str(e)}'
        }, status=500)


@api_view(['GET'])
def model_info(request):
    """Get information about ML models"""
    try:
        # Get priority model info
        priority_info = priority_classifier.info()

        # Get sentiment model info (if available)
        try:
            from .ml_utils import sentiment_analyzer
            sentiment_info = sentiment_analyzer.info()
        except:
            sentiment_info = {'ready': False, 'error': 'Sentiment model not available'}

        return Response({
            'priority_model': priority_info,
            'sentiment_model': sentiment_info,
            'models_ready': priority_info.get('ready', False) and sentiment_info.get('ready', False)
        })
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=500)


# ===== ADDITIONAL UTILITY ENDPOINTS =====

@api_view(['GET'])
def get_categories(request):
    """Get unique categories from existing reclamations"""
    categories = Reclamation.objects.values_list('category', flat=True).distinct()
    categories = [cat for cat in categories if cat]  # Remove empty values
    return Response(categories)


@api_view(['GET'])
def get_priority_stats(request):
    """Get priority statistics for dashboard"""
    urgent = Reclamation.objects.filter(priority='urgent').count()
    normal = Reclamation.objects.filter(priority='normal').count()
    low = Reclamation.objects.filter(priority='low').count()
    total = urgent + normal + low

    return Response({
        'urgent': urgent,
        'normal': normal,
        'low': low,
        'total': total,
        'urgent_percentage': round(urgent / total * 100, 1) if total > 0 else 0,
        'normal_percentage': round(normal / total * 100, 1) if total > 0 else 0,
        'low_percentage': round(low / total * 100, 1) if total > 0 else 0,
    })


@api_view(['GET'])
def get_sentiment_stats(request):
    """Get sentiment statistics for dashboard"""
    positive = Reclamation.objects.filter(sentiment='positive').count()
    negative = Reclamation.objects.filter(sentiment='negative').count()
    neutral = Reclamation.objects.filter(sentiment='neutral').count()
    total = positive + negative + neutral

    return Response({
        'positive': positive,
        'negative': negative,
        'neutral': neutral,
        'total': total,
        'positive_percentage': round(positive / total * 100, 1) if total > 0 else 0,
        'negative_percentage': round(negative / total * 100, 1) if total > 0 else 0,
        'neutral_percentage': round(neutral / total * 100, 1) if total > 0 else 0,
    })


@api_view(['POST'])
@permission_classes([IsAdminUser])
def bulk_update_status(request):
    """Bulk update reclamation statuses"""
    try:
        data = request.data
        reclamation_ids = data.get('ids', [])
        new_status = data.get('status')

        if not reclamation_ids or not new_status:
            return Response({'error': 'Missing ids or status'}, status=400)

        # Update reclamations
        updated = Reclamation.objects.filter(id__in=reclamation_ids).update(status=new_status)

        return Response({
            'success': True,
            'message': f'Updated {updated} reclamations to status: {new_status}',
            'updated_count': updated
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)


# reclamations/views.py - ADD THESE NEW VIEW FUNCTIONS

@api_view(['GET'])
def get_user_reclamations(request, username):
    """Get all reclamations for a specific user"""
    reclamations = Reclamation.objects.filter(username=username).order_by('-date')

    if not reclamations.exists():
        return Response({
            'success': False,
            'message': f'No reclamations found for user: {username}'
        }, status=404)

    serializer = ReclamationSerializer(reclamations, many=True)

    # Summary statistics
    total = reclamations.count()
    pending = reclamations.filter(status='pending').count()
    resolved = reclamations.filter(status='resolved').count()
    urgent = reclamations.filter(priority='urgent').count()

    return Response({
        'success': True,
        'username': username,
        'total_reclamations': total,
        'pending': pending,
        'resolved': resolved,
        'urgent': urgent,
        'reclamations': serializer.data
    })


@api_view(['GET'])
def get_reclamation_status(request, reclamation_id=None, username=None):
    """Get status of a specific reclamation"""
    try:
        if username and reclamation_id:
            # Check if this reclamation belongs to the user
            reclamation = Reclamation.objects.get(id=reclamation_id, username=username)
        elif reclamation_id:
            # Just get by ID (admin or if username not provided)
            reclamation = Reclamation.objects.get(id=reclamation_id)
        else:
            return Response({
                'success': False,
                'message': 'Please provide reclamation ID'
            }, status=400)

        serializer = ReclamationSerializer(reclamation)

        # Format dates nicely
        from django.utils import timezone
        now = timezone.now()
        created_date = reclamation.date
        days_ago = (now - created_date).days

        status_info = {
            'id': reclamation.id,
            'username': reclamation.username,
            'message_preview': reclamation.message[:100] + ('...' if len(reclamation.message) > 100 else ''),
            'status': reclamation.status,
            'priority': reclamation.priority,
            'sentiment': reclamation.sentiment,
            'created_date': created_date.strftime('%Y-%m-%d %H:%M'),
            'days_ago': days_ago,
            'category': reclamation.category,
            'admin_notes': reclamation.admin_notes if reclamation.admin_notes else "No notes yet",
            'location': reclamation.location
        }

        return Response({
            'success': True,
            'reclamation': status_info
        })

    except Reclamation.DoesNotExist:
        return Response({
            'success': False,
            'message': f'Reclamation not found'
        }, status=404)
    except Exception as e:
        return Response({
            'success': False,
            'message': f'Error: {str(e)}'
        }, status=500)