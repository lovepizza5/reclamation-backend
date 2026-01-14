# reclamations/urls.py
from django.urls import path
from .views import (
    ReclamationListView,
    ReclamationCreateView,
    ReclamationDetailView,
    stats,
    export_csv,
    train_model,  # Legacy sentiment training
    train_priority_model,  # New priority training
    analyze_sentiment,
    analyze_priority,
    model_info,
    get_categories,
    get_priority_stats,
    get_sentiment_stats,
    bulk_update_status,
    get_user_reclamations,
    get_reclamation_status,
)

urlpatterns = [
    # ===== PRIMARY API =====
    path("reclamations/", ReclamationListView.as_view(), name="reclamation_list"),
    path("reclamations/add/", ReclamationCreateView.as_view(), name="reclamation_add"),
    path("reclamations/<int:pk>/", ReclamationDetailView.as_view(), name="reclamation_detail"),
    path("reclamations/stats/", stats, name="reclamation_stats"),
    path("reclamations/export/", export_csv, name="reclamation_export"),

    # ===== STATISTICS =====
    path("reclamations/categories/", get_categories, name="get_categories"),
    path("reclamations/priority-stats/", get_priority_stats, name="priority_stats"),
    path("reclamations/sentiment-stats/", get_sentiment_stats, name="sentiment_stats"),

    # ===== BULK OPERATIONS =====
    path("reclamations/bulk-update-status/", bulk_update_status, name="bulk_update_status"),

    # ===== ML ENDPOINTS =====
    path("ml/train/", train_model, name="train_model"),  # Legacy sentiment training
    path("ml/train-priority/", train_priority_model, name="train_priority_model"),
    path("ml/analyze/", analyze_sentiment, name="analyze_sentiment"),
    path("ml/analyze-priority/", analyze_priority, name="analyze_priority"),
    path("ml/info/", model_info, name="model_info"),

    # ===== RASA INTEGRATION =====
    path("reclamations/add/", ReclamationCreateView.as_view(), name="reclamation_add"),

    # NEW: Tracking endpoints for Rasa
    path("reclamations/user/<str:username>/", get_user_reclamations, name="user_reclamations"),
    path("reclamations/status/<int:reclamation_id>/", get_reclamation_status, name="reclamation_status"),
    path("reclamations/check/<str:username>/<int:reclamation_id>/", get_reclamation_status, name="check_reclamation"),
]