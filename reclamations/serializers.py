# reclamations/serializers.py - UPDATED
from rest_framework import serializers
from .models import Reclamation


class ReclamationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Reclamation
        fields = "__all__"
        read_only_fields = ("id", "date", "length", "sentiment", "sentiment_confidence")  # Added priority

