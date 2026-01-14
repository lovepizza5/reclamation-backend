# reclamations/admin.py - UPDATED
from django.contrib import admin
from .models import Reclamation


@admin.register(Reclamation)
class ReclamationAdmin(admin.ModelAdmin):
    list_display = ("username", "date", "category", "sentiment", "priority", "status", "length")  # Added priority
    list_filter = ("status", "priority", "category", "sentiment", "date")  # Added priority
    search_fields = ("username", "message", "admin_notes")
    readonly_fields = ("date", "length", "sentiment", "sentiment_confidence", "priority")

    # Add priority color coding in admin
    def get_priority_color(self, obj):
        colors = {
            'urgent': 'red',
            'normal': 'orange',
            'low': 'green'
        }
        return colors.get(obj.priority, 'black')

    get_priority_color.short_description = "Priority"