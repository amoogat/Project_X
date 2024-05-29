from rest_framework import serializers
from .models import TraderProfile

class TraderProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = TraderProfile
        fields = ['id', 'name', 'twitter_handle']
