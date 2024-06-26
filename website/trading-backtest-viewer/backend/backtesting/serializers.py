from rest_framework import serializers
from .models import BacktestResult, StockData

class StockDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = StockData
        fields = '__all__'

class BacktestResultSerializer(serializers.ModelSerializer):
    portfolio_chart_data = serializers.JSONField(required=False)

    class Meta:
        model = BacktestResult
        fields = '__all__'
