from django.db import models
from datetime import datetime

class StockData(models.Model):
    ticker = models.CharField(max_length=10,default=' ')
    date = models.DateTimeField(default=datetime.now,blank=True)
    close = models.DecimalField(max_digits=10, decimal_places=2,default=0.00)
    def __str__(self):
        return f"{self.ticker} data on {self.date.strftime('%Y-%m-%d')}"


class BacktestResult(models.Model):
    username = models.CharField(max_length=100, default='')
    ticker = models.CharField(max_length=10, default=' ')
    tweet_text = models.TextField(default=' ')
    created_at = models.DateTimeField(default=datetime.now, blank=True)
    atr_multiplier = models.FloatField(default=0.0)
    trailing_stop_multiplier = models.FloatField(default=0.0)
    atr_period = models.IntegerField(default=0)
    total_return = models.FloatField(default=0.0)
    portfolio_variance = models.FloatField(default=0.0)
    sharpe_ratio = models.FloatField(default=0.0)
    maximum_drawdown = models.FloatField(default=0.0)
    minutes_taken = models.IntegerField(default=0)
    sold_at_date = models.DateTimeField(default=datetime.now(),blank=True)
    score = models.FloatField(default=0.0)
    portfolio_chart_data = models.JSONField(default=dict)

    def __str__(self):
        return f"{self.ticker} on {self.created_at.strftime('%Y-%m-%d')}"
