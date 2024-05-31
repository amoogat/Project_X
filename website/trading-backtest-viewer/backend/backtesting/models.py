from django.db import models
from datetime import datetime

class Strategy(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(default='yo')

def get_default_strategy():
    # Ensure there is at least one strategy available, or create one if not
    strategy, created = Strategy.objects.get_or_create(name='Default Strategy', defaults={'description': 'Auto-created default strategy'})
    return strategy.id

class BacktestResult(models.Model):
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE, related_name='backtest_results', null=True)
    ticker = models.CharField(max_length=10, default=' ')
    created_at = models.DateTimeField(default=datetime.now, blank=True)
    atr_multiplier = models.FloatField(default=0.0)
    trailing_stop_multiplier = models.FloatField(default=0.0)
    atr_period = models.IntegerField(default=0)
    total_return = models.FloatField(default=0.0)
    portfolio_variance = models.FloatField(default=0.0)
    sharpe_ratio = models.FloatField(default=0.0)
    final_equity = models.FloatField(default=0.0)
    maximum_drawdown = models.FloatField(default=0.0)
    successful_trades = models.IntegerField(default=0)
    minutes_taken = models.IntegerField(default=0)
    score = models.FloatField(default=0.0) 

    def __str__(self):
        return f"{self.ticker} on {self.created_at.strftime('%Y-%m-%d')}"
