from django.contrib import admin
from .models import Strategy, BacktestResult

admin.site.register(Strategy)
admin.site.register(BacktestResult)
