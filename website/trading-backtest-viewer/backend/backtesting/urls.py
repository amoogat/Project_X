from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_file, name='upload_file'),
    path('results/', views.results_view, name='results_view'),
]
