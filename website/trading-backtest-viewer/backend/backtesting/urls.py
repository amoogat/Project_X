from django.urls import path
from . import views
from django.shortcuts import render

urlpatterns = [
    path('upload/', views.upload_file, name='upload_file'),  # Correct view function
    path('results/', views.results_view, name='results_view'),
    path('upload-form/', views.upload_form_view, name='upload_form'),  # Correct view function
]
