from django.urls import path
from . import views
from django.shortcuts import render

urlpatterns = [
    path('upload/', views.upload_file, name='upload_file'),
    path('results/', views.results_view, name='results_view'),
    path('upload-form/', lambda request: render(request, 'upload.html'), name='upload_form'),
]
