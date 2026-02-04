from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('titles/', views.get_titles, name='get_titles'),
]
