from django.urls import path

from . import views

app_name = 'sja'
urlpatterns = [
    path('object_detections/', views.IndexView.as_view(), name='index'),
    path('object_detections/<int:pk>/', views.DetailView.as_view(), name='detail'),
    path('object_detections/<int:pk>/delete', views.delete_detection, name='delete_detection'),
    path('object_detections/create', views.createDetection, name='createDetection'),
]