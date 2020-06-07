from django.urls import path


from . import views

app_name = 'sja'
urlpatterns = [
    path('', views.index, name='index'),
    
    path('object_detections/', views.DetectionsView.as_view(), name='detections'),
    # path('object_detections/<int:pk>/', views.DetailView.as_view(), name='detail'),
    path('object_detections/<int:pk>/', views.detail_detection, name='detail'),
    
    path('object_detections/<int:pk>/delete', views.delete_detection, name='delete_detection'),
    path('object_detections/create', views.create_detection, name='create_detection'),
    path('about', views.about_sja, name='about_sja')
]