from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_video, name='upload_video'),
    path('add-tracking/', views.add_tracking_point, name='add_tracking'),
    path('apply-effects/', views.apply_effects, name='apply_effects'),
    path('get-frame-info/', views.get_frame_info, name='get_frame_info'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)