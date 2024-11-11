from django.urls import path
from .views import upload_video, init_state_view, add_point_view

urlpatterns = [
    path('', upload_video, name='upload_video'),
    path('init_state/', init_state_view, name='init_state'),
    path('add_point/', add_point_view, name='add_point'),
]