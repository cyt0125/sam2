from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from djangoProject import settings
import os

from videoProcessing.utils import init_state, add_point


def upload_video(request):
    if request.method == 'POST' and request.FILES['video']:
        video_file = request.FILES['video']
        fs = FileSystemStorage()
        video_path = fs.save(video_file.name, video_file)
        video_url = fs.url(video_path)
        video_dir = os.path.dirname(video_path)  # Get the directory of the uploaded video
        return JsonResponse({'video_url': video_url, 'video_dir': video_dir})

    return render(request, 'video_processing/mainPage.html', {
        'MEDIA_URL': settings.MEDIA_URL,
    })

import json

def init_state_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        video_filename = data.get('video_filename')
        model_size = data.get('model_size')

        video_dir = os.path.join(settings.MEDIA_ROOT, video_filename)

        # 调用 init_state 函数
        predictor, inference_state = init_state(video_dir, model_size)

        return predictor, inference_state, JsonResponse({'status': 'success', 'message': 'State initialized successfully.'})

    return JsonResponse({'status': 'error', 'message': 'Invalid request.'})

import json
from django.http import JsonResponse

def add_point_view(request, predictor, inference_state):
    if request.method == 'POST':
        data = json.loads(request.body)
        frame_index = data.get('frame_index')
        x = data.get('x')
        y = data.get('y')
        obj_id = 1

        # 在这里调用 add_point 函数
        add_point(predictor, inference_state, frame_index, obj_id, points=[x, y], labels=[1])

        return JsonResponse({'status': 'success', 'message': 'Point added successfully.'})

    return JsonResponse({'status': 'error', 'message': 'Invalid request.'})

