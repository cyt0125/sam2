from datetime import datetime

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
from django.conf import settings
from .utils import *  # Import all functions from utils.py

# Global variables to store predictor and inference_state
global_predictor = None
global_inference_state = None


def index(request):
    return render(request, 'video_processing/mainPage.html')


@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']

        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        # Save uploaded video
        video_path = os.path.join(upload_dir, video_file.name)
        with open(video_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)

        # Initialize model
        global global_predictor, global_inference_state
        global_predictor, global_inference_state = init_state(video_path, model_size='tiny')

        return JsonResponse({
            'status': 'success',
            'video_path': video_path,
            'message': 'Video uploaded successfully'
        })

    return JsonResponse({'status': 'error', 'message': 'Invalid request'})


@csrf_exempt
def add_tracking_point(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        frame_idx = int(data.get('frame_idx'))
        obj_id = int(data.get('obj_id'))
        points = data.get('points')  # This will now be [x, y] coordinates
        labels = data.get('labels')

        # Convert points to numpy array with correct shape
        points_np = np.array([points], dtype=np.float32)  # Shape: (1, 2)
        labels_np = np.array(labels, dtype=np.int32)

        global global_predictor, global_inference_state

        # Add point and get masks
        out_obj_ids, out_mask_logits = add_point(
            global_predictor,
            global_inference_state,
            frame_idx,
            obj_id,
            points=points_np,
            labels=labels_np
        )

        # Predict video segments
        video_segments_all = predict_video_all(
            global_predictor,
            global_inference_state,
            frame_idx
        )

        # Create output directory if it doesn't exist
        output_dir = os.path.join(settings.MEDIA_ROOT, 'output')
        os.makedirs(output_dir, exist_ok=True)

        # Generate output path
        video_name = f"tracked_video_{obj_id}.avi"
        output_path = os.path.join(output_dir, video_name)

        # Apply masks to video
        apply_masks_to_video(
            data.get('video_path'),
            video_segments_all,
            output_path,
            effect=None,
            object_effect=None,
            background_effect=None
        )
        video_name = f"tracked_video_{obj_id}.mp4"
        output_path = os.path.join(output_dir, video_name)
        relative_output_path = os.path.relpath(output_path, settings.MEDIA_ROOT)
        video_url = os.path.join(settings.MEDIA_URL, relative_output_path)

        return JsonResponse({
            'status': 'success',
            'output_video': video_url,
            'timestamp': str(datetime.now().timestamp())  # Add timestamp to prevent caching
        })
        response['Content-Disposition'] = 'inline'
        return response


    return JsonResponse({'status': 'error', 'message': 'Invalid request'})





@csrf_exempt
def apply_effects(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        object_effect = data.get('object_effect')
        background_effect = data.get('background_effect')
        video_path = data.get('video_path')

        # Create output directory if it doesn't exist
        output_dir = os.path.join(settings.MEDIA_ROOT, 'output')
        os.makedirs(output_dir, exist_ok=True)

        # Generate output path for effect video
        video_name = f"effect_video_{object_effect}_{background_effect}.avi"
        output_path = os.path.join(output_dir, video_name)

        global global_predictor, global_inference_state

        # Get the latest video segments
        frame_idx = int(data.get('frame_idx', 0))
        video_segments_all = predict_video_all(
            global_predictor,
            global_inference_state,
            frame_idx
        )

        # Apply effects to video
        apply_masks_to_video(
            video_path,
            video_segments_all,
            output_path,
            effect=True,
            object_effect=object_effect,
            background_effect=background_effect
        )

        video_name = f"effect_video_{object_effect}_{background_effect}.mp4"
        return JsonResponse({
            'status': 'success',
            'output_video': f'/media/output/{video_name}'
        })

    return JsonResponse({'status': 'error', 'message': 'Invalid request'})


@csrf_exempt
def get_frame_info(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        video_path = data.get('video_path')

        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        return JsonResponse({
            'total_frames': total_frames,
            'fps': fps
        })

    return JsonResponse({'status': 'error', 'message': 'Invalid request'})


@csrf_exempt
def serve_video(request, path):
    video_path = os.path.join(settings.MEDIA_ROOT, path)
    content_type = 'video/mp4'  # or determine dynamically based on file extension

    response = FileResponse(open(video_path, 'rb'), content_type=content_type)
    response['Accept-Ranges'] = 'bytes'
    return response