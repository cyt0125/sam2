import os
import subprocess

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import cv2

from sam2.build_sam import build_sam2_video_predictor

def sys_check():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    return device


def init_predictor(model_size, device):
    checkpoint_map = {
        'large': 'sam2.1_hiera_large.pt',
        'base_plus': 'sam2.1_hiera_base_plus.pt',
        'small': 'sam2.1_hiera_small.pt',
        'tiny': 'sam2.1_hiera_tiny.pt'
    }

    cfg_map = {
        'large': 'sam2.1_hiera_l.yaml',
        'base_plus': 'sam2.1_hiera_b+.yaml',
        'small': 'sam2.1_hiera_s.yaml',
        'tiny': 'sam2.1_hiera_t.yaml'
    }

    if model_size not in checkpoint_map or model_size not in cfg_map:
        raise ValueError("Invalid model size. Please choose from 'large', 'base_plus', 'small', or 'tiny'.")

    checkpoint = checkpoint_map[model_size]
    cfg = cfg_map[model_size]

    print(f"Checkpoint: {checkpoint}, Config: {cfg}")

    sam2_checkpoint = f"checkpoints/{checkpoint}"
    model_cfg = cfg

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    return predictor

def init_state(video_dir, model_size):
    # initialize the state for inference
    predictor = init_predictor(model_size=model_size, device=sys_check())
    inference_state = predictor.init_state(video_path=video_dir)
    return predictor, inference_state

def reset_state(predictor,inference_state):
    predictor.reset_state(inference_state)
    return predictor

def add_point(
        predictor,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        box=None,
):
    points = np.array(points, dtype=np.float32)
    labels = np.array(labels, np.int32)
    # prompts[obj_id] = points, labels
    out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
        clear_old_points=clear_old_points,
        box=box,
    )

    return out_obj_ids, out_mask_logits

def predict_video(
        predictor,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    predict_result = predictor.propagate_in_video(
        inference_state,
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=max_frame_num_to_track,
        reverse=reverse)
    for out_frame_idx, out_obj_ids, out_mask_logits in predict_result:
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments

def predict_video_all(
        predictor,
        inference_state,
        start_frame_idx,
        max_frame_num_to_track=None,
        reverse=False,
    ):

    if start_frame_idx > 0:
        # 生成两个新字典
        video_segments_pre = predict_video(predictor, inference_state, start_frame_idx=start_frame_idx-1, reverse=True)
        video_segments_post = predict_video(predictor, inference_state, start_frame_idx=start_frame_idx)

        # 合并字典
        combined_dict = {**video_segments_pre, **video_segments_post}
        # 排序键并创建新字典
        sorted_keys = sorted(combined_dict.keys())
        sorted_dict = {key: combined_dict[key] for key in sorted_keys}
        video_segments_all = sorted_dict
    else:
        video_segments_all = predict_video(predictor, inference_state, start_frame_idx)

    return video_segments_all



# Helper Function to Apply Object Effects
def apply_object_effect(frame, mask, effect):
    result = frame.copy()

    if effect == "erase":
        # Replace object with white (erased)
        result[mask == 255] = [255, 255, 255]  # Set object area to white

    elif effect == "gradient":
        # Create a horizontal gradient across the width of the mask
        gradient = np.linspace(0, 255, frame.shape[1], dtype=np.uint8)  # Generate gradient over width
        gradient = np.tile(gradient, (frame.shape[0], 1))  # Repeat gradient across height
        gradient_3channel = np.dstack([gradient] * 3)  # Convert to 3-channel (R, G, B)

        # Apply the gradient to the object region
        result[mask == 255] = gradient_3channel[mask == 255]

    elif effect == "pixelate":
        # Pixelate the object by downscaling and then upscaling the object region
        small = cv2.resize(result, (10, 10))  # Downscale to 10x10
        pixelated_region = cv2.resize(small, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_NEAREST)
        result[mask == 255] = pixelated_region[mask == 255]

    elif effect == "overlay":
        # Apply a green overlay to the object
        overlay = np.full_like(result, [0, 255, 0])  # Green overlay
        result[mask == 255] = overlay[mask == 255]

    elif effect == "emoji":
        # Apply an emoji overlay to the object region (Make sure emoji.png exists)
        emoji = cv2.resize(cv2.imread("C:/Users/26087/Desktop/emoji.png"), (mask.shape[1], mask.shape[0]))
        result[mask == 255] = emoji[mask == 255]

    elif effect == "burst":
        result = draw_burst(result, mask)  # Use the 2D mask

    return result


# Helper Function to Apply Background Effects
def apply_background_effect(frame, mask, effect):
    result = frame.copy()

    # Invert the mask to get the background (where mask == 0)
    background_mask = (mask == 0)

    if effect == "erase":
        # Set the background to white (erased)
        result[background_mask] = [255, 255, 255]  # Set background to white

    elif effect == "gradient":
        # Create a horizontal gradient across the width of the image
        gradient = np.linspace(0, 255, frame.shape[1], dtype=np.uint8)  # Generate gradient over width
        gradient = np.tile(gradient, (frame.shape[0], 1))  # Repeat gradient across height
        gradient_3channel = np.dstack([gradient] * 3)  # Convert to 3-channel (R, G, B)

        # Apply the gradient to the background region
        result[background_mask] = gradient_3channel[background_mask]

    elif effect == "pixelate":
        # Pixelate the background by downscaling and then upscaling the background region
        small = cv2.resize(result, (10, 10))  # Downscale to 10x10
        pixelated_region = cv2.resize(small, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_NEAREST)
        result[background_mask] = pixelated_region[background_mask]

    elif effect == "desaturate":
        # Desaturate the background (convert to grayscale)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result[background_mask] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)[background_mask]

    elif effect == "blur":
        # Blur the background using Gaussian blur
        blurred_bg = cv2.GaussianBlur(result, (21, 21), 0)
        result[background_mask] = blurred_bg[background_mask]

    return result


# Function to draw rays around the object (burst effect)
def draw_burst(image, mask):
    result = image.copy()

    # Ensure the mask is single-channel before finding contours
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return the original image
    if len(contours) == 0:
        return result

    # Get the center of the object based on the mask contours
    contour = contours[0]  # Assuming the largest contour is the object
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return result

    # Calculate the object center from the moments
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    center = (center_x, center_y)

    # Get image dimensions
    height, width = image.shape[:2]

    # Define the number of rays and the angle step between them
    num_rays = 10  # You can adjust this for more or fewer rays
    angle_step = 360 / num_rays

    # Draw rays from the center point outward
    for angle in np.arange(0, 360, angle_step):
        radian = np.deg2rad(angle)
        end_x = int(center_x + width * np.cos(radian))
        end_y = int(center_y + width * np.sin(radian))

        # Draw the line from the center to the calculated endpoint (white color, thickness 2)
        cv2.line(result, (center_x, center_y), (end_x, end_y), (100, 0, 0), 10)

    return result


# Updated apply_masks_to_video function
def apply_masks_to_video(video_path, video_segments_all, output_path, effect, object_effect, background_effect):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the basic information of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # Use mp4 encoding
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Gets the mask of the current frame
        if frame_index in video_segments_all:
            masks = video_segments_all[frame_index]  # Gets all the masks for the current frame

            for obj_id, mask in masks.items():
                # The shape of the mask is (1, 720, 1280) and we need to convert it to (720, 1280)
                mask = mask[0]  # Remove the first dimension and become (720, 1280)

                # Convert the Boolean mask to a uint8 type
                mask = (mask * 255).astype(np.uint8)  # Convert True/False to 255/0

                if effect:
                    # Apply object and background effects
                    masked_frame = apply_background_effect(frame, mask, effect=background_effect)
                    masked_frame = apply_object_effect(masked_frame, mask, effect=object_effect)
                else:
                    # Create a color mask
                    colored_mask = np.zeros((height, width, 3),
                                            dtype=np.uint8)  # Create an image that is completely black
                    colored_mask[mask == 255] = [0, 255, 0]  # Set the mask area to green (BGR format)

                    # Applies a color mask to the current frame
                    masked_frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)  # Overlay the mask onto the frame

                # Write the processed frame to the output video
                out.write(masked_frame)
        else:
            # If there is no mask, write directly to the original frame
            out.write(frame)

        frame_index += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    output_avi_path = output_path

    base_name, ext = os.path.splitext(output_avi_path)

    output_mp4_path = base_name + ".mp4"

    output_avi_path = output_avi_path.replace('\\', '/')

    output_mp4_path = output_mp4_path.replace('\\', '/')

    print(output_mp4_path, output_avi_path)

    convert_avi_to_mp4(output_avi_path, output_mp4_path)

    # Optionally, you can remove the AVI file after conversion
    # if os.path.exists(output_avi_path):
    #     os.remove(output_avi_path)


def convert_avi_to_mp4(input_avi, output_mp4):
    print(input_avi, output_mp4)
    ffmpeg_command = [
        'ffmpeg', '-y',  # -y to overwrite the output file if it exists
        '-i', input_avi,  # Input AVI file
        '-vcodec', 'libx264',  # Use H.264 for video
        '-acodec', 'aac',  # Use AAC for audio
        '-strict', 'experimental',  # Use experimental AAC encoder if needed
        output_mp4  # Output MP4 file
    ]

    try:
        print(ffmpeg_command)
        # Run the FFmpeg command to convert AVI to MP4
        subprocess.run(ffmpeg_command, check=True)
        print(f"Conversion to MP4 successful! Saved as {output_mp4}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
