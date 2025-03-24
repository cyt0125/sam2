# System Design

- [Task 2.1: Specify Requirements](#task-2-1-specify-requirements)
- [Task 2.2: Design System Architecture](#task-2-2-design-system-architecture)

## Task 2.1: Specify Requirements

### Hardware Requirements:
- **Computer**: A computer with a multi-core processor (Intel i5 or higher) and at least 16GB of RAM.
- **GPU**: A dedicated GPU (NVIDIA GTX 3060 or higher) for accelerating deep learning model inference.
- **Storage**: At least 50GB of free disk space for storing models and datasets.

### Software Requirements:
- **Python**: Version 3.10 or higher.
- **OpenCV**: For image processing and computer vision tasks.
- **Segment Anything Model (SAM2)**: For object segmentation.
- **PyTorch**: For loading and running the SAM2 model.
- **NumPy**: For array and matrix operations.
- **Matplotlib**: For visualizing results (optional).


## Task 2.2: Design System Architecture

### System Architecture Design

#### 1. **Overview**
The system architecture for the Real-time Object Segmentation and Replacement System Based on SAM includes the following components:
- **Input Module**: Captures real-time video input from a camera.
- **Preprocessing Module**: Converts the video frames to the format required by the SAM model.
- **Segmentation Module**: Uses the SAM model to segment objects in the video frames.
- **Processing Module**: Applies simple processing to the segmented objects.
- **Replacement Module**: Replaces the segmented objects with user-specified objects.
- **Output Module**: Displays the processed video frames in real-time.


#### 2. **Component Descriptions**

The `add_point` function is a key component of the Video Creative System. It allows for the addition of new points in a video frame for object tracking and manipulation. 

```python
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
    prompts[obj_id] = points, labels
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
```

The `predict_video` function is another essential component of the Video Creative System. It handles the propagation of predictions throughout a video and collects the segmentation results for each frame. 

```python
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
```

The `predict_video_all` function is a vital component of the Video Creative System. It combines the predictions from different segments of the video, ensuring a comprehensive and continuous result. 

```python
def predict_video_all(
        predictor,
        inference_state,
        start_frame_idx,
        max_frame_num_to_track=None,
        reverse=False,
    ):

    if start_frame_idx > 0:
        # Generate two new dictionaries
        video_segments_pre = predict_video(predictor, inference_state, start_frame_idx=start_frame_idx-1, reverse=True)
        video_segments_post = predict_video(predictor, inference_state, start_frame_idx=start_frame_idx)

        # Merge dictionaries
        combined_dict = {**video_segments_pre, **video_segments_post}
        # Sort keys and create a new dictionary
        sorted_keys = sorted(combined_dict.keys())
        sorted_dict = {key: combined_dict[key] for key in sorted_keys}
        video_segments_all = sorted_dict
    else:
        video_segments_all = predict_video(predictor, inference_state, start_frame_idx)

    return video_segments_all
```

The `apply_object_effect` function applies various effects to the object in the video frame. 

```python
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
```

The `apply_background_effect` function applies various effects to the background of the video frame. 

```python
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
```

The `apply_masks_to_video` function processes the video frames by applying the specified object and background effects and writes the results to an output video file. 

```python
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
```



