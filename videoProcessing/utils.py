import os
import subprocess

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import sam2
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

def apply_masks_to_video(video_path, video_segments_all, output_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建 VideoWriter 对象以保存输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 获取当前帧的掩码
        if frame_index in video_segments_all:
            masks = video_segments_all[frame_index]  # 获取当前帧的所有掩码

            for obj_id, mask in masks.items():
                # mask 的形状是 (1, 720, 1280)，我们需要将其转换为 (720, 1280)
                mask = mask[0]  # 去掉第一个维度，变为 (720, 1280)

                # 将布尔掩码转换为 uint8 类型
                mask = (mask * 255).astype(np.uint8)  # 将 True/False 转换为 255/0

                colored_mask = np.zeros((height, width, 3), dtype=np.uint8)  # 创建一个全黑的图像
                colored_mask[mask == 255] = [0, 255, 255]

                masked_frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)  # 叠加掩码到帧上

                out.write(masked_frame)
        else:
            out.write(frame)

        frame_index += 1

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()