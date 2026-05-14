from model import get_model
import torch
import cv2
import numpy as np
from ultralytics.utils.nms import non_max_suppression
from ultralytics import YOLO
import torch.nn as nn
import argparse
from huggingface_hub import hf_hub_download
import os

CLASS_NAMES = {
    0: "gun",
    1: "knife",
    2: "weapon"
}

def video_to_tensor(video_path, target_fps=5, target_size: tuple[int, int]=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = []
    current_frame = 0.0
    step = original_fps/target_fps
    while current_frame < total_frames:
        frame_indices.append(int(current_frame))
        current_frame += step

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()

    frames_np = np.array(frames, dtype=np.uint8)
    tensor = torch.tensor(frames_np).permute(0, 3, 1, 2)
    tensor = tensor.float() /255.0
    print(f"Video tensor shape: {tensor.shape}  ({tensor.shape[0]} frames at {target_fps}fps)")
    return tensor, original_fps

def get_yolo_model(device):
    print("Loading yolo model...")
    model = YOLO('yolo11s.pt').model
    num_classes = 3
    detect_layer = model.model[23]
    model.model[-1].nc = 3
    model.nc = 3

    for i in range(3):
        old_conv = detect_layer.cv3[i][2]
        in_ch = old_conv.in_channels
        detect_layer.cv3[i][2] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
        nn.init.normal_(detect_layer.cv3[i][2].weight, std=0.01)
        nn.init.constant_(detect_layer.cv3[i][2].bias, 0)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.model[23].parameters():
        p.requires_grad = True

    weights_path = hf_hub_download(
        repo_id="b1n4r33/WeaponYolo",
        filename="yolo.pt"
    )

    yolo_state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(yolo_state)
    model.to(device).eval()
    print("Yolo loaded")
    return model

def preprocess_frame_for_yolo(frame, img_size=640):
    """frame: [C, H, W] float tensor 0-1"""
    frame = torch.nn.functional.interpolate(
        frame.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False
    )  # [1, C, img_size, img_size]
    return frame

def get_vivit_model(device):
    print("Loading vivit model...")
    model = get_model(device, False, None)
    model.config.id2label = {0: "non_violent", 1: "violent"}
    model.config.label2id = {"non_violent": 0, "violent": 1}

    weights_path = hf_hub_download(
        repo_id="b1n4r33/ViolenceVivit",
        filename="model.pt"
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.to(device).eval()
    print("Loaded vivit model!")
    return model

def parse_yolo_output(outputs, conf_threshold=0.3, iou_threshold=0.45, num_classes=3):
    """
    Raw YOLO output is a list of tensors from 3 detection heads.
    We need to decode boxes and filter by confidence.
    """

    # outputs from raw model is a tuple — grab the prediction tensor
    preds = outputs[0]  # [1, num_anchors, 4 + num_classes]

    results = non_max_suppression(preds, conf_threshold, iou_threshold, nc=num_classes)
    return results[0]  # boxes for first image: [N, 6] (x1,y1,x2,y2,conf,cls)

def infer_video(video_path, threshold=0.3, target_fps=5, window_size=10, stride=5, detection_save_dir="detections"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load both models
    vivit = get_vivit_model(device)
    yolo = get_yolo_model(device)

    # Full video as one tensor [T, C, H, W]
    video, original_fps = video_to_tensor(video_path, target_fps=target_fps)
    video = video.to(device)
    T = video.shape[0]

    print(f"\n{'Window':<10} {'Time':<12} {'P(violent)':<14}  {'WeaponConf': <10}   {"Detected":<20} {'Result'}")
    print("-" * 95)

    results = []
    with torch.no_grad():
        for i in range(0, T - window_size + 1, stride):
            window = video[i:i + window_size]

            logits = vivit(window.unsqueeze(0)).logits
            prob_violent = torch.softmax(logits, dim=1)[0][1].item()

            max_weapon_conf = 0.0
            detected_weapons = set()
            for j, frame in enumerate(window):  # frame: [C, H, W]
                frame_input = preprocess_frame_for_yolo(frame)
                yolo_out = yolo(frame_input)
                detections = parse_yolo_output(yolo_out)

                if detections is not None and len(detections):
                    max_weapon_conf = max(max_weapon_conf, detections[:, 4].max().item())
                    for det in detections:
                        cls_id = int(det[5].item())
                        weapon_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                        detected_weapons.add(weapon_name)
                    global_frame_idx = i+j
                    save_detection_frame(frame, detections, global_frame_idx, detection_save_dir)

            is_violent = prob_violent > threshold
            timestamp_sec = i / target_fps
            m = int(timestamp_sec // 60)
            s = timestamp_sec % 60

            weapon_text = ", ".join(sorted(detected_weapons)) if detected_weapons else "none"
            flag = "DANGER" if is_violent else "NORMAL"
            print(f"{i:<10} {m:02d}:{s:05.2f}     {prob_violent:<14.4f}  {max_weapon_conf:<10.4f}   {weapon_text:<20} {flag}")

            results.append({
                'window': i,
                'timestamp_sec': timestamp_sec,
                'prob_violent': prob_violent,
                'max_weapon_conf': max_weapon_conf,
                "detected_weapons": list(detected_weapons),
                'violent': is_violent,
            })

    # Print summary
    violent = [r for r in results if r['violent']]
    print(f"\n{'=' * 55}")
    print(f"Windows analyzed : {len(results)}")
    print(f"Violent windows  : {len(violent)}")
    print(f"% flagged        : {len(violent) / len(results) * 100:.1f}%")
    print(f"\nSaved all detection frames at: {detection_save_dir}")
    return results

def save_detection_frame(frame_tensor, detections, frame_idx, output_dir="detections"):
    """
    frame_tensor: [C,H,W] float tensor in range 0-1
    detections: [N,6] tensor (x1,y1,x2,y2,conf,cls)
    """

    os.makedirs(output_dir, exist_ok=True)

    # Convert tensor -> uint8 OpenCV image
    frame = frame_tensor.permute(1, 2, 0).cpu().numpy()
    frame = (frame * 255).astype(np.uint8)

    # RGB -> BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    h, w = frame.shape[:2]

    # Your YOLO input was resized to 640x640
    scale_x = w / 640
    scale_y = h / 640

    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()

        # Scale boxes back to original frame size
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        cls = int(cls)

        label = f"{CLASS_NAMES.get(cls, cls)} {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    save_path = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
    cv2.imwrite(save_path, frame)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    video_path = args.video
    infer_video(video_path)
