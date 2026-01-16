import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
import glob
import concurrent.futures
import threading
import time
from pathlib import Path
from PIL import Image

# Hugging Face Transformers for ViTPose
from transformers import AutoImageProcessor, VitPoseForPoseEstimation

# --- CẤU HÌNH ---
ROOT = Path(__file__).resolve().parents[3]  # Repo root
INPUT_FOLDER = ROOT / 'data' / 'golfdb' / 'videos_hd' / 'fourth_part'  # Folder chứa video gốc (0.mp4, 1.mp4...)
OUTPUT_FOLDER = ROOT / 'data' / 'raw_golfdb_skeletons_npy' / 'fourth_part'  # Folder lưu kết quả
CONF_THRESHOLD = 0.5  # Độ tin cậy tối thiểu để chọn người
MAX_WORKERS = 7  # Số luồng chạy song song (Giảm bớt vì ViTPose tốn VRAM hơn YOLO)

# Đường dẫn model ViTPose (Hugging Face)
# Load from local directory for offline usage
MODELS_DIR = ROOT / 'apps' / 'models' # Updated to match where download_models.py put it
VITPOSE_MODEL_ID = str(MODELS_DIR / "vitpose-base-simple")
YOLO_DET_MODEL = str(MODELS_DIR / "yolov8n.pt")

# Verify models exist
if not os.path.exists(VITPOSE_MODEL_ID) or not os.path.exists(YOLO_DET_MODEL):
    raise RuntimeError(f"Offline models not found in {MODELS_DIR}. Please run 'apps/ml/src/download_models.py' first.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tạo folder output nếu chưa có
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# COCO 17 keypoint skeleton connections for visualization
# Each tuple represents a connection between two keypoint indices
COCO_SKELETON_CONNECTIONS = [
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),  # nose-eyes, eyes-ears
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),  # shoulders, shoulders-hips, hips
    # Left arm
    (5, 7), (7, 9),  # shoulder-elbow-wrist
    # Right arm
    (6, 8), (8, 10),  # shoulder-elbow-wrist
    # Left leg
    (11, 13), (13, 15),  # hip-knee-ankle
    # Right leg
    (12, 14), (14, 16),  # hip-knee-ankle
]


def visualize_pose_video(video_path, npy_path, output_path=None, 
                         conf_threshold=0.1, show_confidence=False):
    """
    Visualize pose estimation results by overlaying keypoints and skeleton on video.
    
    Args:
        video_path: Path to the original video file
        npy_path: Path to the .npy file containing skeleton data
        output_path: Path to save the output video (if None, will auto-generate)
        conf_threshold: Minimum confidence to draw a keypoint (default: 0.1)
        show_confidence: Whether to display confidence scores as text (default: False)
    
    Returns:
        Path to the output video file
    """
    # Load skeleton data
    skeleton_data = np.load(npy_path)  # Shape: (Frames, 17, 4)
    num_frames, num_keypoints, _ = skeleton_data.shape
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Auto-generate output path if not provided
    if output_path is None:
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir, f"{video_name}_pose_viz.mp4")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Visualizing pose for {os.path.basename(video_path)}...")
    print(f"  Skeleton frames: {num_frames}, Video frames: {total_video_frames}")
    print(f"  Output: {output_path}")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get skeleton data for this frame (if available)
        if frame_idx < num_frames:
            keypoints = skeleton_data[frame_idx]  # (17, 4) - [x, y, z, score]
            
            # Draw skeleton connections
            for connection in COCO_SKELETON_CONNECTIONS:
                pt1_idx, pt2_idx = connection
                x1, y1, _, score1 = keypoints[pt1_idx]
                x2, y2, _, score2 = keypoints[pt2_idx]
                
                # Only draw if both keypoints have sufficient confidence
                if score1 > conf_threshold and score2 > conf_threshold:
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    # Draw line with transparency effect (thicker, semi-transparent look)
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Draw keypoints
            for kpt_idx in range(num_keypoints):
                x, y, _, score = keypoints[kpt_idx]
                
                if score > conf_threshold:
                    center = (int(x), int(y))
                    
                    # Color based on keypoint type (different colors for left/right/center)
                    if kpt_idx in [1, 3, 5, 7, 9, 11, 13, 15]:  # Left side
                        color = (255, 0, 0)  # Blue
                    elif kpt_idx in [2, 4, 6, 8, 10, 12, 14, 16]:  # Right side
                        color = (0, 0, 255)  # Red
                    else:  # Center (nose)
                        color = (0, 255, 255)  # Yellow
                    
                    # Draw filled circle for keypoint
                    cv2.circle(frame, center, 5, color, -1, cv2.LINE_AA)
                    cv2.circle(frame, center, 5, (255, 255, 255), 1, cv2.LINE_AA)  # White border
                    
                    # Optionally show confidence score
                    if show_confidence:
                        cv2.putText(frame, f"{score:.2f}", 
                                  (center[0] + 8, center[1]), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                                  (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add frame number overlay
        cv2.putText(frame, f"Frame: {frame_idx + 1}/{total_video_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{total_video_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"✓ Visualization complete: {output_path}")
    return output_path


def process_video_to_npy(args):
    try:
        video_path, file_index, total_files = args

        # Lấy tên file (ví dụ: '0.mp4' -> '0')
        filename = os.path.basename(video_path)
        filename_no_ext = os.path.splitext(filename)[0]
        output_npy_path = os.path.join(OUTPUT_FOLDER, f"{filename_no_ext}.npy")
        
        # Debug
        print(f"[{file_index}/{total_files}] Processing {filename}...")

        # Resume: Nếu file npy đã có rồi thì bỏ qua không chạy lại
        if os.path.exists(output_npy_path):
            print(f"[Skip] {filename} đã có file npy.")
            return filename, "Skipped"

        # --- KHỞI TẠO LOCAL (Mỗi process 1 model riêng để tránh xung đột) ---
        # 1. Load YOLO (Detector)
        det_model = YOLO(YOLO_DET_MODEL)

        # 2. Load ViTPose (Pose Estimator)
        try:
            processor = AutoImageProcessor.from_pretrained(VITPOSE_MODEL_ID, use_fast=True)
            pose_model = VitPoseForPoseEstimation.from_pretrained(VITPOSE_MODEL_ID)
            pose_model.to(DEVICE)
            pose_model.eval()
        except Exception as e:
            print(f"ERROR loading model for {filename}: {e}")
            return filename, f"Error Loading ViTPose: {e}"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR opening video {filename}")
            return filename, "Error Open"

        # ViTPose (COCO) output 17 keypoints
        num_keypoints = 17 
        video_skeleton_data = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"[{filename}] Frame {frame_idx}")

            # Mặc định frame này là toàn số 0 (nếu không tìm thấy người)
            # Shape: (num_keypoints, 4) -> [x, y, z (2D => 0), score]
            current_frame_landmarks = np.zeros((num_keypoints, 4), dtype=np.float32)

            # 1. Detect Person with YOLO
            # classes=[0] -> chỉ detect class 'person'
            results = det_model(frame, classes=[0], conf=CONF_THRESHOLD, device=DEVICE, verbose=False)
            
            best_box = None
            best_conf = -1.0

            # Tìm người có confidence cao nhất trong frame
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                
                for box in boxes:
                    conf = float(box.conf.item())
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]

            # 2. Nếu tìm thấy người -> Crop & Run ViTPose
            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box)
                h, w, _ = frame.shape
                
                # Clamp crop coords
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 > x1 and y2 > y1:
                    # Crop image (OpenCV is BGR, ViTPose expects RGB usually via processor which handles PIL/numpy)
                    # Convert BGR to RGB for PIL
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # crop_img = frame_rgb[y1:y2, x1:x2]
                    pil_img = Image.fromarray(frame_rgb)

                    crop_w = x2 - x1
                    crop_h = y2 - y1
                    
                    # Prepare input for ViTPose
                    # Processor requires 'boxes' (list of list).
                    # We pass the GLOBAL box [x, y, w, h] to post_process_pose_estimation later,
                    # but for the FORWARD PASS, we are feeding the cropped image.
                    # The `boxes` argument in `processor` call is primarily for some internal checks or resizing 
                    # if the model supports it, but for standard ViTPose usage with `images` as crops,
                    # we often just feed the crop.
                    
                    # However, to let `post_process_pose_estimation` know the real position,
                    # we should prepare the GLOBAL box [x, y, w, h].
                    global_box = [[x1, y1, crop_w, crop_h]]

                    # For the forward pass, we can just use the standard pipeline.
                    inputs = processor(images=pil_img, boxes=[global_box], return_tensors="pt")
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = pose_model(**inputs)
                    
                    # Use the standard post_process_pose_estimation method
                    # This properly handles the VitPoseEstimatorOutput format
                    # IMPORTANT: We pass the GLOBAL box here so it knows where the keypoints are in the original image.
                    # The format expected is often [[ [x, y, w, h] ]] corresponding to the batch.
                    pose_results = processor.post_process_pose_estimation(
                        outputs, 
                        boxes=[global_box]
                    )
                    
                    # pose_results is a list[list[Dict]] format:
                    # - Outer list: one element per image in batch
                    # - Inner list: one element per detected person
                    # - Dict: contains 'keypoints' and 'scores' for that person
                    
                    if len(pose_results) > 0 and len(pose_results[0]) > 0:
                        # Get the first person's keypoints and scores
                        # pose_results[0] = first image (list of persons)
                        # pose_results[0][0] = first person (dict)
                        keypoints = pose_results[0][0]['keypoints']  # (17, 2)
                        scores = pose_results[0][0]['scores']  # (17,)
                        
                        # Convert to numpy if they're tensors
                        if torch.is_tensor(keypoints):
                            keypoints = keypoints.cpu().numpy()
                        if torch.is_tensor(scores):
                            scores = scores.cpu().numpy()
                        
                        # Since we passed the global box, `post_process_pose_estimation` should have 
                        # already adjusted the keypoints to the global coordinate system.
                        # No need for manual offset addition.
                        
                        # Fill the current frame landmarks
                        current_frame_landmarks[:, 0:2] = keypoints  # x, y
                        current_frame_landmarks[:, 2] = 0.0  # z (always 0 for 2D)
                        current_frame_landmarks[:, 3] = scores  # confidence scores

            # Lưu frame hiện tại vào list tổng
            video_skeleton_data.append(current_frame_landmarks)

        cap.release()

        # --- LƯU FILE .NPY ---
        if len(video_skeleton_data) > 0:
            # Final Shape: (Frames, num_keypoints, 4)
            final_array = np.array(video_skeleton_data, dtype=np.float32)
            np.save(output_npy_path, final_array)
            status = f"Done ({len(video_skeleton_data)} frames)"
            print(f"Saved to {output_npy_path}")
        else:
            status = "Empty Video (No frames)"

        print(f"[{file_index}/{total_files}] {filename:<10} | {status}")

        return filename, status

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR in {args[0]}: {e}")
        return args[0], "Failed"


def run_extraction():
    # Tìm tất cả file mp4 trong thư mục (bao gồm folder con)
    search_path = os.path.join(INPUT_FOLDER, "**/*.mp4")
    video_files = glob.glob(search_path, recursive=True)

    # Sắp xếp tên file (0.mp4, 1.mp4...) cho đẹp đội hình
    try:
        video_files.sort(
            key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f))))
        )
    except:
        video_files.sort()

    total_files = len(video_files)
    print(f"--> Tìm thấy {total_files} video.")
    print(f"--> Đang lưu kết quả vào: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"--> Config: YOLOv8 Detection + ViTPose ({VITPOSE_MODEL_ID})")
    print(f"--> Device: {DEVICE} | Workers: {MAX_WORKERS}")
    
    # Pre-load model in the main process to ensure everything is working
    print(f"--> Verifying local model at {VITPOSE_MODEL_ID}...")
    try:
        AutoImageProcessor.from_pretrained(VITPOSE_MODEL_ID, use_fast=True)
        VitPoseForPoseEstimation.from_pretrained(VITPOSE_MODEL_ID)
        print("--> Model is ready.")
    except Exception as e:
        print(f"--> Warning: Main process model check failed: {e}")
        # We continue, hoping workers might succeed individually or user sees error

    start_time = time.time()

    # Tạo danh sách tham số để chạy đa luồng
    args_list = [(f, i + 1, total_files) for i, f in enumerate(video_files)]

    # Bắt đầu chạy
    # Lưu ý: Với CUDA và PyTorch multiprocessing, cần cẩn thận start method.
    # ProcessPoolExecutor mặc định dùng spawn trên Windows nhưng đôi khi conflict CUDA context
    # Nếu gặp lỗi CUDA initialization, giảm max_workers=1 hoặc dùng ThreadPoolExecutor (nhưng GIL sẽ chặn).
    # Update: ViTPose + YOLO load song song 4 process có thể OOM trên 1 GPU thường.
    # Khuyến nghị: Chạy tuần tự hoặc max_workers thấp nếu GPU VRAM < 12GB.
    
    # Để an toàn nhất trên Windows + CUDA -> Dùng loop thường hoặc max_workers nhỏ.
    # Trong code này giữ ProcessPoolExecutor nhưng set start method nếu cần.
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
         executor.map(process_video_to_npy, args_list)

    duration = time.time() - start_time
    print(f"\n[HOÀN TẤT] Tổng thời gian: {duration:.2f} giây")


if __name__ == "__main__":
    # Fix multiprocessing start method for PyTorch/CUDA consistency on Windows
    # import multiprocessing
    # multiprocessing.set_start_method('spawn', force=True)
    run_extraction()

    # Visualize a test video
    # input_video_path = str(INPUT_FOLDER / '2.mp4')
    # npy_path = str(ROOT / 'data' / 'raw_TDTU_skeletons_npy' / '2.npy')
    # # npy_path = str(ROOT / 'data' / 'TDTU_yolopose_skeletons' / '2.npy')
    # output_dir = ROOT / 'data' / 'raw_TDTU_skeletons_viz'
    # os.makedirs(output_dir, exist_ok=True)
    # output_path = str(output_dir / '2.mp4')
    # visualize_pose_video(input_video_path, npy_path, output_path=output_path, conf_threshold=0.1, show_confidence=False)
