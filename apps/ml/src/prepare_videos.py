import cv2
import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CẤU HÌNH ---
ROOT = Path(__file__).resolve().parents[3]  # Repo root
INPUT_FOLDERS = {
    ROOT / 'data' / 'TDTU-Golf-Pose-v1' / 'videos': ROOT / 'data' / 'ready_for_mediapipe' / 'TDTU'
}
YOLO_MODEL = ROOT / 'apps' / 'ml' / 'src' / 'yolov8n.pt'  # Model nano siêu nhanh
TARGET_HEIGHT = 720  # Resize về 720p (giữ aspect ratio)

# Load model
model = YOLO(YOLO_MODEL)

def apply_clahe(image):
    # 1. Chuyển sang không gian LAB để đo độ sáng (Kênh L)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 2. Tính độ sáng trung bình
    avg_brightness = np.mean(l)
    
    # 3. Ra quyết định dựa trên độ sáng
    if avg_brightness < 90: 
        # Rất tối (thường là BTC indoor) -> Kéo mạnh
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
    elif avg_brightness < 140:
        # Hơi tối -> Kéo nhẹ để tránh sạn
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        l = clahe.apply(l)
        
    else:
        # Đã sáng (thường là GolfDB outdoor) -> Giữ nguyên
        return image 
        
    # Gộp lại và trả về
    limg = cv2.merge((l, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def process_single_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    
    # Lấy thông số video gốc
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Resize về chiều cao chuẩn 720p
        h, w = frame.shape[:2]
        scale = TARGET_HEIGHT / h
        new_w = int(w * scale)
        frame_resized = cv2.resize(frame, (new_w, TARGET_HEIGHT))
        
        # 2. Apply CLAHE (Cải thiện ánh sáng)
        frame_enhanced = apply_clahe(frame_resized)
        
        frames.append(frame_enhanced)
        
    cap.release()
    
    if len(frames) == 0: return

    # Save video mới
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    
    for f in frames:
        out.write(f)
    out.release()
    print(f" -> Saved: {save_path}")

def run_pipeline():
    for input_folder, output_folder in INPUT_FOLDERS.items():
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)

        if not input_folder.exists():
            print(f"Skip, missing input: {input_folder}")
            continue
        
        output_folder.mkdir(parents=True, exist_ok=True)
            
        files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
        print(f"Processing folder: {input_folder} ({len(files)} videos)")
        
        tasks = []
        for vid in files:
            in_path = input_folder / vid
            out_path = output_folder / vid
            if out_path.exists(): continue  # Skip nếu đã làm
            tasks.append((in_path, out_path))
        
        if not tasks:
            print("Nothing to process.")
            continue
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_single_video, str(in_path), str(out_path)) for in_path, out_path in tasks]
            for future in as_completed(futures):
                future.result()

if __name__ == "__main__":
    run_pipeline()