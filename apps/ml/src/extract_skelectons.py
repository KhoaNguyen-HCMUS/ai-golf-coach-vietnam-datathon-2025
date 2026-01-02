import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
import glob
import concurrent.futures
import threading
import time

# --- CẤU HÌNH ---
INPUT_FOLDER = "./data"  # Folder chứa video gốc (0.mp4, 1.mp4...)
OUTPUT_FOLDER = "./data/TDTU_skeletons_npy"  # Folder lưu kết quả
CONF_THRESHOLD = 0.5  # Độ tin cậy tối thiểu để chọn người
MAX_WORKERS = 6  # Số luồng chạy song song (Tùy CPU mạnh yếu)

# Đường dẫn model YOLOv8 Pose (thay bằng bản bạn muốn dùng hoặc đặt biến môi trường YOLO_POSE_MODEL)
POSE_MODEL_PATH = os.environ.get("YOLO_POSE_MODEL", "yolov8m-pose.pt")
YOLO_DEVICE = os.environ.get(
    "YOLO_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"
)

# Tạo folder output nếu chưa có
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Khóa để in log không bị lộn xộn
print_lock = threading.Lock()


def _infer_num_keypoints(model: YOLO) -> int:
    """Cố gắng đọc số keypoint từ kiến trúc YOLOv8 Pose, fallback về 17."""
    try:
        head = model.model.model[-1]
        kpt_shape = getattr(head, "kpt_shape", None)
        if kpt_shape:
            return int(kpt_shape[0])
    except AttributeError:
        pass

    kpt_shape = getattr(model.model, "kpt_shape", None)
    if kpt_shape:
        return int(kpt_shape[0])

    return 17  # Mặc định theo COCO


def process_video_to_npy(args):
    video_path, file_index, total_files = args

    # Lấy tên file (ví dụ: '0.mp4' -> '0')
    filename = os.path.basename(video_path)
    filename_no_ext = os.path.splitext(filename)[0]
    output_npy_path = os.path.join(OUTPUT_FOLDER, f"{filename_no_ext}.npy")

    # Resume: Nếu file npy đã có rồi thì bỏ qua không chạy lại
    if os.path.exists(output_npy_path):
        with print_lock:
            print(f"[Skip] {filename} đã có file npy.")
        return filename, "Skipped"

    # --- KHỞI TẠO LOCAL (Mỗi process 1 model riêng để tránh xung đột) ---
    pose_model = YOLO(POSE_MODEL_PATH)
    num_keypoints = _infer_num_keypoints(pose_model)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return filename, "Error Open"

    # List chứa dữ liệu của từng frame
    video_skeleton_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mặc định frame này là toàn số 0 (nếu không tìm thấy người)
        # Shape: (num_keypoints, 4) -> [x, y, z (2D => 0), score]
        current_frame_landmarks = np.zeros((num_keypoints, 4), dtype=np.float32)

        # 1. YOLOv8 Pose (vừa detect vừa trả ra keypoint)
        results = pose_model(
            frame, conf=CONF_THRESHOLD, device=YOLO_DEVICE, verbose=False
        )

        best_keypoints = None
        best_scores = None
        best_conf = -1.0

        for result in results:
            kp_struct = getattr(result, "keypoints", None)
            boxes = getattr(result, "boxes", None)
            if kp_struct is None or boxes is None or len(boxes) == 0:
                continue

            kp_xy = kp_struct.xy
            kp_conf = getattr(kp_struct, "conf", None)

            for idx in range(len(boxes)):
                conf_tensor = boxes.conf
                conf_value = (
                    float(conf_tensor[idx].item()) if conf_tensor is not None else 0.0
                )
                if conf_value <= best_conf:
                    continue

                keypoints_np = kp_xy[idx]
                if isinstance(keypoints_np, torch.Tensor):
                    keypoints_np = keypoints_np.detach().cpu().numpy()
                else:
                    keypoints_np = np.asarray(keypoints_np)

                if kp_conf is not None:
                    kp_scores = kp_conf[idx]
                    if isinstance(kp_scores, torch.Tensor):
                        kp_scores = kp_scores.detach().cpu().numpy()
                    else:
                        kp_scores = np.asarray(kp_scores)
                else:
                    kp_scores = np.full(num_keypoints, conf_value, dtype=np.float32)

                best_conf = conf_value
                best_keypoints = keypoints_np
                best_scores = kp_scores

        if best_keypoints is not None:
            kp_count = best_keypoints.shape[0]
            if kp_count != num_keypoints:
                raise RuntimeError(
                    f"Số keypoint thực tế ({kp_count}) khác cấu hình ({num_keypoints}). Kiểm tra lại model pose."
                )

            current_frame_landmarks[:, 0:2] = best_keypoints.astype(np.float32)
            current_frame_landmarks[:, 2] = 0.0  # Không có trục Z trong YOLO pose
            current_frame_landmarks[:, 3] = best_scores.astype(np.float32)

        # Lưu frame hiện tại vào list tổng
        video_skeleton_data.append(current_frame_landmarks)

    cap.release()

    # --- LƯU FILE .NPY ---
    if len(video_skeleton_data) > 0:
        # Final Shape: (Frames, num_keypoints, 4)
        final_array = np.array(video_skeleton_data, dtype=np.float32)
        np.save(output_npy_path, final_array)
        status = f"Done ({len(video_skeleton_data)} frames)"
    else:
        status = "Empty Video (No frames)"

    with print_lock:
        print(f"[{file_index}/{total_files}] {filename:<10} | {status}")

    return filename, status


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
    print(f"--> Thiết bị YOLO Pose: {YOLO_DEVICE} | {MAX_WORKERS} process song song")

    start_time = time.time()

    # Tạo danh sách tham số để chạy đa luồng
    args_list = [(f, i + 1, total_files) for i, f in enumerate(video_files)]

    # Bắt đầu chạy
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_video_to_npy, args_list)

    duration = time.time() - start_time
    print(f"\n[HOÀN TẤT] Tổng thời gian: {duration:.2f} giây")


if __name__ == "__main__":
    run_extraction()
