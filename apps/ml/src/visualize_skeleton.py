import cv2
import numpy as np
import os
import argparse
import glob
import random

# --- CẤU HÌNH MẶC ĐỊNH ---
DEFAULT_VIDEO_DIR = "../../../data/TDTU-Golf-Pose-v1/videos"
DEFAULT_NPY_DIR = "../../../data/TDTU-Golf-Pose-v1/raw_yolov8_npy"
OUTPUT_VIS_DIR = "../../../data/TDTU-Golf-Pose-v1/yolov8_vis_results"

# --- COCO KEYPOINTS CONFIG ---
# 17 Keypoints mapping
KP_NAMES = {
    0: "Nose", 1: "L-Eye", 2: "R-Eye", 3: "L-Ear", 4: "R-Ear",
    5: "L-Shoulder", 6: "R-Shoulder", 7: "L-Elbow", 8: "R-Elbow",
    9: "L-Wrist", 10: "R-Wrist", 11: "L-Hip", 12: "R-Hip",
    13: "L-Knee", 14: "R-Knee", 15: "L-Ankle", 16: "R-Ankle"
}

# Các đường nối (Start Index, End Index, Color)
# Color format: BGR
SKELETON_CONNECTIONS = [
    # Face
    (0, 1, (255, 0, 0)), (0, 2, (0, 0, 255)), 
    (1, 3, (255, 0, 0)), (2, 4, (0, 0, 255)),
    # Torso
    (5, 6, (255, 0, 255)), 
    (5, 11, (255, 0, 255)), (6, 12, (255, 0, 255)),
    (11, 12, (255, 0, 255)),
    # Left Arm
    (5, 7, (0, 255, 0)), (7, 9, (0, 255, 0)),
    # Right Arm
    (6, 8, (0, 0, 255)), (8, 10, (0, 0, 255)),
    # Left Leg
    (11, 13, (0, 255, 0)), (13, 15, (0, 255, 0)),
    # Right Leg
    (12, 14, (0, 0, 255)), (14, 16, (0, 0, 255)),
]

def draw_skeleton(frame, keypoints, conf_threshold=0.5):
    """
    Vẽ skeleton lên frame.
    keypoints shape: (17, 4) -> [x, y, z, score]
    """
    # Vẽ các khớp (Points)
    for i in range(keypoints.shape[0]):
        x, y, z, score = keypoints[i]
        if score < conf_threshold:
            continue
        
        # Vẽ chấm tròn
        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

    # Vẽ các đường nối (Limbs)
    for start_idx, end_idx, color in SKELETON_CONNECTIONS:
        kp_start = keypoints[start_idx]
        kp_end = keypoints[end_idx]

        # Kiểm tra độ tin cậy của cả 2 điểm đầu cuối
        if kp_start[3] < conf_threshold or kp_end[3] < conf_threshold:
            continue

        start_pt = (int(kp_start[0]), int(kp_start[1]))
        end_pt = (int(kp_end[0]), int(kp_end[1]))

        cv2.line(frame, start_pt, end_pt, color, 2, cv2.LINE_AA)

    return frame

def visualize_video(video_path, npy_path, output_path, show=False):
    if not os.path.exists(video_path):
        print(f"[Error] Không tìm thấy video: {video_path}")
        return
    if not os.path.exists(npy_path):
        print(f"[Error] Không tìm thấy file npy: {npy_path}")
        return

    if os.path.exists(output_path):
        print(f"[Skip] Output đã tồn tại: {output_path}")
        return

    # Load data
    skeleton_data = np.load(npy_path) # Shape: (Frames, 17, 4)
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup Video Writer
    # Output width sẽ gấp đôi (Video gốc + Black background)
    out_width = width * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. Tạo Canvas màu đen cùng kích thước frame
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 2. Lấy keypoints tương ứng frame hiện tại
        if frame_idx < len(skeleton_data):
            current_kpts = skeleton_data[frame_idx]
            
            # Vẽ lên frame gốc (Overlay)
            frame_overlay = draw_skeleton(frame.copy(), current_kpts)
            
            # Vẽ lên nền đen (Separate Skeleton)
            frame_skeleton = draw_skeleton(black_frame, current_kpts)
        else:
            frame_overlay = frame
            frame_skeleton = black_frame

        # 3. Ghép 2 frame lại (Side-by-side)
        # Trái: Video gốc + Skeleton | Phải: Nền đen + Skeleton
        combined_frame = np.hstack((frame_overlay, frame_skeleton))

        # Hiển thị thông tin Frame
        cv2.putText(combined_frame, f"Frame: {frame_idx}/{total_frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined_frame, "Overlay", (10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(combined_frame, "Skeleton Only", (width + 10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        out.write(combined_frame)

        if show:
            cv2.imshow('Skeleton Visualization', combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_idx += 1

    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()
    
    print(f"[Done] Đã lưu video kết quả tại: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Skeleton from NPY file")
    parser.add_argument("--id", type=str, help="ID của video để visualize (ví dụ: 127)")
    parser.add_argument("--video", type=str, help="Đường dẫn đến file video cụ thể (Optional)")
    parser.add_argument("--npy", type=str, help="Đường dẫn đến file npy cụ thể (Optional)")
    parser.add_argument("--output", type=str, default=OUTPUT_VIS_DIR, help="Thư mục output")
    parser.add_argument("--show", action="store_true", help="Hiển thị cửa sổ popup khi chạy")
    parser.add_argument("--random", action="store_true", help="Chọn ngẫu nhiên 1 video trong data để test")
    parser.add_argument("--all", action="store_true", help="Xử lý tất cả video tìm thấy")
    
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. Chế độ dùng ID
    if args.id:
        video_id = args.id
        print(f"--> Tìm kiếm ID: {video_id}")
        
        # Construct paths
        target_video = os.path.join(DEFAULT_VIDEO_DIR, f"{video_id}.mp4")
        target_npy = os.path.join(DEFAULT_NPY_DIR, f"{video_id}.npy")
        
        if not os.path.exists(target_video):
            print(f"[Error] Không tìm thấy video ID {video_id} tại: {target_video}")
            return
            
        output_filename = f"vis_{video_id}.mp4"
        output_path = os.path.join(args.output, output_filename)
        
        visualize_video(target_video, target_npy, output_path, args.show)
        return

    target_video = args.video
    target_npy = args.npy

    # 2. Chế độ chỉ định Path cụ thể
    if target_video:
        if not target_npy:
            filename = os.path.splitext(os.path.basename(target_video))[0]
            target_npy = os.path.join(DEFAULT_NPY_DIR, f"{filename}.npy")
        
        output_filename = f"vis_{os.path.basename(target_video)}"
        output_path = os.path.join(args.output, output_filename)
        visualize_video(target_video, target_npy, output_path, args.show)
        return

    # 3. Tìm tất cả video mp4
    all_videos = glob.glob(os.path.join(DEFAULT_VIDEO_DIR, "*.mp4"))
    if not all_videos:
        print("Không tìm thấy video nào trong thư mục")
        return
    
    # Lọc ra những video đã có file npy tương ứng
    valid_pairs = []
    for vid in all_videos:
        filename = os.path.splitext(os.path.basename(vid))[0]
        npy = os.path.join(DEFAULT_NPY_DIR, f"{filename}.npy")
        if os.path.exists(npy):
            valid_pairs.append((vid, npy))
    
    if not valid_pairs:
        print("Không tìm thấy cặp video-npy nào khớp nhau!")
        return

    # 4. Các chế độ khác (Random / All / Default)
    if args.random:
        # Chọn ngẫu nhiên 1 video
        target_video, target_npy = random.choice(valid_pairs)
        print(f"--> Chọn ngẫu nhiên: {os.path.basename(target_video)}")
        output_filename = f"vis_{os.path.basename(target_video)}"
        output_path = os.path.join(args.output, output_filename)
        visualize_video(target_video, target_npy, output_path, args.show)
    elif args.all:
        # Xử lý tất cả video
        total = len(valid_pairs)
        print(f"--> Tìm thấy {total} video-npy pairs, bắt đầu xử lý...")
        
        for idx, (vid, npy) in enumerate(valid_pairs, 1):
            video_name = os.path.basename(vid)
            print(f"[{idx}/{total}] Đang xử lý: {video_name}")
            
            output_filename = f"vis_{video_name}"
            output_path = os.path.join(args.output, output_filename)
            
            # Skip nếu đã tồn tại
            if os.path.exists(output_path):
                print(f"  -> Đã tồn tại, bỏ qua.")
                continue
            
            visualize_video(vid, npy, output_path, show=False)
        
        print(f"\n[HOÀN TẤT] Đã xử lý {total} video.")
    else:
        # Mặc định: chọn video đầu tiên
        target_video, target_npy = valid_pairs[0]
        print(f"--> Chọn video đầu tiên tìm thấy: {os.path.basename(target_video)}")
        output_filename = f"vis_{os.path.basename(target_video)}"
        output_path = os.path.join(args.output, output_filename)
        visualize_video(target_video, target_npy, output_path, args.show)

if __name__ == "__main__":
    main()
