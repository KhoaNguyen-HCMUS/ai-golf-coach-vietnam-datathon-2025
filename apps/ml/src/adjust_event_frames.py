import cv2
import pandas as pd
import ast
import os
import re
from pathlib import Path

def get_video_fps(video_path):
    """Đọc FPS từ video file"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def adjust_event_frames(csv_path, videos_folder, output_csv_path=None, offset_seconds=0.5):
    """
    Điều chỉnh event frames trong CSV bằng cách cộng thêm offset * fps
    """
    # Đọc CSV
    df = pd.read_csv(csv_path)
    print(f"Đã đọc {len(df)} dòng từ {csv_path}")
    
    # Tạo bản sao để xử lý
    df_adjusted = df.copy()
    
    # Xử lý từng dòng
    videos_folder = Path(videos_folder)
    processed_count = 0
    error_count = 0
    fps_stats = {}
    
    for idx, row in df.iterrows():
        video_id = row['id']
        video_path = videos_folder / f"{video_id}.mp4"
        
        if not video_path.exists():
            print(f"Cảnh báo: Không tìm thấy video {video_path}")
            error_count += 1
            continue
        
        try:
            # Đọc FPS
            fps = get_video_fps(video_path)
            
            # Thống kê FPS
            fps_rounded = round(fps)
            if fps_rounded not in fps_stats:
                fps_stats[fps_rounded] = 0
            fps_stats[fps_rounded] += 1
            
            # Parse event frames từ string sang list
            events_str = row['events']
            
            if isinstance(events_str, float):  # NaN
                print(f"Video {video_id}: events là NaN, bỏ qua")
                error_count += 1
                continue
            
            events_str = str(events_str).strip()
            events_str = events_str.strip('[]').strip()
            events_str = re.sub(r'\s+', ',', events_str)
            events_str = f'[{events_str}]'
            
            events = ast.literal_eval(events_str)
            
            # Tính offset frame - làm tròn FPS trước
            fps_rounded = round(fps)
            offset_frames = round(offset_seconds * fps_rounded)
            
            # Debug: In ra lần đầu tiên gặp FPS này
            if idx < 5 or (idx + 1) % 50 == 0:
                print(f"Video {video_id} (dòng {idx}): FPS={fps:.2f} → rounded={fps_rounded}, offset_frames={offset_frames}")
            
            # Cộng offset vào tất cả event frames
            adjusted_events = [event + offset_frames for event in events]
            
            # Cập nhật vào dataframe
            df_adjusted.at[idx, 'events'] = str(adjusted_events)
            
            processed_count += 1
            
            if (idx + 1) % 100 == 0:
                print(f"Đã xử lý {idx + 1}/{len(df)} videos...")
                
        except Exception as e:
            print(f"Lỗi khi xử lý video {video_id} (dòng {idx}): {str(e)}")
            print(f"  Giá trị events: {repr(row['events'][:100])}")
            error_count += 1
    
    # Lưu kết quả
    if output_csv_path is None:
        output_csv_path = csv_path
    
    df_adjusted.to_csv(output_csv_path, index=False)
    print(f"\n=== Hoàn thành ===")
    print(f"Đã xử lý thành công: {processed_count} videos")
    print(f"Lỗi: {error_count} videos")
    print(f"Thống kê FPS: {fps_stats}")
    print(f"Đã lưu vào: {output_csv_path}")

if __name__ == "__main__":
    # Cấu hình đường dẫn
    csv_path = "" # Đường dẫn tới file GolfDB.csv
    videos_folder = "" # Thư mục chứa videos_hd GolfDB
    output_csv_path = "" # Đường dẫn lưu file CSV đã điều chỉnh
    
    # Chạy điều chỉnh
    adjust_event_frames(
        csv_path=csv_path,
        videos_folder=videos_folder,
        output_csv_path=output_csv_path,
        offset_seconds=0.5
    )