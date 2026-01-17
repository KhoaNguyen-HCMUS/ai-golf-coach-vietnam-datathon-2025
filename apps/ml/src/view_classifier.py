import numpy as np
import os
import glob

class GolfViewClassifier:
    """
    Class phân loại góc nhìn Golf (Backside vs Side) dựa trên Skeleton.
    Logic: Rule-based dựa trên tỷ lệ độ rộng vai/hông trong Phase Setup (15 frame đầu).
    """
    
    # Cấu hình Threshold cho từng loại Model/Input (Dựa trên thực nghiệm)
    THRESHOLDS = {
        'yolov8': 0.20,
        'vitpose_preprocessed': 0.2485,
        'vitpose_raw': 0.2152
    }

    def __init__(self, model_type='vitpose_preprocessed'):
        """
        Args:
            model_type (str): 'yolov8', 'vitpose_preprocessed', hoặc 'vitpose_raw'
        """
        if model_type not in self.THRESHOLDS:
            raise ValueError(f"Unknown model_type: {model_type}. Supported: {list(self.THRESHOLDS.keys())}")
        
        self.threshold = self.THRESHOLDS[model_type]
        self.model_type = model_type
        print(f"[ViewClassifier] Init with type='{model_type}', Threshold={self.threshold}")

    def predict(self, skeleton_data):
        """
        Dự đoán góc nhìn từ dữ liệu skeleton.
        
        Args:
            skeleton_data (np.ndarray): Shape (Frames, 17, 3+). Ít nhất cần (x, y). 
                                        Frames nên > 5 để lấy trung bình setup.
        
        Returns:
            str: 'backside' hoặc 'side'
            dict: Thông tin chi tiết (ratio, confidence, etc.)
        """
        if skeleton_data is None or len(skeleton_data) == 0:
            return "unknown", {}

        # 1. Lấy giai đoạn Setup (15 frame đầu)
        # Nếu video quá ngắn thì lấy tất cả
        setup_duration = min(15, skeleton_data.shape[0])
        early_frames = skeleton_data[:setup_duration]
        
        # 2. Tính trung bình tọa độ (Mean Skeleton) để giảm nhiễu
        # Shape: (17, C) -> Lấy (17, 2) cho x,y
        kp_avg = np.mean(early_frames[:, :, :2], axis=0)
        
        # 3. Trích xuất các điểm quan trọng
        # COCO Indices: 5:LSh, 6:RSh, 11:LHip, 12:RHip
        l_sh = kp_avg[5]
        r_sh = kp_avg[6]
        l_hip = kp_avg[11]
        r_hip = kp_avg[12]
        
        # 4. Tính toán kích thước hình học
        # - Torso Height (Chiều cao thân): Tham chiếu để chuẩn hóa (Scale Invariant)
        #   Tính từ trung điểm vai đến trung điểm hông
        shoulder_mid = (l_sh + r_sh) / 2
        hip_mid = (l_hip + r_hip) / 2
        torso_height = np.linalg.norm(shoulder_mid - hip_mid)
        
        if torso_height == 0:
            return "unknown", {"reason": "Zero torso height"}

        # - Shoulder Width (Độ rộng vai theo trục X)
        shoulder_w = np.abs(l_sh[0] - r_sh[0])
        
        # - Hip Width (Độ rộng hông theo trục X)
        hip_w = np.abs(l_hip[0] - r_hip[0])
        
        # 5. Tính tỷ lệ (Ratio)
        s_ratio = shoulder_w / torso_height
        h_ratio = hip_w / torso_height
        
        # Trung bình cộng của tỷ lệ Vai và Hông
        avg_ratio = (s_ratio + h_ratio) / 2
        
        # 6. Phân loại dựa trên Threshold
        # Quy tắc: 
        # - Backside/Down-the-line: Vai chồng lên nhau -> Width nhỏ -> Ratio < Threshold
        # - Side/Face-on: Vai dang rộng -> Width lớn -> Ratio >= Threshold
        
        if avg_ratio < self.threshold:
            view = 'backside'
        else:
            view = 'side'
            
        return view, {
            "ratio": float(avg_ratio), 
            "threshold": self.threshold,
            "shoulder_ratio": float(s_ratio),
            "hip_ratio": float(h_ratio)
        }

# --- Module Testing Block ---
if __name__ == "__main__":
    # Test nhanh với 1 file trong folder dữ liệu (nếu có)
    # Giả lập input
    test_npy_folder = r"D:\Competition\ai-golf-coach-vietnam-datathon-2025\data\TDTU-Golf-Pose-v1\preprocessed_vitpose_npy"
    if os.path.exists(test_npy_folder):
        print("--- Testing View Classifier Module ---")
        classifier = GolfViewClassifier(model_type='vitpose_preprocessed')
        
        npy_files = glob.glob(os.path.join(test_npy_folder, "*.npy"))[:5] # Test 5 file đầu
        for f in npy_files:
            data = np.load(f)
            view, info = classifier.predict(data)
            fid = os.path.basename(f)
            print(f"File: {fid:<10} | View: {view:<8} | Ratio: {info['ratio']:.4f}")
