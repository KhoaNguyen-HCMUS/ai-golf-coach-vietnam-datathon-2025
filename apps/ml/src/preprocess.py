import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
import torch
from concurrent.futures import ThreadPoolExecutor
import time

# Config
INPUT_PATH = "../../../data/TDTU-Golf-Pose-v1/videos"
OUTPUT_PATH = "../../../data/TDTU-Golf-Pose-v1/preprocessed"
NUM_WORKERS = 4

class VideoPreprocessor:
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        print(f"Initializing VideoPreprocessor using device: {self.device}")
        
        # Load YOLO model for smart cropping (using 'x' model for best accuracy as requested)
        self.yolo_model = YOLO('yolov8x-pose.pt') 

    def apply_clahe(self, image):
        """
        Step 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Chống ngược sáng và tăng chi tiết.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Calculate average brightness
        avg_brightness = np.mean(l)

        # Apply CLAHE based on brightness
        if avg_brightness < 90: # Very dark
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
        elif avg_brightness < 140: # Slightly dark
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l = clahe.apply(l)
        elif avg_brightness > 180: # Very bright (overexposed)
            # Apply stronger CLAHE to recover details in washed out areas
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
        # If bright enough, keep original L channel

        # Merge and convert back to BGR
        limg = cv2.merge((l, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

    def estimate_motion(self, frames):
        """
        Estimates camera motion between frames using Optical Flow.
        Returns: transforms (List of 2x3 affine matrices)
        """
        if not frames:
            return []

        transforms = [np.identity(3)] # 3x3 for easier multiplication
        
        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Detect feature points in previous frame
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            
            if p0 is None:
                transforms.append(transforms[-1]) # No motion detected, copy previous
                prev_gray = curr_gray
                continue

            # Calculate optical flow (i.e., track feature points)
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
            
            # Select good points
            good_prev = p0[st==1]
            good_curr = p1[st==1]
            
            if len(good_prev) < 4: # Need at least a few points to estimate transform
                transforms.append(transforms[-1])
            else:
                # Estimate affine transformation (rotation + translation)
                m, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
                if m is None:
                     transforms.append(transforms[-1])
                else:
                    # m is 2x3, convert to 3x3
                    m_3x3 = np.vstack([m, [0, 0, 1]])
                    
                    # Accumulate transformation (matrix multiplication)
                    # Current absolute transform = Previous absolute transform * Current relative transform
                    # Actually we usually store relative transforms for smoothing, 
                    # but here let's store absolute trajectory then smooth it.
                    
                    # Let's simplify: Standard stabilization uses trajectory smoothing.
                    # 1. Get relative transform dx, dy, da
                    dx = m[0, 2]
                    dy = m[1, 2]
                    da = np.arctan2(m[1, 0], m[0, 0])
                    
                    transforms.append([dx, dy, da])
            
            prev_gray = curr_gray
            
        return np.array(transforms)

    def smooth_trajectory(self, trajectory, smoothing_radius=30):
        """
        Smooths the trajectory using a moving average window.
        """
        smoothed_trajectory = np.copy(trajectory)
        if len(trajectory) <= smoothing_radius:
            return trajectory # Too short to smooth

        # Padding for convolution
        # We process dx, dy, da separately
        for i in range(3):
            kernel = np.ones(smoothing_radius) / smoothing_radius
            # Reflect padding to minimize edge effects
            padded = np.pad(trajectory[:, i], (smoothing_radius//2, smoothing_radius//2), mode='edge')
            smoothed = np.convolve(padded, kernel, mode='valid')
            
            # Adjust length match if needed (sometimes conv valid mode changes size slightly depending on odd/even)
            min_len = min(len(smoothed_trajectory), len(smoothed))
            smoothed_trajectory[:min_len, i] = smoothed[:min_len]

        return smoothed_trajectory

    def stabilize_video(self, frames):
        """
        Step 2: Video Stabilization
        """
        if len(frames) < 2:
            return frames

        # 1. Estimate motion (trajectory) -> [dx, dy, da] per frame
        # Note: This is a simplified "rigid" stabilization. 
        # For full affine, we need to smooth the full affine matrix elements.
        # Let's use a simpler approach of smoothing the 'trajectory' of x, y, angle.
        
        # Re-implementation for robust trajectory smoothing
        transforms = [] 
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        # Get relative transforms
        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
            
            dx, dy, da = 0, 0, 0
            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
                if p1 is not None:
                    good_prev = p0[st==1]
                    good_curr = p1[st==1]
                    if len(good_prev) > 4:
                        m, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
                        if m is not None:
                            dx = m[0, 2]
                            dy = m[1, 2]
                            da = np.arctan2(m[1, 0], m[0, 0])
            
            transforms.append([dx, dy, da])
            prev_gray = curr_gray
        
        transforms = np.array(transforms)
        
        # Calculate trajectory (cumulative sum of transforms)
        trajectory = np.cumsum(transforms, axis=0) 
        
        # Smooth trajectory
        smoothed_trajectory = self.smooth_trajectory(trajectory)
        
        # Calculate difference (new transform = transform + (smoothed - trajectory))
        difference = smoothed_trajectory - trajectory
        transforms_smooth = transforms + difference
        
        # Apply smoothed transforms
        stabilized_frames = [frames[0]] # Keep first frame as anchor or crop it slightly
        h, w = frames[0].shape[:2]
        
        for i in range(len(transforms_smooth)):
            dx = transforms_smooth[i, 0]
            dy = transforms_smooth[i, 1]
            da = transforms_smooth[i, 2]
            
            # Build transformation matrix
            m = np.zeros((2, 3))
            m[0, 0] = np.cos(da)
            m[0, 1] = -np.sin(da)
            m[1, 0] = np.sin(da)
            m[1, 1] = np.cos(da)
            m[0, 2] = dx
            m[1, 2] = dy
            
            frame_warped = cv2.warpAffine(frames[i+1], m, (w, h))
            stabilized_frames.append(frame_warped)
            
        return stabilized_frames

    def get_smart_crops(self, frames):
        """
        Step 3.1: Calculate Smart Cropping Boxes using YOLOv8
        Returns a single fixed crop box (x1, y1, x2, y2) for the whole video (or moving window).
        To assume "Môi trường", we should track the player.
        Golfer stays relatively in one place during swing.
        We will find the bounding box of the person in all frames, get the union/avg, and add margins.
        """
        
        # We don't need to detect on EVERY frame to get the general crop area.
        # Detecting every 5th frame is enough for finding the "arena".
        # However, for highest accuracy in "Smart Cropping", let's be robust.
        
        boxes = []
        
        # Inference in batches for speed if memory allows, but simple loop is safer for video memory
        # Create a list of indices to check (e.g., every 5 frames)
        
        stride = 5
        indices = range(0, len(frames), stride)
        subset_frames = [frames[i] for i in indices]
        
        if not subset_frames:
            return None
            
        # Run YOLO inference
        results = self.yolo_model(subset_frames, device=self.device, classes=[0], verbose=False) # class 0 is person
        
        all_boxes = []
        for r in results:
            # Get the largest person box
            if r.boxes:
                # Get box with highest confidence or largest area. Usually largest area is the golfer in frame.
                # r.boxes.xyxy is (N, 4)
                # Find max area
                areas = (r.boxes.xyxy[:, 2] - r.boxes.xyxy[:, 0]) * (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1])
                max_idx = torch.argmax(areas).item()
                box = r.boxes.xyxy[max_idx].cpu().numpy() # [x1, y1, x2, y2]
                all_boxes.append(box)
                
        if not all_boxes:
            return None # No person found

        all_boxes = np.array(all_boxes)
        
        # Strategy: To stabilize the crop, we can take the median box of the golfer
        # and expand it by 15%
        # Or take the union of all boxes if the camera moves (but we stabilized it!).
        # Since we stabilized, the golfer should be centered.
        
        median_box = np.median(all_boxes, axis=0) # Robust to outliers
        x1, y1, x2, y2 = median_box
        
        # Expand more horizontally to catch full swing arc, less vertically
        w_box = x2 - x1
        h_box = y2 - y1
        margin_x = 0.80
        margin_y = 0.40
        
        x1 = max(0, x1 - w_box * margin_x)
        y1 = max(0, y1 - h_box * margin_y)
        x2 = min(frames[0].shape[1], x2 + w_box * margin_x)
        y2 = min(frames[0].shape[0], y2 + h_box * margin_y)
        
        return (int(x1), int(y1), int(x2), int(y2))

    def process_video(self, input_path, output_path):
        try:
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                print(f"Error opening video: {input_path}")
                return

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                frames.append(frame)
            cap.release()
            
            if not frames:
                return

            # --- Step 1: CLAHE ---
            # Process frames in memory? If video is too long, this crashes RAM.
            # Assuming short clips (golf swings < 10s).
            frames = [self.apply_clahe(f) for f in frames]

            # --- Step 2: Stabilization ---
            # frames = self.stabilize_video(frames) # Using simple stabilization logic

            # --- Step 3: Smart Cropping ---
            crop_box = self.get_smart_crops(frames)
            
            final_frames = []
            if crop_box:
                x1, y1, x2, y2 = crop_box
                for f in frames:
                    crop = f[y1:y2, x1:x2]
                    # Resize to multiple of 32 better for models, but original asked just to crop
                    final_frames.append(crop)
            else:
                final_frames = frames # Fallback

            # Save
            if final_frames:
                h, w = final_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, original_fps, (w, h))
                for f in final_frames:
                    out.write(f)
                out.release()
                print(f"Completed: {input_path} -> {output_path}")

        except Exception as e:
            print(f"Failed to process {input_path}: {e}")

    def process_folder(self, input_folder, output_folder, max_workers=4):
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        video_files = list(input_folder.glob("*.mp4"))
        print(f"Found {len(video_files)} videos in {input_folder}")
        
        # Use ThreadPoolExecutor
        # Note: Since YOLO uses GPU, multithreading might have contention if batching isn't handled by internal logic.
        # But for video I/O and pre/post processing, threads help.
        # Limit workers to avoid OOM if loading many videos into RAM.
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for vid in video_files:
                out_path = output_folder / vid.name
                futures.append(executor.submit(self.process_video, vid, out_path))
            
            for f in futures:
                f.result() # Wait for all


def main():
    print(f"Processing from: {INPUT_PATH}")
    print(f"Output to: {OUTPUT_PATH}")
    
    preprocessor = VideoPreprocessor()
    
    if os.path.isfile(INPUT_PATH):
        preprocessor.process_video(INPUT_PATH, OUTPUT_PATH)
    else:
        preprocessor.process_folder(INPUT_PATH, OUTPUT_PATH, NUM_WORKERS)

if __name__ == "__main__":
    main()
