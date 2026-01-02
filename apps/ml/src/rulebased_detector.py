import numpy as np
import pandas as pd
import os
import glob
from scipy.signal import savgol_filter
from visualize_phases import visualize_video

class GolfNpyDetector:
    def __init__(self):
        # Mapping from keypoint count to (left wrist, right wrist) indices
        self.WRIST_INDEX_MAP = {
            33: (15, 16),  # MediaPipe
            25: (15, 16),  # Some full-body variants
            17: (9, 10)    # COCO/YOLO
        }

    def _resolve_wrists(self, num_keypoints):
        """Return wrist indices based on skeleton layout."""
        if num_keypoints in self.WRIST_INDEX_MAP:
            return self.WRIST_INDEX_MAP[num_keypoints]

        # Fallback: best-effort mapping
        if num_keypoints > 17:
            return 15, 16
        # Minimal skeletons usually follow COCO order
        return 9, 10

    def smooth_signal(self, data, window_length=5, polyorder=2):
        """Smooths jittery predictions using Savitzky-Golay filter."""
        if len(data) < window_length: return data
        return savgol_filter(data, window_length, polyorder)

    def detect_phases(self, keypoints):
        """
        Input: keypoints shape (Frames, K, >=2)
        """
        phases = {'Address': 0, 'Top': 0, 'Impact': 0, 'Finish': 0}
        total_frames = len(keypoints)
        if total_frames < 10: return phases

        num_keypoints = keypoints.shape[1]
        left_idx, right_idx = self._resolve_wrists(num_keypoints)
        if left_idx >= num_keypoints or right_idx >= num_keypoints:
            raise ValueError(
                f"Wrist indices ({left_idx}, {right_idx}) vượt quá số keypoint {num_keypoints}."
            )

        # 1. Extract Average Wrist Y-coordinate
        # Note: In images, Y=0 is TOP. So "Highest Hands" = "Lowest Y value".
        l_y = keypoints[:, left_idx, 1]
        r_y = keypoints[:, right_idx, 1]
        wrist_y = (l_y + r_y) / 2.0
        
        # Smooth the signal to remove random detection noise
        wrist_y = self.smooth_signal(wrist_y)

        # 2. Calculate Velocity (for Address/Finish)
        # Diff between frames: sqrt((x2-x1)^2 + (y2-y1)^2)
        diff = np.diff(keypoints[:, :, :2], axis=0)
        velocity = np.linalg.norm(diff, axis=2).mean(axis=1) # Avg velocity of all joints
        velocity = np.insert(velocity, 0, 0) # Pad to match size
        
        # --- PHASE 1: FIND "TOP" (First Significant Peak) ---
        # Logic: Find local minima (valleys) in the Y-plot.
        # The Top is the first valley that is "deep enough" (high hands).
        
        # A. Find all local minima (frame indices where Y is lower than neighbors)
        # We check neighbors +/- 5 frames to ensure it's a real phase, not jitter
        local_minima = []
        for i in range(5, total_frames - 5):
            if wrist_y[i] < wrist_y[i-1] and wrist_y[i] < wrist_y[i+1]:
                # Check wider window to confirm it's a peak
                if wrist_y[i] == np.min(wrist_y[i-5:i+6]):
                    local_minima.append(i)
        
        if not local_minima:
            # Fallback: Just global min if no local found
            phases['Top'] = np.argmin(wrist_y)
        else:
            # B. Filter Minima: Ignore valleys that are "too low"
            # The Top of swing usually puts hands in the upper 50% of the screen height range.
            min_height = np.min(wrist_y)
            max_height = np.max(wrist_y)
            height_threshold = min_height + 0.6 * (max_height - min_height)
            
            # Select the FIRST minimum that satisfies the height threshold
            found_top = False
            for peak_frame in local_minima:
                if wrist_y[peak_frame] < height_threshold:
                    phases['Top'] = peak_frame
                    found_top = True
                    break # STOP after the first one! This avoids the Finish.
            
            if not found_top:
                phases['Top'] = np.argmin(wrist_y)

        # --- PHASE 2: FIND "ADDRESS" ---
        # Search BACKWARDS from Top for stillness
        stillness_thresh = 0.002 # Tune based on your data scale
        
        phases['Address'] = 0 # Default
        for i in range(phases['Top'], 0, -1):
            if velocity[i] < stillness_thresh:
                phases['Address'] = i
                break
        
        # --- PHASE 3: FIND "IMPACT" ---
        # Search FORWARD from Top. 
        # Impact is the lowest point (Max Y) between Top and Finish.
        # It usually happens within 1 second (30-60 frames) after top.
        search_window = 45
        end_search = min(total_frames, phases['Top'] + search_window)
        
        if end_search > phases['Top']:
            downswing_y = wrist_y[phases['Top']:end_search]
            # Find Max Y (Lowest physical point) in downswing
            local_max_idx = np.argmax(downswing_y)
            phases['Impact'] = phases['Top'] + local_max_idx
        else:
            phases['Impact'] = phases['Top'] + 10

        # --- PHASE 4: FIND "FINISH" ---
        # Search FORWARD from Impact for stillness
        finish_search_start = phases['Impact'] + 5
        if finish_search_start < total_frames:
            follow_through_y = wrist_y[finish_search_start:]
            
            # 2. Find the Highest Point (Lowest Y) in the follow-through
            if len(follow_through_y) > 0:
                # This gives us the frame where hands are physically highest
                peak_height_idx = np.argmin(follow_through_y)
                peak_finish_frame = finish_search_start + peak_height_idx
                
                phases['Finish'] = peak_finish_frame
                
                # 3. Optional Refinement: Check for stability *after* the peak
                # (Sometimes the "Pose Hold" is slightly after the absolute peak height)
                # Look 10 frames ahead of the peak to see if they hold it
                check_end = min(total_frames, peak_finish_frame + 15)
                best_frame = peak_finish_frame
                min_vel = float('inf')
                
                for i in range(peak_finish_frame, check_end):
                    if velocity[i] < min_vel:
                        min_vel = velocity[i]
                        best_frame = i
                
                # If they held a pose with low velocity, snap to that. 
                # Otherwise, stick to the peak height.
                if min_vel < stillness_thresh * 2: 
                    phases['Finish'] = best_frame
            else:
                phases['Finish'] = total_frames - 1
        else:
            phases['Finish'] = total_frames - 1

        return phases

def process_npy_folder(folder_path, output_csv):
    detector = GolfNpyDetector()
    results = []
    
    # Get all .npy files
    npy_files = glob.glob(os.path.join(folder_path, "*.npy"))
    npy_files.sort()
    
    print(f"Found {len(npy_files)} .npy files. Processing...")
    
    for npy_path in npy_files:
        vid_name = os.path.basename(npy_path).replace('.npy', '') # Video name
        
        try:
            # 1. Load Data
            keypoints = np.load(npy_path)
            
            # 2. Apply Rules
            phases = detector.detect_phases(keypoints)
	
            # 3. Store Result
            row = {
                'Video': vid_name,
                'Address': phases['Address'],
                'Top': phases['Top'],
                'Impact': phases['Impact'],
                'Finish': phases['Finish']
            }
            results.append(row)
            # Optional: Print for debugging
            # print(f"{vid_name}: Top={phases['Top']}, Impact={phases['Impact']}")
            if vid_name:
                visualize_video(video_path=f'data/TDTU_raw/{vid_name}.mp4', output_path=f'data/debug_rule/debug_{vid_name}.mp4', events=phases)
                
        except Exception as e:
            print(f"Error processing {vid_name}: {e}")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nDone! Results saved to {output_csv}")

if __name__ == "__main__":
    input_folder = "data/TDTU_skeletons_npy"
    output_file = "tdtu_npy_detections.csv"
    process_npy_folder(input_folder, output_file)

    

    