import cv2
import numpy as np
import pandas as pd
import os
import argparse
import random
from tqdm import tqdm

def apply_flip(frame):
    """Flips the frame horizontally."""
    return cv2.flip(frame, 1)

def apply_perspective(frame, strength=0.1):
    """
    Applies a random perspective transform to the frame.
    strength: Controls the degree of distortion (0.0 to 1.0).
    """
    h, w = frame.shape[:2]
    
    # Define random source and destination points
    src_points = np.float32([
        [0, 0],
        [w-1, 0],
        [0, h-1],
        [w-1, h-1]
    ])
    
    # Perturb the corners
    # Limit distortion to avoid black borders being too huge or chopping off too much
    dx = w * strength
    dy = h * strength
    
    dst_points = np.float32([
        [random.uniform(0, dx), random.uniform(0, dy)],             # Top-left
        [random.uniform(w-dx, w), random.uniform(0, dy)],           # Top-right
        [random.uniform(0, dx), random.uniform(h-dy, h)],           # Bottom-left
        [random.uniform(w-dx, w), random.uniform(h-dy, h)]          # Bottom-right
    ])
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(frame, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return warped

def apply_color_jitter(frame, brightness=0.2, contrast=0.2):
    """
    Adjusts brightness and contrast randomly.
    brightness: Max fraction to adjust brightness (+/-).
    contrast: Max fraction to adjust contrast (+/-).
    """
    # Contrast: alpha
    alpha = 1.0 + random.uniform(-contrast, contrast)
    # Brightness: beta (scale by 255 for CV2)
    beta = random.uniform(-brightness, brightness) * 255
    
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def process_video(input_path, output_path, transform_type='flip'):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return False

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Define codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'avc1' or 'X264' depending on system
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Pre-calculate random parameters for consistent video augmentation if needed
    # For perspective/jitter, we usually want it constant per video or per frame?
    # Requirement: "Perspective Transform (Giả lập nghiêng)... Train model trên đống này để nó không bị "sốc" khi gặp video test bị nghiêng."
    # implies creating a video that LOOKS tilted. If we change it per frame, it will look like an earthquake.
    # So we should calculate the matrix ONCE per video for perspective.
    # Same for brightness/contrast usually, to simulate a specific environment.
    
    # Store augmentation params
    perspective_matrix = None
    alpha = 1.0
    beta = 0.0

    if transform_type == 'augment':
        # Prepare perspective matrix
        strength = 0.1 # Adjust this if too strong/weak
        h, w = height, width
        dx = w * strength
        dy = h * strength
        
        src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
        dst_points = np.float32([
            [random.uniform(0, dx), random.uniform(0, dy)],
            [random.uniform(w-dx, w), random.uniform(0, dy)],
            [random.uniform(0, dx), random.uniform(h-dy, h)],
            [random.uniform(w-dx, w), random.uniform(h-dy, h)]
        ])
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Prepare color jitter
        alpha = 1.0 + random.uniform(-0.2, 0.2)
        beta = random.uniform(-0.2, 0.2) * 255
        
        # Random flip decision for augmentation too?
        # User said: "Flip Horizontal: ... (X2 dữ liệu)." 
        # Then "Perspective... Train model on this pile"
        # Often we want Flip + Perspective + Color.
        do_flip = random.choice([True, False])

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if transform_type == 'flip':
            frame = apply_flip(frame)
        elif transform_type == 'augment':
            if do_flip:
                frame = apply_flip(frame)
            
            # Warp
            frame = cv2.warpPerspective(frame, perspective_matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
            
            # Color
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    return True

def main():
    parser = argparse.ArgumentParser(description="Augment Golf Dataset")
    parser.add_argument('--metadata', type=str, required=True, help="Path to video_metadata.csv")
    parser.add_argument('--video_dir', type=str, required=True, help="Path to videos directory")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    
    # Find max ID to append new ones
    if 'id' in df.columns:
        next_id = df['id'].max() + 1
    else:
         print("Error: 'id' column not found in metadata.")
         return

    new_rows = []
    
    # Iterate over existing rows
    # We use a list to avoid modifying the df while iterating
    original_rows = df.to_dict('records')

    print(f"Found {len(original_rows)} videos. Starting augmentation...")

    for row in tqdm(original_rows):
        original_file_name = row['original_name'] # Note: this seems to be the logical name.
        # However, checking the user's file list, the files are named '1.mp4', '2.mp4', etc.
        # But the metadata shows 'original_name' as 'Backside-xxxxx.mov'.
        # Let's double check how files are actually named on disk vs metadata.
        # User list_dir showed: "1.mp4", "2.mp4"... "50.mp4".
        # Metadata 'id' matches these numbers likely.
        # Let's assume the file on disk corresponding to row['id'] is "{id}.mp4".
        
        vid_id = row['id']
        input_filename = f"{vid_id}.mp4"
        input_path = os.path.join(args.video_dir, input_filename)
        
        if not os.path.exists(input_path):
            print(f"Warning: File {input_path} not found. Skipping.")
            continue

        # 1. Flip Augmentation
        flip_id = next_id
        flip_filename = f"{flip_id}.mp4"
        flip_path = os.path.join(args.video_dir, flip_filename)
        
        if process_video(input_path, flip_path, transform_type='flip'):
            new_row = row.copy()
            new_row['id'] = flip_id
            # Determine new 'view' (optional, but good for consistent metadata)
            # If original is 'side', flipped is still 'side' but facing other way.
            # If 'original_name' matters, we might want to suffix it.
            new_row['original_name'] = f"flipped_{row['original_name']}"
            new_rows.append(new_row)
            next_id += 1
        
        # 2. Random Augmentation (Perspective + Color)
        aug_id = next_id
        aug_filename = f"{aug_id}.mp4"
        aug_path = os.path.join(args.video_dir, aug_filename)
        
        if process_video(input_path, aug_path, transform_type='augment'):
            new_row = row.copy()
            new_row['id'] = aug_id
            new_row['original_name'] = f"augmented_{row['original_name']}"
            new_rows.append(new_row)
            next_id += 1

    # Append new rows to dataframe
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        final_df = pd.concat([df, new_df], ignore_index=True)
        
        # Save updated metadata
        # Backup original first
        backup_path = args.metadata.replace('.csv', '_backup.csv')
        df.to_csv(backup_path, index=False)
        
        final_df.to_csv(args.metadata, index=False)
        print(f"\nCompleted! Added {len(new_rows)} new videos.")
        print(f"Original metadata backed up to {backup_path}")
        print(f"Updated metadata saved to {args.metadata}")
    else:
        print("\nNo new videos were generated.")

if __name__ == "__main__":
    main()
