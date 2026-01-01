import os
import csv
import shutil

# Configuration
DATA_FOLDER = '' # Thư mục gốc chứa video .mov của TDTU Golf Pose Dataset
OUTPUT_VIDEO_FOLDER = '' # Thư mục đích để lưu video đã tổ chức
OUTPUT_CSV = '' # Đường dẫn tới file output CSV metadata

def extract_metadata(folder_path, filename):
    """Extract place, band, and view from folder structure and filename"""
    parts = folder_path.split(os.sep)
    
    # Extract place (Indoor/Outdoor)
    place = None
    band = None
    for part in parts:
        if 'Indoor' in part or 'Trong nhà' in part:
            place = 'indoor'
        elif 'Outdoor' in part or 'Ngoài trời' in part:
            place = 'outdoor'
        
        # Extract band
        if any(b in part for b in ['1-2', '2-4', '4-6', '6-8', '8-10']):
            band = part.strip()
    
    # Extract view from filename (back/side)
    view = None
    filename_lower = filename.lower()
    if 'backside' in filename_lower:
        view = 'backside'
    elif 'side' in filename_lower:
        view = 'side'
    
    return place, band, view

def main():
    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_VIDEO_FOLDER):
        os.makedirs(OUTPUT_VIDEO_FOLDER)
    
    # Initialize CSV data
    csv_data = []
    video_counter = 0
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            if file.endswith('.mov'):
                video_counter += 1
                video_id = video_counter
                
                # Get metadata
                place, band, view = extract_metadata(root, file)
                
                # Source and destination paths
                src_path = os.path.join(root, file)
                dst_filename = f"{video_id}.mp4"
                dst_path = os.path.join(OUTPUT_VIDEO_FOLDER, dst_filename)
                
                # Copy video
                shutil.copy2(src_path, dst_path)
                print(f"✓ Copied: {dst_filename}")
                
                # Add to CSV data
                csv_data.append({
                    'id': video_id,
                    'place': place,
                    'band': band,
                    'view': view,
                    'original_name': file
                })
    
    # Write CSV file
    if csv_data:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'place', 'band', 'view', 'original_name'])
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"\n✓ Created: {OUTPUT_CSV} with {len(csv_data)} videos")
    else:
        print("No MP4 files found!")

if __name__ == "__main__":
    main()