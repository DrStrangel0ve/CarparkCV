import cv2
import subprocess
import os
from pathlib import Path


def select_crop_roi(video_path):
    """
    Allow user to select the crop region of interest
    Returns the ROI coordinates (x, y, width, height), or None to skip cropping
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read video frame")
        return None
    
    skip = input("Do you want to crop the video? (y/n): ").strip().lower()
    if skip == 'n':
        print("✓ Skipping crop - full video will be used")
        return (0, 0, frame.shape[1], frame.shape[0])  # Return full frame dimensions
    
    print(f"\nFrame size: {frame.shape[1]}x{frame.shape[0]} pixels")
    print("Select the crop area by clicking and dragging to create a rectangle")
    print("Press ENTER when satisfied, ESC to cancel")
    print()
    
    # Create a named window
    cv2.namedWindow("Select Crop Area", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Crop Area", 1280, 720)
    
    # Show the frame for user to select ROI
    roi = cv2.selectROI("Select Crop Area", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    
    if roi[2] > 0 and roi[3] > 0:  # Valid selection
        print(f"\n✓ Selected crop region:")
        print(f"  X: {roi[0]}, Y: {roi[1]}")
        print(f"  Width: {roi[2]}, Height: {roi[3]}")
        return roi
    else:
        print("✗ No valid crop region selected")
        return None


def get_output_filename(video_path):
    """
    Ask user for output filename
    Returns full path with appropriate extension
    """
    video_dir = os.path.dirname(video_path)
    
    print(f"\nOutput will be saved in: {video_dir}")
    filename = input("Enter output filename (without extension): ").strip()
    
    if not filename:
        print("✗ No filename provided")
        return None
    
    # Use .mp4 as default extension
    output_path = os.path.join(video_dir, f"{filename}.mp4")
    
    # Check if file already exists
    if os.path.exists(output_path):
        overwrite = input(f"File '{filename}.mp4' already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("✗ Operation cancelled")
            return None
    
    return output_path


def get_video_info(video_path):
    """
    Get video information using ffprobe
    Returns fps, duration, width, height
    """
    try:
        # Use ffprobe to get video information
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,duration,width,height',
            '-of', 'default=noprint_wrappers=1:nokey=1:value_only=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        
        # Parse frame rate (it's in format like 30/1)
        fps_str = lines[0]
        if '/' in fps_str:
            num, denom = map(int, fps_str.split('/'))
            fps = num / denom
        else:
            fps = float(fps_str)
        
        duration = float(lines[1])
        width = int(lines[2])
        height = int(lines[3])
        
        return fps, duration, width, height
    except Exception as e:
        print(f"Warning: Could not get video info with ffprobe: {e}")
        return None, None, None, None


def crop_video_ffmpeg(video_path, output_path, roi_coords):
    """
    Crop video using ffmpeg
    roi_coords: (x, y, width, height)
    Returns True if successful, False otherwise
    """
    x, y, w, h = roi_coords
    
    # Ensure width and height are even (required for many codecs)
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1
    
    print(f"\n{'='*60}")
    print("FFMPEG PROCESSING")
    print(f"{'='*60}")
    print(f"Input video:  {video_path}")
    print(f"Output video: {output_path}")
    
    # Check if we're actually cropping or just re-encoding
    is_cropping = x > 0 or y > 0 or (w > 0 and h > 0)
    
    if is_cropping:
        print(f"Crop region:  x={x}, y={y}, width={w}, height={h}")
    else:
        print("Mode: Full video (no crop)")
    print()
    
    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-i', video_path,
    ]
    
    # Only add crop filter if cropping is needed
    if is_cropping:
        cmd.extend(['-vf', f'crop={w}:{h}:{x}:{y}'])
    
    cmd.extend([
        '-c:v', 'libx264',          # Video codec
        '-preset', 'medium',         # Encoding speed (fast, medium, slow)
        '-crf', '23',                # Quality (0-51, lower is better)
        '-c:a', 'aac',               # Audio codec
        '-b:a', '128k',              # Audio bitrate
        output_path
    ])
    
    try:
        print("Starting operation...")
        print("This may take a while depending on video length...\n")
        
        # Run ffmpeg
        process = subprocess.run(cmd, check=True, capture_output=False)
        
        print("\n✓ Video processed successfully!")
        print(f"✓ Output saved to: {output_path}")
        
        # Get output file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ File size: {file_size_mb:.2f} MB")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during processing: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


def main():
    """
    Main program flow
    """
    print("\n" + "="*60)
    print("VIDEO CROP TOOL")
    print("="*60)
    print("Select a region in a video and crop it using ffmpeg")
    print()
    
    # Get video file from user
    video_path = input("Enter path to video file: ").strip().strip('"')
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"✗ File not found: {video_path}")
        return
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except FileNotFoundError:
        print("✗ Error: ffmpeg not found. Please install ffmpeg and add it to your PATH")
        return
    except Exception as e:
        print(f"✗ Error checking ffmpeg: {e}")
        return
    
    print(f"✓ Found video: {os.path.basename(video_path)}")
    
    # Get video info
    fps, duration, width, height = get_video_info(video_path)
    if fps:
        print(f"  Resolution: {width}x{height} pixels")
        print(f"  Frame rate: {fps:.2f} fps")
        print(f"  Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    
    # Let user select crop region
    roi_coords = select_crop_roi(video_path)
    if roi_coords is None:
        return
    
    # Get output filename
    output_path = get_output_filename(video_path)
    if output_path is None:
        return
    
    # Crop the video
    success = crop_video_ffmpeg(video_path, output_path, roi_coords)
    
    if success:
        print("\n" + "="*60)
        print("OPERATION COMPLETE")
        print("="*60)
    else:
        print("\n✗ Cropping failed. Check the error messages above.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Operation cancelled by user")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
