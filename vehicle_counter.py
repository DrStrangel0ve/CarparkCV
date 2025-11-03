import cv2
import glob
from ultralytics import YOLO
import supervision as sv
import numpy as np
import subprocess  # To run FFMPEG
import os          # To handle file paths

# --- CONFIGURATION ---
# Path to your video files using a wildcard to get all MTS files
VIDEO_FILES_PATH = r"C:\EPICS\*.MTS" 

# Define the coordinates for the counting line.
# You will need to adjust these coordinates to fit your video frame.
# A vertical line in the middle of the entrance works well.
# Format is (x1, y1) and (x2, y2)
LINE_START = sv.Point(900, 250)
LINE_END = sv.Point(900, 750)

# --- HELPER FUNCTION ---

def convert_mts_to_mp4(input_path):
    """
    Converts a video file from .MTS to .MP4 using FFMPEG.
    Assumes FFMPEG is installed and in the system's PATH.

    Args:
        input_path (str): The full path to the input .MTS file.

    Returns:
        str: The path to the converted .MP4 file, or None if conversion failed.
    """
    # Create the output path by changing the file extension to .mp4
    output_path = os.path.splitext(input_path)[0] + ".mp4"
    
    # Optional: Skip conversion if the MP4 file already exists
    if os.path.exists(output_path):
        print(f"'{os.path.basename(output_path)}' already exists. Skipping conversion.")
        return output_path

    print(f"--- Converting '{os.path.basename(input_path)}' to MP4 ---")
    
    # Construct the FFMPEG command:
    # -i: specifies the input file
    # -c:v copy: copies the video stream without re-encoding (fast and lossless quality)
    # -c:a aac: re-encodes the audio to AAC, a standard for MP4 containers
    # -loglevel error: suppresses all non-essential output from FFMPEG
    # -y: automatically overwrite the output file if it exists
    command = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-loglevel', 'error',
        '-y',
        output_path
    ]
    
    try:
        # Execute the FFMPEG command
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("Conversion successful.")
        return output_path
    except FileNotFoundError:
        print("🛑 Error: FFMPEG not found. Please ensure it is installed and in your system's PATH.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"🛑 Error during FFMPEG conversion for {input_path}:")
        print(f"FFMPEG stderr: {e.stderr}")
        return None

# --- SCRIPT ---

def main():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Get a list of all MTS video files
    mts_files = glob.glob(VIDEO_FILES_PATH)
    if not mts_files:
        print(f"No MTS files found at path: {VIDEO_FILES_PATH}")
        return
    
    print(f"Found {len(mts_files)} MTS videos to process.")

    # --- Convert all MTS files to MP4 before processing ---
    mp4_files_to_process = []
    for mts_path in mts_files:
        mp4_path = convert_mts_to_mp4(mts_path)
        if mp4_path:
            mp4_files_to_process.append(mp4_path)
        else:
            print(f"Skipping file {mts_path} due to conversion error.")
    
    # Exit if no files were successfully converted
    if not mp4_files_to_process:
        print("No video files to process after conversion step.")
        return
    
    print(f"\n✅ Successfully prepared {len(mp4_files_to_process)} MP4 files. Starting detection...")
    
    # Instantiate the line zone counter
    line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
    
    # Instantiate the annotators for drawing
    box_annotator = sv.BoxAnnotator(
        thickness=2
    )
    line_annotator = sv.LineZoneAnnotator(
        thickness=2
    )

    # Process each CONVERTED video file
    for video_path in mp4_files_to_process:
        print(f"--- Processing video: {os.path.basename(video_path)} ---")
        
        # Use model.track as a generator for efficient video processing
        for results in model.track(source=video_path, stream=True, persist=True, classes=[2]):
            frame = results.orig_img

            if results.boxes.id is not None:
                detections = sv.Detections.from_yolov8(results)
                line_zone.trigger(detections=detections)
            else:
                detections = sv.Detections.empty()
            
            # --- VISUALIZATION ---
            labels = [
                f"{model.names[int(class_id)]} #{tracker_id}"
                for _, _, confidence, class_id, tracker_id in detections
            ]

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(), 
                detections=detections, 
                labels=labels
            )

            line_annotator.annotate(
                frame=annotated_frame, 
                line_counter=line_zone
            )
            
            cv2.imshow("Vehicle Counter", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    cv2.destroyAllWindows()
    print("--- Processing complete ---")
    print(f"Total cars counted entering: {line_zone.in_count}")


if __name__ == "__main__":
    main()