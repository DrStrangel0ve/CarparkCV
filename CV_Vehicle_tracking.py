from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Part 2: Run CV algorithm
print("Starting vehicle detection analysis...")

# Load YOLO model (using YOLOv8 if YOLOv12 isn't available yet)
try:
    model = YOLO("yolo12n.pt")  # Try YOLOv12 first
    print("Using YOLOv12 model")
except:
    model = YOLO("yolov8n.pt")  # Fallback to YOLOv8
    print("Using YOLOv8 model (YOLOv12 not available)")

# Define vehicle classes from COCO dataset
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle', 
    5: 'bus',
    7: 'truck'
}

def detect_vehicles_in_video(video_path, output_csv="vehicle_detections.csv"):
    """
    Process video and detect vehicles frame by frame
    Returns DataFrame with time and vehicle detection boolean
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps
    
    print(f"Video: {duration_seconds:.2f} seconds, {fps:.2f} FPS, {total_frames} frames")
    
    # Storage for results
    results_data = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate timestamp for this frame
        timestamp = frame_count / fps
        
        # Run YOLO inference on frame
        results = model(frame, verbose=False)
        
        # Check for vehicles in this frame
        vehicle_detected = False
        detected_vehicles = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Check if it's a vehicle class with good confidence
                    if class_id in VEHICLE_CLASSES and confidence > 0.5:
                        vehicle_detected = True
                        detected_vehicles.append({
                            'class': VEHICLE_CLASSES[class_id],
                            'confidence': confidence
                        })
        
        # Store results
        results_data.append({
            'time_seconds': timestamp,
            'frame_number': frame_count,
            'vehicle_detected': vehicle_detected,
            'num_vehicles': len(detected_vehicles),
            'vehicles': detected_vehicles
        })
        
        # Progress update every 1000 frames
        if frame_count % 1000 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processing: {progress:.1f}% complete")
        
        frame_count += 1
    
    cap.release()
    
    # Convert to DataFrame
    df = pd.DataFrame(results_data)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
    return df

def create_vehicle_timeline_plot(df, output_plot="vehicle_timeline.png"):
    """
    Create a timeline plot showing vehicle detections over time
    """
    plt.figure(figsize=(15, 6))
    
    # Create binary plot
    plt.subplot(2, 1, 1)
    plt.plot(df['time_seconds'], df['vehicle_detected'].astype(int), 'b-', linewidth=1)
    plt.fill_between(df['time_seconds'], df['vehicle_detected'].astype(int), alpha=0.3)
    plt.ylabel('Vehicle Detected')
    plt.title('Vehicle Detection Timeline')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Create vehicle count plot
    plt.subplot(2, 1, 2)
    plt.plot(df['time_seconds'], df['num_vehicles'], 'r-', linewidth=1)
    plt.fill_between(df['time_seconds'], df['num_vehicles'], alpha=0.3, color='red')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Number of Vehicles')
    plt.title('Vehicle Count Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Timeline plot saved to {output_plot}")

def analyze_detection_statistics(df):
    """
    Print summary statistics of vehicle detections
    """
    total_duration = df['time_seconds'].max()
    total_frames = len(df)
    frames_with_vehicles = df['vehicle_detected'].sum()
    percent_with_vehicles = (frames_with_vehicles / total_frames) * 100
    
    print("\n=== VEHICLE DETECTION SUMMARY ===")
    print(f"Total video duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Total frames analyzed: {total_frames}")
    print(f"Frames with vehicles: {frames_with_vehicles}")
    print(f"Percentage of time with vehicles: {percent_with_vehicles:.1f}%")
    print(f"Average vehicles per frame: {df['num_vehicles'].mean():.2f}")
    print(f"Maximum vehicles in single frame: {df['num_vehicles'].max()}")
    
    # Vehicle-free periods
    vehicle_periods = df.groupby((df['vehicle_detected'] != df['vehicle_detected'].shift()).cumsum())
    no_vehicle_periods = vehicle_periods.filter(lambda x: not x['vehicle_detected'].iloc[0])
    
    if len(no_vehicle_periods) > 0:
        longest_gap = no_vehicle_periods.groupby(level=0)['time_seconds'].apply(lambda x: x.max() - x.min()).max()
        print(f"Longest period without vehicles: {longest_gap:.1f} seconds")

# Main execution
if __name__ == "__main__":
    # Run vehicle detection on your video
    video_path = "C:/EPICS/FirstVid.mp4"
    
    print("Analyzing video for vehicle detections...")
    results_df = detect_vehicles_in_video(video_path)
    
    # Create visualizations
    create_vehicle_timeline_plot(results_df)
    
    # Print statistics
    analyze_detection_statistics(results_df)
    
    print("\n=== PART 2 COMPLETE ===")
    print("Next steps for Part 4:")
    print("- Load your distance/time graph data")
    print("- Align timestamps between vehicle detections and distance measurements") 
    print("- Superimpose vehicle detection markers onto distance graph")