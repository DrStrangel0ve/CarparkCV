from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import torch

# Part 2: Run CV algorithm
print("Starting vehicle detection analysis...")

# Load YOLO model (using YOLOv8 if YOLOv12 isn't available yet)
try:
    model = YOLO("yolo12n.pt")  # Try YOLOv12 first
    print("Using YOLOv12 model")
except:
    model = YOLO("yolov8n.pt")  # Fallback to YOLOv8
    print("Using YOLOv8 model (YOLOv12 not available)")

# Enable GPU acceleration if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
model.to(device)

# Define detection classes from COCO dataset
DETECTION_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
}

def select_roi(video_path):
    """
    Allow user to select the carpark region of interest
    Returns the ROI coordinates
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read video frame")
        return None
    
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]} pixels")
    print("Select the carpark area by clicking and dragging to create a rectangle")
    print("Press ENTER when satisfied, ESC to cancel")
    print("The window will show the FULL frame - you can select anywhere within it")
    
    # Create a named window and set it to normal (resizable)
    cv2.namedWindow("Select Carpark Area", cv2.WINDOW_NORMAL)
    
    # Show the full frame first so user can see what they're selecting from
    cv2.imshow("Select Carpark Area", frame)
    cv2.waitKey(1000)  # Show for 1 second
    
    # Now select ROI - showCrosshair=True, fromCenter=False for easier selection
    roi = cv2.selectROI("Select Carpark Area", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    
    if roi[2] > 0 and roi[3] > 0:  # Valid selection
        print(f"Selected ROI: x={roi[0]}, y={roi[1]}, width={roi[2]}, height={roi[3]}")
        return roi
    else:
        print("No valid ROI selected")
        return None

def create_roi_mask(frame_shape, roi_coords):
    """
    Create a binary mask for the region of interest
    """
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    x, y, w, h = roi_coords
    mask[y:y+h, x:x+w] = 255
    return mask

def is_center_in_roi(box_coords, roi_coords):
    """
    Check if the center of a bounding box is within the ROI
    """
    x1, y1, x2, y2 = box_coords
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    roi_x, roi_y, roi_w, roi_h = roi_coords
    
    return (roi_x <= center_x <= roi_x + roi_w and 
            roi_y <= center_y <= roi_y + roi_h)

def calculate_box_similarity(box1, box2):
    """
    Calculate similarity between two bounding boxes based on position and size
    Returns a value between 0 (completely different) and 1 (identical)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate centers
    center1_x = (x1_1 + x2_1) / 2
    center1_y = (y1_1 + y2_1) / 2
    center2_x = (x1_2 + x2_2) / 2
    center2_y = (y1_2 + y2_2) / 2
    
    # Calculate areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Distance between centers
    center_distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    # Area similarity (ratio of smaller to larger)
    area_similarity = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
    
    # Normalize distance based on box size
    avg_box_size = np.sqrt((area1 + area2) / 2)
    normalized_distance = center_distance / avg_box_size if avg_box_size > 0 else float('inf')
    
    # Distance similarity (closer = more similar)
    distance_similarity = max(0, 1 - normalized_distance / 2)  # Threshold at 2x box size
    
    # Combined similarity score
    similarity = (area_similarity * 0.4) + (distance_similarity * 0.6)
    
    return similarity

class ObjectTracker:
    """
    Persistent object tracker to prevent double counting (vehicles, people, bicycles)
    
    This tracker maintains the identity of objects across frames and only counts
    them once when they first enter the ROI. It remembers objects even if they
    temporarily leave the frame and reappear.
    """
    def __init__(self, retention_frames=30, similarity_threshold=0.7):
        # Persistent tracking data: {object_id: {'bbox': bbox, 'class': class, 'last_frame': frame_num, 'counted': bool, 'matches': count}}
        self.active_objects = {}  
        self.retention_frames = retention_frames  # How many frames to remember
        self.similarity_threshold = similarity_threshold
        self.next_id = 1
        self.object_counts = {'person': 0, 'bicycle': 0, 'car': 0}  # Total counts
        
    def update(self, current_frame, detected_objects):
        """
        Update tracker with new detections
        Returns: {
            'all_detections': list of all currently detected objects with IDs,
            'newly_counted': list of objects counted for the first time this frame,
            'counts': {'person': int, 'bicycle': int, 'car': int}
        }
        """
        # Clean old entries (objects not seen for retention_frames)
        expired_ids = [obj_id for obj_id, obj_data in self.active_objects.items()
                      if current_frame - obj_data['last_frame'] > self.retention_frames]
        for obj_id in expired_ids:
            del self.active_objects[obj_id]
        
        all_detections = []
        newly_counted = []
        
        # Track which active objects were matched in this frame
        matched_ids = set()
        
        for detected_obj in detected_objects:
            bbox = detected_obj['bbox']
            obj_class = detected_obj['class']
            best_match_id = None
            best_similarity = self.similarity_threshold
            
            # Find the best matching tracked object
            for tracked_id, tracked_data in self.active_objects.items():
                tracked_bbox = tracked_data['bbox']
                similarity = calculate_box_similarity(bbox, tracked_bbox)
                
                # Better match found
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = tracked_id
            
            if best_match_id is not None:
                # Update existing tracked object
                object_id = best_match_id
                matched_ids.add(object_id)
                self.active_objects[object_id]['bbox'] = bbox
                self.active_objects[object_id]['last_frame'] = current_frame
                self.active_objects[object_id]['matches'] += 1
            else:
                # New object detected
                object_id = self.next_id
                self.next_id += 1
                matched_ids.add(object_id)
                
                # Initialize new tracked object
                self.active_objects[object_id] = {
                    'bbox': bbox,
                    'class': obj_class,
                    'first_frame': current_frame,
                    'last_frame': current_frame,
                    'counted': False,
                    'matches': 1
                }
            
            # Create detection info
            detection_info = detected_obj.copy()
            detection_info['object_id'] = object_id
            detection_info['is_new'] = not self.active_objects[object_id]['counted']
            all_detections.append(detection_info)
            
            # Count the object only once (on first detection)
            if not self.active_objects[object_id]['counted']:
                self.active_objects[object_id]['counted'] = True
                self.object_counts[obj_class] += 1
                newly_counted.append(detection_info)
        
        return {
            'all_detections': all_detections,
            'newly_counted': newly_counted,
            'counts': self.object_counts.copy(),
            'total': sum(self.object_counts.values())
        }
    
    def get_counts(self):
        """Get current object counts"""
        return {
            'people': self.object_counts['person'],
            'bicycles': self.object_counts['bicycle'],
            'vehicles': self.object_counts['car'],
            'total': sum(self.object_counts.values())
        }

def detect_objects_in_video(video_path, output_csv="detection_results.csv", show_realtime=True, roi_coords=None, start_time_seconds=0):
    """
    Process video and detect objects (people, bicycles, vehicles) frame by frame
    If show_realtime=True, displays video with bounding boxes in real-time
    If roi_coords is provided, only count objects within that region
    start_time_seconds: Start processing from this timestamp (in seconds)
    Returns DataFrame with time and detection counts
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps
    
    print(f"Video: {duration_seconds:.2f} seconds, {fps:.2f} FPS, {total_frames} frames")
    
    # Set start position if specified
    if start_time_seconds > 0:
        start_frame = int(start_time_seconds * fps)
        if start_frame >= total_frames:
            print(f"Error: Start time {start_time_seconds}s is beyond video duration {duration_seconds:.2f}s")
            cap.release()
            return None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Starting from {start_time_seconds}s (frame {start_frame})")
        frame_count = start_frame
    else:
        frame_count = 0
    
    # Get ROI if not provided
    if roi_coords is None:
        print("No ROI specified. Selecting carpark area...")
        roi_coords = select_roi(video_path)
        if roi_coords is None:
            print("No ROI selected. Processing entire frame.")
    
    if roi_coords:
        print(f"ROI selected: x={roi_coords[0]}, y={roi_coords[1]}, w={roi_coords[2]}, h={roi_coords[3]}")
    
    if show_realtime:
        print("Press 'q' to quit, 'p' to pause/resume, 's' to save current frame, 'r' to reselect ROI")
        # Create window for display
        cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Object Detection', 1280, 720)
        paused = False
    
    # Storage for results
    results_data = []
    
    # Initialize object tracker
    object_tracker = ObjectTracker(retention_frames=int(fps * 2), similarity_threshold=0.6)  # 2 second retention
    
    while True:
        if not paused or not show_realtime:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Calculate timestamp for this frame
            timestamp = frame_count / fps
            
            # Run YOLO inference on frame
            results = model(frame, verbose=False)
            
            # Check for objects in this frame and draw bounding boxes
            detected_objects = []
            display_frame = frame.copy()
            
            # Apply ROI mask if specified (black out background)
            if roi_coords and show_realtime:
                roi_mask = create_roi_mask(frame.shape, roi_coords)
                # Black out everything outside ROI
                display_frame[roi_mask == 0] = 0
                # Draw ROI boundary
                x, y, w, h = roi_coords
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
                cv2.putText(display_frame, "CARPARK AREA", (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Check if it's a detected class with good confidence
                        if class_id in DETECTION_CLASSES and confidence > 0.5:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Check if object is in ROI (if ROI is specified)
                            in_roi = True
                            if roi_coords:
                                in_roi = is_center_in_roi((x1, y1, x2, y2), roi_coords)
                            
                            # Only count objects in ROI
                            if in_roi:
                                detected_objects.append({
                                    'class': DETECTION_CLASSES[class_id],
                                    'confidence': confidence,
                                    'bbox': (x1, y1, x2, y2)
                                })
                            
                            # Draw red bounding box for objects outside ROI (for reference)
                            elif show_realtime and roi_coords:
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                                cv2.putText(display_frame, "IGNORED", (x1, y1 - 5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Process detections through tracker
            tracker_result = object_tracker.update(frame_count, detected_objects)
            all_objects = tracker_result['all_detections']
            newly_counted = tracker_result['newly_counted']
            cumulative_counts = tracker_result['counts']
            
            # Current frame counts (visible objects in this frame)
            people_count = len([obj for obj in all_objects if obj['class'] == 'person'])
            bicycle_count = len([obj for obj in all_objects if obj['class'] == 'bicycle'])
            vehicle_count = len([obj for obj in all_objects if obj['class'] == 'car'])
            total_count = len(all_objects)
            
            # Draw bounding boxes for all currently visible objects
            if show_realtime:
                for obj in all_objects:
                    x1, y1, x2, y2 = obj['bbox']
                    object_id = obj['object_id']
                    obj_class = obj['class']
                    is_new = obj.get('is_new', False)
                    
                    # Color by class: green for vehicles, blue for people, purple for bicycles
                    if obj_class == 'car':
                        color = (0, 255, 0)
                    elif obj_class == 'person':
                        color = (255, 0, 0)
                    elif obj_class == 'bicycle':
                        color = (255, 0, 255)
                    else:
                        color = (255, 255, 255)
                    
                    # Use thicker lines if this is a newly counted object
                    line_thickness = 3 if is_new else 2
                    
                    # Draw bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, line_thickness)
                    
                    # Add label with object ID
                    label = f"ID:{object_id} {obj_class} {obj['confidence']:.2f}"
                    if is_new:
                        label += " [NEW]"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(display_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Add frame info overlay with counts
            if show_realtime:
                info_text = f"Frame: {frame_count}/{total_frames} | Time: {timestamp:.1f}s"
                cv2.putText(display_frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                
                # Current frame count overlay
                frame_count_text = f"In Frame: People: {people_count} | Bicycles: {bicycle_count} | Vehicles: {vehicle_count} | Total: {total_count}"
                cv2.putText(display_frame, frame_count_text, (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, frame_count_text, (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                
                # Cumulative count overlay
                cumulative_text = f"TOTAL COUNTED: People: {cumulative_counts['person']} | Bicycles: {cumulative_counts['bicycle']} | Vehicles: {cumulative_counts['car']} | Total: {cumulative_counts['person'] + cumulative_counts['bicycle'] + cumulative_counts['car']}"
                cv2.putText(display_frame, cumulative_text, (10, 110), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display_frame, cumulative_text, (10, 110), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            
            # Store results
            results_data.append({
                'time_seconds': timestamp,
                'frame_number': frame_count,
                'people_in_frame': people_count,
                'bicycles_in_frame': bicycle_count,
                'vehicles_in_frame': vehicle_count,
                'total_in_frame': total_count,
                'total_people_counted': cumulative_counts['person'],
                'total_bicycles_counted': cumulative_counts['bicycle'],
                'total_vehicles_counted': cumulative_counts['car'],
                'total_objects_counted': cumulative_counts['person'] + cumulative_counts['bicycle'] + cumulative_counts['car'],
                'newly_counted_this_frame': len(newly_counted),
                'detected_objects': all_objects,
                'total_detections': len(detected_objects)
            })
            
            frame_count += 1
        
        # Display frame if real-time mode
        if show_realtime:
            cv2.imshow('Object Detection', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('s'):
                # Save current frame
                save_filename = f"detected_frame_{frame_count:06d}.jpg"
                cv2.imwrite(save_filename, display_frame)
                print(f"Frame saved as {save_filename}")
            elif key == ord('r'):
                # Reselect ROI
                print("Reselecting ROI...")
                cv2.destroyAllWindows()
                roi_coords = select_roi(video_path)
                if roi_coords:
                    print(f"New ROI selected: x={roi_coords[0]}, y={roi_coords[1]}, w={roi_coords[2]}, h={roi_coords[3]}")
                else:
                    print("No ROI selected - will process entire frame")
                cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Object Detection', 1280, 720)
        else:
            # Progress update every 1000 frames for background processing
            if frame_count % 1000 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Processing: {progress:.1f}% complete")
    
    cap.release()
    if show_realtime:
        cv2.destroyAllWindows()
    
    # Convert to DataFrame
    df = pd.DataFrame(results_data)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
    return df

def create_detection_timeline_plot(df, output_plot="detection_timeline.png"):
    """
    Create a timeline plot showing cumulative object counts and frame-by-frame detections
    """
    plt.figure(figsize=(15, 10))
    
    # Create cumulative people count plot
    plt.subplot(3, 2, 1)
    plt.plot(df['time_seconds'], df['total_people_counted'], 'b-', linewidth=2, label='Cumulative People')
    plt.fill_between(df['time_seconds'], df['total_people_counted'], alpha=0.3, color='blue')
    plt.ylabel('Total People Counted')
    plt.title('Cumulative Object Counts (Each object counted once)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Create cumulative bicycle count plot
    plt.subplot(3, 2, 3)
    plt.plot(df['time_seconds'], df['total_bicycles_counted'], 'm-', linewidth=2, label='Cumulative Bicycles')
    plt.fill_between(df['time_seconds'], df['total_bicycles_counted'], alpha=0.3, color='magenta')
    plt.ylabel('Total Bicycles Counted')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Create cumulative vehicle count plot
    plt.subplot(3, 2, 5)
    plt.plot(df['time_seconds'], df['total_vehicles_counted'], 'g-', linewidth=2, label='Cumulative Vehicles')
    plt.fill_between(df['time_seconds'], df['total_vehicles_counted'], alpha=0.3, color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Total Vehicles Counted')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Frame-by-frame people count
    plt.subplot(3, 2, 2)
    plt.plot(df['time_seconds'], df['people_in_frame'], 'b-', linewidth=1, alpha=0.7, label='People in Frame')
    plt.fill_between(df['time_seconds'], df['people_in_frame'], alpha=0.2, color='blue')
    plt.ylabel('People Visible')
    plt.title('Objects Visible Per Frame')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Frame-by-frame bicycle count
    plt.subplot(3, 2, 4)
    plt.plot(df['time_seconds'], df['bicycles_in_frame'], 'm-', linewidth=1, alpha=0.7, label='Bicycles in Frame')
    plt.fill_between(df['time_seconds'], df['bicycles_in_frame'], alpha=0.2, color='magenta')
    plt.ylabel('Bicycles Visible')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Frame-by-frame vehicle count
    plt.subplot(3, 2, 6)
    plt.plot(df['time_seconds'], df['vehicles_in_frame'], 'g-', linewidth=1, alpha=0.7, label='Vehicles in Frame')
    plt.fill_between(df['time_seconds'], df['vehicles_in_frame'], alpha=0.2, color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Vehicles Visible')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Timeline plot saved to {output_plot}")

def analyze_detection_statistics(df):
    """
    Print summary statistics of object detections
    """
    total_duration = df['time_seconds'].max()
    total_frames = len(df)
    
    # Get final counts
    final_people = df['total_people_counted'].iloc[-1]
    final_bicycles = df['total_bicycles_counted'].iloc[-1]
    final_vehicles = df['total_vehicles_counted'].iloc[-1]
    total_counted = final_people + final_bicycles + final_vehicles
    
    print("\n=== OBJECT DETECTION SUMMARY ===")
    print(f"Total video duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Total frames analyzed: {total_frames}")
    print()
    print("CUMULATIVE COUNTS (Each object counted once):")
    print(f"  Total People: {int(final_people)}")
    print(f"  Total Bicycles: {int(final_bicycles)}")
    print(f"  Total Vehicles: {int(final_vehicles)}")
    print(f"  TOTAL OBJECTS: {int(total_counted)}")
    print()
    
    # People statistics
    frames_with_people = (df['people_in_frame'] > 0).sum()
    max_people = df['people_in_frame'].max()
    avg_people = df['people_in_frame'].mean()
    print(f"PEOPLE VISIBILITY:")
    print(f"  Frames with people visible: {frames_with_people} ({(frames_with_people/total_frames)*100:.1f}%)")
    print(f"  Average people visible per frame: {avg_people:.2f}")
    print(f"  Maximum people in single frame: {int(max_people)}")
    
    # Bicycle statistics
    frames_with_bicycles = (df['bicycles_in_frame'] > 0).sum()
    max_bicycles = df['bicycles_in_frame'].max()
    avg_bicycles = df['bicycles_in_frame'].mean()
    print(f"\nBICYCLES VISIBILITY:")
    print(f"  Frames with bicycles visible: {frames_with_bicycles} ({(frames_with_bicycles/total_frames)*100:.1f}%)")
    print(f"  Average bicycles visible per frame: {avg_bicycles:.2f}")
    print(f"  Maximum bicycles in single frame: {int(max_bicycles)}")
    
    # Vehicle statistics
    frames_with_vehicles = (df['vehicles_in_frame'] > 0).sum()
    max_vehicles = df['vehicles_in_frame'].max()
    avg_vehicles = df['vehicles_in_frame'].mean()
    print(f"\nVEHICLES VISIBILITY:")
    print(f"  Frames with vehicles visible: {frames_with_vehicles} ({(frames_with_vehicles/total_frames)*100:.1f}%)")
    print(f"  Average vehicles visible per frame: {avg_vehicles:.2f}")
    print(f"  Maximum vehicles in single frame: {int(max_vehicles)}")
    
    # Total statistics
    frames_with_objects = (df['total_in_frame'] > 0).sum()
    print(f"\nVISIBILITY TOTALS:")
    print(f"  Frames with any objects visible: {frames_with_objects} ({(frames_with_objects/total_frames)*100:.1f}%)")
    print(f"  Average total objects visible per frame: {df['total_in_frame'].mean():.2f}")
    print(f"  Maximum objects in single frame: {int(df['total_in_frame'].max())}")

# Main execution
if __name__ == "__main__":
    # Run object detection on your video
    video_path = "C:\\Users\\Arnav\\Footage2.MOV"
    
    print("Starting real-time object detection (people, bicycles, vehicles)...")
    print("First, select the carpark area to monitor...")
    print()
    
    # Get ROI selection first
    roi_coords = select_roi(video_path)
    
    if roi_coords:
        print("Carpark area selected successfully!")
        print("Controls during playback:")
        print("- Press 'q' to quit")
        print("- Press 'p' to pause/resume")
        print("- Press 's' to save current frame")
        print("- Press 'r' to reselect carpark area")
        print()
        
        # Run with real-time display and ROI starting at 3:30
        results_df = detect_objects_in_video(video_path, output_csv="detection_results.csv", show_realtime=True, roi_coords=roi_coords, start_time_seconds=0)
    else:
        print("No carpark area selected. Processing entire frame...")
        results_df = detect_objects_in_video(video_path, output_csv="detection_results.csv", show_realtime=True, start_time_seconds=0)
    
    # After real-time analysis, ask if user wants to create additional visualizations
    print("\nReal-time analysis complete!")
    
    create_plots = input("Would you like to create timeline plots? (y/n): ").lower().strip()
    if create_plots == 'y':
        create_detection_timeline_plot(results_df)
    
    # Print statistics
    analyze_detection_statistics(results_df)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Detection data saved to detection_results.csv")
    print("Next steps for Part 4:")
    print("- Load your distance/time graph data")
    print("- Align timestamps between detections and distance measurements") 
    print("- Superimpose detection markers onto distance graph")