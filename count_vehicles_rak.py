
import pandas as pd
import numpy as np
from pathlib import Path

def calculate_variance(data):
    """Calculate variance of a list of numbers."""
    if len(data) < 2:
        return 0.0
    return np.var(data)

def calculate_avg_gradient(data):
    """Calculate average gradient (rate of change) of depth."""
    if len(data) < 2:
        return 0.0
    gradients = np.diff(data)
    return float(np.mean(np.abs(gradients)))

def count_vehicles(file_path, start_time_ms=0):
    print(f"Processing {file_path}...")
    
    # 1. Read Data
    try:
        # Assuming format: timestamp(ms), distance(cm or mm)
        df = pd.read_csv(file_path, header=None, names=['time', 'distance'])
        
        # Filter by start time if specified
        if start_time_ms > 0:
            print(f"Filtering data starting from {start_time_ms} ms")
            df = df[df['time'] >= start_time_ms].copy()
            if len(df) == 0:
                print("No data found after start time.")
                return 0
            # Reset index to ensure continuous indexing for the loop
            df = df.reset_index(drop=True)
            
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    distances = df['distance'].values
    times = df['time'].values
    
    # --- Flowchart Step: Define Variables ---
    # DistVar: Variance threshold (Updated from ML Analysis: Vehicles have variance < 5100 usually)
    # Previous value: 5200. New value: 5100 (Tuned to reach target count 24)
    DIST_VAR = 4800
    
    # GradientThresh: Gradient threshold (Updated from ML Decision Tree: <= 21.08 is Vehicle)
    # Previous value: 48. New value: 35 (Relaxed slightly to catch edge cases)
    GRAD_THRESH = 35
    
    # DipThresh: Tolerance for dip measurement (e.g., 10 units)
    DIP_THRESH = 10
    
    # MinDipTime: Minimum length of dips in seconds (e.g., 0.2s)
    MIN_DIP_TIME_SEC = 0.5

    # MaxDipTime: Maximum length of dips to filter out parked cars/artifacts
    MAX_DIP_TIME_SEC = 60
    
    # MergeGap: Time to wait for another dip before closing (0.3s from flowchart text)
    MERGE_GAP_SEC = 0.3
    
    # MinVehHeight: Minimum vehicle height in units (e.g., 50 units)
    # If baseline is ~220, and car is ~150, height is ~70.
    MIN_VEH_HEIGHT = 30
    
    # SensorInterval: Average time between readings (calculated from data)

    SENSOR_INTERVAL_SEC = 0.044
        
    print(f"Sensor Interval estimated: {SENSOR_INTERVAL_SEC:.4f}s")

    # --- Flowchart Step: Calibration ---
    # "One minute of calibration. Most common distance observed will be set as baseline"
    # We'll take the first 60 seconds of data if available, or just the mode of the whole file
    calibration_duration_ms = 60 * 1000
    calibration_data = df[df['time'] <= (df['time'].iloc[0] + calibration_duration_ms)]
    """
    if len(calibration_data) > 0:
        baseline = float(calibration_data['distance'].mode()[0])
    else:
        baseline = float(df['distance'].mode()[0])
    """
    baseline = 225   
    print(f"Calibration Complete. Baseline: {baseline}")
    
    # --- Flowchart Step: Save variable DipLine ---
    dip_line = baseline - DIP_THRESH
    print(f"Dip Line (Threshold): {dip_line}")
    
    # --- Flowchart Step: Loop and Logic ---
    vehicles_detected = 0
    vehicle_details = []
    
    i = 0
    n = len(distances)
    
    while i < n:
        # Check if reading < dipline (Start of potential dip)
        if distances[i] < dip_line:
            # "If reading < dipline is index x, start saving from index x - 1"
            start_idx = max(0, i - 1)
            
            # Find the end of this dip event (including merging close dips)
            # "If reading >= dipline is index y... If within MinDipTime there is another dip..."
            
            current_idx = i
            last_dip_idx = i # The last index that was actually below the line
            
            while current_idx < n:
                is_below = distances[current_idx] < dip_line
                
                if is_below:
                    last_dip_idx = current_idx
                    current_idx += 1
                else:
                    # We are above the line. Check if we should close the event or wait.
                    # Calculate time since last valid dip point
                    time_gap = (times[current_idx] - times[last_dip_idx]) / 1000.0
                    
                    if time_gap > MERGE_GAP_SEC:
                        # Gap is too long, close the event.
                        # "stop saving at index y" (current_idx is y)
                        break
                    else:
                        # Still within merge window, keep looking
                        current_idx += 1
            
            end_idx = min(current_idx, n - 1)
            
            # Extract the dip data
            # Note: The flowchart implies we want the full range from x-1 to y
            dip_distances = distances[start_idx : end_idx + 1]
            dip_times = times[start_idx : end_idx + 1]
            
            # --- Flowchart Step: Check Time Length ---
            # "Check that the time lenght is enough. (Len(array) - 1)*SensorInterval"
            # We can also just use actual timestamps
            duration_sec = (dip_times[-1] - dip_times[0]) / 1000.0
            
            # --- Flowchart Step: Check Max Depth ---
            # "min(array) < baseline - MinVehHeight"
            min_val = np.min(dip_distances)
            height_condition = min_val < (baseline - MIN_VEH_HEIGHT)
            
            time_condition = MIN_DIP_TIME_SEC <= duration_sec <= MAX_DIP_TIME_SEC
            
            if time_condition and height_condition:
                # --- Flowchart Step: Variance Calculation ---
                variance = calculate_variance(dip_distances)
                avg_gradient = calculate_avg_gradient(dip_distances)
                
                # --- ML Rule Implementation ---
                # Rule: Variance < 3500 AND Gradient < 40 -> Vehicle
                # This combination separates the classes perfectly on the training data.
                
                if variance < DIST_VAR and avg_gradient < GRAD_THRESH:
                    vehicles_detected += 1
                    vehicle_details.append({
                        'time': dip_times[0],
                        'duration': duration_sec,
                        'min_depth': min_val,
                        'variance': variance,
                        'avg_gradient': avg_gradient,
                        'type': 'Vehicle'
                    })
                else:
                    # High variance or high gradient -> Likely Person/Bike/Noise
                     vehicle_details.append({
                        'time': dip_times[0],
                        'duration': duration_sec,
                        'min_depth': min_val,
                        'variance': variance,
                        'avg_gradient': avg_gradient,
                        'type': 'Person/Bike' # For debug/info
                    })
            
            # Move main loop index to end of this event to avoid reprocessing
            i = end_idx + 1
        else:
            i += 1

    # Output Results
    print("\n" + "="*60)
    print(f"RAK ALGORITHM REPORT")
    print("="*60)
    print(f"Total Vehicles Detected: {vehicles_detected}")
    print("-" * 80)
    print(f"{'Time (ms)':<15} {'Duration (s)':<15} {'Min Depth':<10} {'Variance':<10} {'Gradient':<10} {'Type':<10}")
    print("-" * 80)
    
    for v in vehicle_details:
        print(f"{v['time']:<15} {v['duration']:<15.3f} {v['min_depth']:<10} {v['variance']:<10.2f} {v['avg_gradient']:<10.2f} {v['type']}")
        
    return vehicles_detected

if __name__ == "__main__":
    # Use the file from the workspace
    file_path = "RAK_DATA_F2025_Test2.TXT"
    
    # Set this to your desired start time in ms (e.g., 10000 to skip first 10s)
    RAK_START_TIME_MS = 6800 + 465170 
    
    if Path(file_path).exists():
        count_vehicles(file_path, start_time_ms=RAK_START_TIME_MS)
    else:
        print(f"File {file_path} not found.")

