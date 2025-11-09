"""
RAK Data Processing - Extract ultrasonic dip features
"""

import numpy as np
import pandas as pd
from scipy import signal

def read_rak_data(file_path):
    """Read RAK data file and return time-distance pairs"""
    out = []
    """Read RAK data file and print each line with line number"""
    try:
        with open(file_path, 'r') as file:
            line_num = 1
            for line in file:
                # Strip trailing newline/whitespace but preserve content
                line_content = line.rstrip('\n\r')
                #split line content by commas and store in out list
                time, data = line_content.split(',')
                out.append([int(time), int(data)])
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")

    return out


def find_dips(timearr, threshold=200, min_frames=5, max_frames=200):
    """
    Find dips in ultrasonic data where value < threshold.
    
    Parameters:
    - timearr: list of [timestamp, distance] pairs
    - threshold: distance threshold (dip starts/ends at this value)
    - min_frames: minimum dip duration (frames) to keep
    - max_frames: maximum dip duration (frames); longer dips are discarded
    
    Returns:
    - dips: list of dip segments, each as [start_idx, end_idx, values, timestamps]
    """
    dips = []
    in_dip = False
    dip_start = None
    
    for i, (timestamp, distance) in enumerate(timearr):
        if distance < threshold and not in_dip:
            # Start of dip
            in_dip = True
            dip_start = i
        elif distance >= threshold and in_dip:
            # End of dip
            dip_end = i - 1
            dip_length = dip_end - dip_start + 1
            
            # Filter by length
            if min_frames <= dip_length <= max_frames:
                dip_values = [timearr[j][1] for j in range(dip_start, dip_end + 1)]
                dip_timestamps = [timearr[j][0] for j in range(dip_start, dip_end + 1)]
                dips.append({
                    'start_idx': dip_start,
                    'end_idx': dip_end,
                    'values': dip_values,
                    'timestamps': dip_timestamps
                })
            in_dip = False
    
    # Handle dip that extends to end of data
    if in_dip:
        dip_end = len(timearr) - 1
        dip_length = dip_end - dip_start + 1
        if min_frames <= dip_length <= max_frames:
            dip_values = [timearr[j][1] for j in range(dip_start, dip_end + 1)]
            dip_timestamps = [timearr[j][0] for j in range(dip_start, dip_end + 1)]
            dips.append({
                'start_idx': dip_start,
                'end_idx': dip_end,
                'values': dip_values,
                'timestamps': dip_timestamps
            })
    
    return dips


def extract_slopes(values):
    """
    Extract upslope, flatslope, and downslope characteristics.
    
    Returns: dict with slope metrics
    """
    slopes = np.diff(values)
    
    # Define slope thresholds
    upslope_threshold = 0.5  # positive slope
    downslope_threshold = -0.5  # negative slope
    
    upslopes = []
    flatslopes = []
    downslopes = []
    
    current_slope_type = None
    current_slope_start = 0
    current_slope_length = 0
    
    for i, slope in enumerate(slopes):
        if slope > upslope_threshold:
            slope_type = 'up'
        elif slope < downslope_threshold:
            slope_type = 'down'
        else:
            slope_type = 'flat'
        
        if slope_type == current_slope_type:
            current_slope_length += 1
        else:
            # Save previous slope if it exists
            if current_slope_type is not None and current_slope_length > 0:
                if current_slope_type == 'up':
                    upslopes.append(current_slope_length)
                elif current_slope_type == 'down':
                    downslopes.append(current_slope_length)
                elif current_slope_type == 'flat':
                    flatslopes.append(current_slope_length)
            
            current_slope_type = slope_type
            current_slope_length = 1
    
    # Handle last slope
    if current_slope_type is not None and current_slope_length > 0:
        if current_slope_type == 'up':
            upslopes.append(current_slope_length)
        elif current_slope_type == 'down':
            downslopes.append(current_slope_length)
        elif current_slope_type == 'flat':
            flatslopes.append(current_slope_length)
    
    return {
        'upslope_count': len(upslopes),
        'upslope_total_length': sum(upslopes),
        'upslope_avg_length': np.mean(upslopes) if upslopes else 0,
        'flatslope_count': len(flatslopes),
        'flatslope_total_length': sum(flatslopes),
        'flatslope_avg_length': np.mean(flatslopes) if flatslopes else 0,
        'downslope_count': len(downslopes),
        'downslope_total_length': sum(downslopes),
        'downslope_avg_length': np.mean(downslopes) if downslopes else 0,
    }


def extract_arc_length(values):
    """Calculate arc length of the dip curve"""
    arc_length = 0.0
    for i in range(len(values) - 1):
        # Assuming unit spacing in x (frame index)
        dy = values[i + 1] - values[i]
        dx = 1  # frame-to-frame
        arc_length += np.sqrt(dx**2 + dy**2)
    return arc_length


def extract_dip_features(dip, baseline=200):
    """
    Extract all features for a single dip.
    
    Returns: dict with all requested features
    """
    values = np.array(dip['values'])
    timestamps = dip['timestamps']
    
    # Baseline distance (when no car)
    max_depth = baseline - np.min(values)
    avg_depth = baseline - np.mean(values)
    
    # Area under curve (integral of distance below baseline)
    distances_below_baseline = baseline - values
    auc = np.trapezoid(distances_below_baseline)
    
    # Slopes
    slope_features = extract_slopes(values)
    
    # Arc length
    arc_len = extract_arc_length(values)
    
    # Time metrics
    time_start = timestamps[0]
    time_end = timestamps[-1]
    time_duration = time_end - time_start
    
    # Compile all features
    features = {
        'start_idx': dip['start_idx'],
        'end_idx': dip['end_idx'],
        'start_time': time_start,
        'end_time': time_end,
        'duration': time_duration,
        'dip_length_frames': len(values),
        'area_under_curve': auc,
        'upslope_count': slope_features['upslope_count'],
        'upslope_total_length': slope_features['upslope_total_length'],
        'upslope_avg_length': slope_features['upslope_avg_length'],
        'flatslope_length': slope_features['flatslope_total_length'],
        'downslope_count': slope_features['downslope_count'],
        'downslope_total_length': slope_features['downslope_total_length'],
        'downslope_avg_length': slope_features['downslope_avg_length'],
        'arc_length': arc_len,
        'max_depth': max_depth,
        'avg_depth': avg_depth,
    }
    
    return features



if __name__ == "__main__":
    file_path = r"RAK_DATA_F2025.TXT"
    output_csv = file_path.replace(".TXT", ".csv")
    
    print("=" * 70)
    print("RAK Ultrasonic Dip Feature Extraction")
    print("=" * 70)
    
    # Load data
    print(f"\n[1/4] Loading data from {file_path}")
    timearr = read_rak_data(file_path)
    print(f"      Loaded {len(timearr)} data points")
    
    # Find dips
    print(f"\n[2/4] Finding dips (threshold < 200, 5-200 frames)")
    dips = find_dips(timearr, threshold=200, min_frames=5, max_frames=200)
    print(f"      Found {len(dips)} valid dips")
    
    # Extract features
    print(f"\n[3/4] Extracting features for each dip")
    all_features = []
    for i, dip in enumerate(dips):
        features = extract_dip_features(dip)
        all_features.append(features)
        if (i + 1) % 10 == 0:
            print(f"      Processed {i + 1}/{len(dips)} dips")
    
    # Export to CSV
    print(f"\n[4/4] Exporting to CSV: {output_csv}")
    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)
    print(f"      Exported {len(df)} dips with features")
    
    print("\n" + "=" * 70)
    print("FEATURE SUMMARY")
    print("=" * 70)
    print(df.describe())
    
    print("\n" + "=" * 70)
    print("SAMPLE DIPS (first 5)")
    print("=" * 70)
    print(df.head())

