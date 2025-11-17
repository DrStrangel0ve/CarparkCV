"""RAK Data Processing - Extract ultrasonic dip features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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


def trim_initial_stable_period(timearr, baseline_value=220, tolerance=5,
                               stable_duration_seconds=5):
    """Trim initial samples until readings stay near baseline for the desired duration."""
    if not timearr:
        return timearr

    threshold = baseline_value - tolerance
    stable_duration_ms = stable_duration_seconds * 1000
    window_start_idx = None

    for idx, (timestamp, distance) in enumerate(timearr):
        if distance >= threshold:
            if window_start_idx is None:
                window_start_idx = idx
            if timestamp - timearr[window_start_idx][0] >= stable_duration_ms:
                if window_start_idx > 0:
                    print(
                        f"      Trimming first {window_start_idx} samples "
                        f"({timestamp - timearr[0][0]} raw time units) for warm-up"
                    )
                return timearr[window_start_idx:]
        else:
            window_start_idx = None

    return timearr


def calculate_rolling_baseline(timearr, dip_start_idx, window_seconds=100):
    """
    Calculate rolling baseline as the 95th percentile distance value in the past window_seconds.
    This represents the baseline (no object) distance at this point in time.
    
    Parameters:
    - timearr: list of [timestamp, distance] pairs
    - dip_start_idx: index of current dip
    - window_seconds: time window in seconds to look back (converted from raw time units)
    
    Returns:
    - rolling_baseline: 95th percentile distance in the window (baseline distance)
    """
    dip_start_time = timearr[dip_start_idx][0]
    
    # Find all data points within the window
    window_start_time = dip_start_time - window_seconds
    
    # Look back from dip start to find baseline
    baseline_values = []
    for i in range(max(0, dip_start_idx - 1000), dip_start_idx):  # Look back up to 1000 frames
        if timearr[i][0] >= window_start_time:
            baseline_values.append(timearr[i][1])
    
    # Return 95th percentile in window as baseline (excludes occasional spikes)
    if baseline_values:
        return np.percentile(baseline_values, 95)
    else:
        return 200  # Fallback to default baseline


def extract_dip_features(dip, baseline=None):
    """
    Extract all features for a single dip.
    
    Args:
        dip: dict with dip data
        baseline: optional baseline distance (if None, uses max value in dip as proxy)
    
    Returns: dict with all requested features
    """
    values = np.array(dip['values'])
    timestamps = dip['timestamps']
    
    # Use provided baseline, or estimate as max value in the dip region
    if baseline is None:
        baseline = np.max(values)
    
    # Baseline distance (when no car)
    max_depth = baseline - np.min(values)
    max_depth = min(max_depth, 220.0)
    avg_depth = baseline - np.mean(values)
    
    # Area under curve (integral of distance below baseline)
    distances_below_baseline = baseline - values
    auc = np.trapezoid(distances_below_baseline)
    
    # Slopes
    slope_features = extract_slopes(values)
    
    # Arc length
    arc_len = extract_arc_length(values)
    #464656
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ultrasonic dip features from RAK logs.")
    parser.add_argument(
        "--input",
        default="RAK_DATA_F2025_Test2.TXT",
        help="Path to the raw RAK TXT file (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path. Defaults to replacing .TXT with .csv beside the input file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_path = Path(args.input)
    output_csv = Path(args.output) if args.output else file_path.with_suffix(".csv")
    
    print("=" * 70)
    print("RAK Ultrasonic Dip Feature Extraction")
    print("=" * 70)
    
    # Load data
    print(f"\n[1/4] Loading data from {file_path}")
    timearr = read_rak_data(str(file_path))
    print(f"      Loaded {len(timearr)} data points")

    # Trim initial warm-up period where readings are below baseline
    timearr = trim_initial_stable_period(timearr, baseline_value=220, tolerance=5,
                                         stable_duration_seconds=5)
    print(f"      Using {len(timearr)} data points after warm-up trim")
    
    # Find dips
    print(f"\n[2/4] Finding dips (threshold < 200, 5-200 frames)")
    dips = find_dips(timearr, threshold=200, min_frames=5, max_frames=200)
    print(f"      Found {len(dips)} valid dips")
    
    # Extract features
    print(f"\n[3/4] Extracting features for each dip (with rolling 100s baseline)")
    all_features = []
    for i, dip in enumerate(dips):
        # Calculate rolling baseline for this dip
        rolling_baseline = calculate_rolling_baseline(timearr, dip['start_idx'], window_seconds=100)
        features = extract_dip_features(dip, baseline=rolling_baseline)
        all_features.append(features)
        if (i + 1) % 10 == 0:
            print(f"      Processed {i + 1}/{len(dips)} dips")
    
    # Export to CSV
    print(f"\n[4/4] Exporting to CSV: {output_csv}")

    # Expected feature columns (keeps output consistent even if no dips were found)
    expected_columns = [
        'start_idx', 'end_idx', 'start_time', 'end_time', 'duration',
        'dip_length_frames', 'area_under_curve',
        'upslope_count', 'upslope_total_length', 'upslope_avg_length',
        'flatslope_length', 'downslope_count', 'downslope_total_length', 'downslope_avg_length',
        'arc_length', 'max_depth', 'avg_depth'
    ]

    # Create DataFrame; if no features were extracted this will create an empty df with expected columns
    if all_features:
        df = pd.DataFrame(all_features)
        # Ensure all expected columns exist in df (missing keys become NaN)
        for col in expected_columns:
            if col not in df.columns:
                df[col] = pd.NA
    else:
        df = pd.DataFrame(columns=expected_columns)

    # Convert time values from milliseconds to seconds when present
    for tcol in ('start_time', 'end_time', 'duration'):
        if tcol in df.columns:
            # safe division even for empty series
            df[tcol] = df[tcol].astype('float64') / 1000.0

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


if __name__ == "__main__":
    main()

