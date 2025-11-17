import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import csv

# Read RAK data
def read_rak_data(filename):
    """Read RAK data and return time (ms) and distance (m)"""
    times = []
    distances = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                t, d = map(int, line.split(','))
                times.append(t)
                distances.append(d)
    return np.array(times), np.array(distances)

# Read detection results
def read_detections(filename):
    """Read detection results CSV and extract vehicle detection times"""
    df = pd.read_csv(filename)
    # Get times when vehicles were detected (vehicles_in_frame > 0)
    vehicle_times = df[df['vehicles_in_frame'] > 0]['time_seconds'].values
    return vehicle_times

def find_valid_start(times, distances, threshold=220, min_duration_ms=1000):
    """
    Find the first point where distance > threshold for at least min_duration_ms.
    Returns the index of the start point.
    """
    max_stretch = 0
    current_start = None
    current_start_time = None
    best_start_idx = None

    for i in range(len(times)):
        if distances[i] > threshold:
            if current_start is None:
                current_start = i
                current_start_time = times[i]
            current_stretch = times[i] - current_start_time
            if current_stretch > max_stretch:
                max_stretch = current_stretch
                best_start_idx = current_start
        else:
            current_start = None
            current_start_time = None

    if max_stretch >= min_duration_ms:
        return best_start_idx
    return None

def create_corrected_data(times, distances, start_idx):
    """Create corrected time and distance arrays starting from start_idx"""
    if start_idx is None:
        return None, None
    
    corrected_times = (times[start_idx:] - times[start_idx]) / 1000.0  # Convert to seconds
    corrected_distances = distances[start_idx:]
    return corrected_times, corrected_distances

# Read data
rak_times, rak_distances = read_rak_data('RAK_DATA_F2025_Test2.TXT')
vehicle_detections = read_detections('detection_results.csv')

# Find valid start point
start_idx = find_valid_start(rak_times, rak_distances)
print(f"Valid start point found at index: {start_idx}")
if start_idx is not None:
    print(f"Time: {rak_times[start_idx]}ms, Distance: {rak_distances[start_idx]}m")

# Create corrected data
corrected_times, corrected_distances = create_corrected_data(rak_times, rak_distances, start_idx)

if corrected_times is not None:
    # Create figure with slider for syncing
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.25)

    # Plot distance data
    line_distance, = ax.plot(corrected_times, corrected_distances, 'b-', label='Distance (m)', linewidth=2)
    
    # Container for state
    plot_state = {'vehicle_lines': [], 'offset_text': None}
    initial_offset = 0
    
    def update_plot(offset):
        """Update plot with new offset for vehicle detections"""
        # Remove old vehicle detection lines
        for line in plot_state['vehicle_lines']:
            line.remove()
        plot_state['vehicle_lines'].clear()
        
        # Remove old text
        if plot_state['offset_text'] is not None:
            plot_state['offset_text'].remove()
        
        # Add new vehicle detection lines with offset
        vehicle_count = 0
        for det_time in vehicle_detections:
            synced_time = det_time + offset
            if 0 <= synced_time <= corrected_times[-1]:
                line = ax.axvline(x=synced_time, color='r', alpha=0.5, linestyle='--', linewidth=1)
                plot_state['vehicle_lines'].append(line)
                vehicle_count += 1
        
        # Add text showing current offset and vehicle count
        plot_state['offset_text'] = ax.text(0.02, 0.98, f'Offset: {offset:.2f}s | Vehicles shown: {vehicle_count}', 
                             transform=ax.transAxes, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig.canvas.draw_idle()

    # Create slider
    ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
    slider = Slider(ax_slider, 'Time Offset (s)', -30000, 30000, valinit=0, valstep=0.1)
    
    # Create text box for manual offset input
    ax_textbox = plt.axes([0.2, 0.08, 0.15, 0.04])
    textbox = TextBox(ax_textbox, 'Offset (s):', initial='0')
    
    def on_slider_change(val):
        update_plot(val)
        textbox.set_val(f'{val:.2f}')
    
    def on_textbox_change(text):
        try:
            offset_val = float(text)
            # Allow any value, just update the plot
            update_plot(offset_val)
            # Update slider only if within range, otherwise just update plot
            if -300 <= offset_val <= 300:
                slider.set_val(offset_val)
        except ValueError:
            pass  # Ignore invalid input
    
    slider.on_changed(on_slider_change)
    textbox.on_submit(on_textbox_change)

    # Set up the plot
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Distance (metres)', fontsize=12)
    ax.set_title('RAK Data - Distance vs Time with Vehicle Detections (Adjust slider to sync)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='b', lw=2),
                   Line2D([0], [0], color='r', lw=1, linestyle='--')]
    ax.legend(custom_lines, ['Distance (m)', 'Vehicle Detections'], loc='upper right')

    # Initial plot
    update_plot(initial_offset)

    plt.show()
else:
    print("Could not find valid start point!")
