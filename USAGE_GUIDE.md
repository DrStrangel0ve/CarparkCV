# Updated Detection Script - Usage Guide

## What Changed

Your script now uses a **persistent object tracker** that counts each unique object only once, no matter how many frames it appears in.

## How to Use

### Running the Script

```python
python CV_Vehicle_tracking.py
```

### During Video Playback

1. **Select ROI**: Draw a rectangle around the area you want to monitor (e.g., parking lot entrance)
2. **Controls**:
   - `q` - Quit playback
   - `p` - Pause/Resume
   - `s` - Save current frame
   - `r` - Reselect ROI

### Monitoring the Display

The overlay shows **TWO key metrics**:

```
REAL-TIME COUNTS (THIS FRAME):
In Frame: People: 2 | Bicycles: 1 | Vehicles: 3 | Total: 6

CUMULATIVE COUNTS (ENTIRE PASS):
TOTAL COUNTED: People: 5 | Bicycles: 2 | Vehicles: 12 | Total: 19
```

**Example interpretation:**
- A car enters at frame 100 and stays until frame 200
- "In Frame" will show `Vehicles: 1` for frames 100-200
- "TOTAL COUNTED" will increment from 11 to 12 at frame 100 and stay at 12

## Output Files

### `detection_results.csv`

Contains detailed frame-by-frame data:

| Column | Description |
|--------|-------------|
| `time_seconds` | Video timestamp |
| `frame_number` | Frame index |
| `people_in_frame` | People visible in current frame |
| `bicycles_in_frame` | Bicycles visible in current frame |
| `vehicles_in_frame` | Vehicles visible in current frame |
| `total_in_frame` | Total objects visible in current frame |
| `total_people_counted` | Cumulative unique people (all-time) |
| `total_bicycles_counted` | Cumulative unique bicycles (all-time) |
| `total_vehicles_counted` | Cumulative unique vehicles (all-time) |
| `total_objects_counted` | Cumulative unique objects (all-time) |
| `newly_counted_this_frame` | How many new objects were detected in this frame |

**Key insight**: The "total_*_counted" columns will look like a **step function** - they stay flat and only jump up when a NEW object is detected.

### `detection_timeline.png`

6-panel visualization showing:
1. **Top-left**: Cumulative people count (step function pattern)
2. **Middle-left**: Cumulative bicycles count (step function pattern)
3. **Bottom-left**: Cumulative vehicles count (step function pattern)
4. **Top-right**: People visible per frame (varies)
5. **Middle-right**: Bicycles visible per frame (varies)
6. **Bottom-right**: Vehicles visible per frame (varies)

**What you're looking for:**
- Left side (cumulative) = Mostly flat with sudden jumps (new objects)
- Right side (visible) = Variable peaks and valleys (objects entering/leaving)

### Saved Frames

Press `s` during playback to save:
- `detected_frame_XXXXXX.jpg` - Frame with bounding boxes and labels

## Statistics Output

After processing completes:

```
=== OBJECT DETECTION SUMMARY ===
Total video duration: 60.00 seconds (1.0 minutes)
Total frames analyzed: 1800

CUMULATIVE COUNTS (Each object counted once):
  Total People: 5
  Total Bicycles: 2
  Total Vehicles: 12
  TOTAL OBJECTS: 19

PEOPLE VISIBILITY:
  Frames with people visible: 450 (25.0%)
  Average people visible per frame: 1.25
  Maximum people in single frame: 3

BICYCLES VISIBILITY:
  Frames with bicycles visible: 180 (10.0%)
  Average bicycles visible per frame: 0.50
  Maximum bicycles in single frame: 2

VEHICLES VISIBILITY:
  Frames with vehicles visible: 1200 (66.7%)
  Average vehicles visible per frame: 2.10
  Maximum vehicles in single frame: 5

VISIBILITY TOTALS:
  Frames with any objects visible: 1300 (72.2%)
  Average total objects visible per frame: 3.85
  Maximum objects in single frame: 8
```

**Key metrics:**
- **"Total People/Bicycles/Vehicles: X"** - Unique count you want
- **"Frames with X visible: Y%"** - How much of the video had objects
- **"Average X per frame"** - Typical occupancy
- **"Maximum X in frame"** - Peak load

## Tuning Parameters

If the tracking isn't working well, you can adjust in `detect_objects_in_video()`:

```python
object_tracker = ObjectTracker(
    retention_frames=int(fps * 2),     # Seconds * fps (2 sec default)
    similarity_threshold=0.6            # 0.0-1.0 (0.6 default, higher = stricter)
)
```

### Common Issues & Solutions

**Problem: Same object counted multiple times**
- ❌ `similarity_threshold` too high (0.9)
- ✅ Decrease to 0.5-0.6

**Problem: Different objects merged into one ID**
- ❌ `similarity_threshold` too low (0.3)
- ✅ Increase to 0.7-0.8

**Problem: Objects lose tracking when temporarily out of frame**
- ❌ `retention_frames` too low (5)
- ✅ Increase to `int(fps * 5)` for 5 seconds

**Problem: Objects still showing as tracked way too long after leaving**
- ❌ `retention_frames` too high (300)
- ✅ Decrease to `int(fps * 1)` for 1 second

## Example Workflow

1. **Run the script**: `python CV_Vehicle_tracking.py`
2. **Select your ROI**: Draw around parking lot entrance
3. **Watch the counts grow**: Press play, observe "TOTAL COUNTED" incrementing
4. **Save interesting frames**: Press `s` when seeing interesting detections
5. **Let it finish**: Script will process entire video
6. **View results**:
   - Open `detection_results.csv` in Excel/Pandas
   - View `detection_timeline.png` in image viewer
   - Read console statistics

## Validating Results

**How to tell if counts are correct:**

1. Open the video and manually count unique objects passing through
2. Compare with "TOTAL COUNTED" values
3. They should match (or be very close)

**Spot-check the CSV:**
- Sort by `total_*_counted` columns
- They should be **non-decreasing** (never go down, only increase or stay flat)
- Large jumps indicate multiple new objects detected in same frame
