# Object Tracking Algorithm Explanation

## The Problem You Had

Previously, if a car appeared in frames 1-10 in the video, it would be **counted 10 times** (once per frame), even though it's the same car. This is because the old tracker only checked if each detected object was a "duplicate" within the last few frames, but once objects were cleaned from the tracking list, they could be counted again.

## How the New Algorithm Works

### Core Concept: Persistent Identity Tracking

The new `ObjectTracker` uses **persistent object IDs** that remember each unique object **for the entire duration it appears** in the video (and even after it leaves, if it returns).

### Key Features:

#### 1. **Persistent Tracking Dictionary**
```python
self.active_objects = {}
# Each object is stored as:
# {object_id: {
#     'bbox': bounding_box,
#     'class': 'person'|'bicycle'|'car',
#     'first_frame': frame_number,
#     'last_frame': frame_number,
#     'counted': True/False,
#     'matches': number_of_frames_seen
# }}
```

#### 2. **Object Matching Across Frames**
For each frame, the algorithm:
1. Compares detected objects with all currently tracked objects
2. Uses `calculate_box_similarity()` to find the best match
3. If similarity > threshold (0.7), assigns the existing ID
4. If no match found, creates a NEW ID

This means:
- **Same car in frames 1-10**: Gets ID=1, appears in all frames
- **Car leaves and returns**: Gets marked as inactive, but if it comes back and matches, it keeps the same ID
- **New car**: Gets a new ID (e.g., ID=2)

#### 3. **One-Time Counting**
```python
if not self.active_objects[object_id]['counted']:
    self.active_objects[object_id]['counted'] = True
    self.object_counts[obj_class] += 1
    newly_counted.append(detection_info)
```

Each object has a `'counted'` flag:
- **First frame detected**: Flag is False → increment counter → set flag to True
- **Frames 2-10**: Flag is True → DO NOT increment counter

**Result**: Same car counted only ONCE

#### 4. **Memory Management**
Objects are only removed from tracking if they haven't been seen for `retention_frames` (default ~2 seconds or 60 frames at 30 FPS).

This allows:
- Objects that temporarily leave the frame to still be recognized when they return
- Objects passing through the ROI briefly to be properly tracked

### Return Values

The `update()` method now returns a dictionary:
```python
{
    'all_detections': [...],      # All objects currently visible in frame
    'newly_counted': [...],        # Objects counted for first time this frame
    'counts': {                    # CUMULATIVE totals
        'person': 5,
        'bicycle': 2,
        'car': 12
    },
    'total': 19
}
```

### What Gets Displayed

On the video overlay, you now see:
```
Frame: 150/1000 | Time: 5.0s

In Frame: People: 2 | Bicycles: 1 | Vehicles: 3 | Total: 6
(These are objects visible RIGHT NOW in this frame)

TOTAL COUNTED: People: 5 | Bicycles: 2 | Vehicles: 12 | Total: 19
(These are cumulative totals - the actual object counts)
```

**Important Distinction:**
- **"In Frame"** = How many objects are visible in THIS frame (can vary)
- **"TOTAL COUNTED"** = How many unique objects have passed through (always increasing or stable)

### Newlydetected Objects Visualization

Objects are marked with `[NEW]` tag in their label when first detected, with thicker bounding boxes (3px instead of 2px).

## Example Scenario

**Video of parking lot entrance:**
- **Frame 1-30**: Car A enters → ID=1 created → counted ONCE (People:0, Bicycles:0, Vehicles:1)
- **Frame 31-60**: Car A still in frame → NOT recounted (People:0, Bicycles:0, Vehicles:1)
- **Frame 61**: Car A leaves frame
- **Frame 62-90**: Car B enters and passes through → ID=2 created → counted (People:0, Bicycles:0, Vehicles:2)
- **Frame 91**: Person walking → ID=1 (person) created → counted (People:1, Bicycles:0, Vehicles:2)
- **Frame 92-120**: Person walks through multiple frames → STILL only 1 person counted total

**Final Result:** 2 vehicles + 1 person = 3 total unique objects counted

## CSV Output

The saved CSV now includes:

| Column | Meaning |
|--------|---------|
| `time_seconds` | Timestamp |
| `frame_number` | Frame number |
| `people_in_frame` | How many people visible in THIS frame |
| `bicycles_in_frame` | How many bicycles visible in THIS frame |
| `vehicles_in_frame` | How many vehicles visible in THIS frame |
| `total_in_frame` | Total visible objects in THIS frame |
| `total_people_counted` | Cumulative people (stays same until new person enters) |
| `total_bicycles_counted` | Cumulative bicycles (stays same until new bicycle enters) |
| `total_vehicles_counted` | Cumulative vehicles (stays same until new vehicle enters) |
| `total_objects_counted` | Cumulative total |
| `newly_counted_this_frame` | How many NEW objects were first detected in THIS frame |

## Parameters You Can Tune

In the `ObjectTracker` initialization:
```python
object_tracker = ObjectTracker(
    retention_frames=int(fps * 2),     # 2 seconds at current FPS
    similarity_threshold=0.6            # How similar boxes must be (0-1, higher = stricter)
)
```

- **Higher `retention_frames`**: Objects stay tracked longer even if out of frame (better for tracking)
- **Higher `similarity_threshold`**: Stricter matching (less likely to lose track, but might create duplicate IDs)
- **Lower `similarity_threshold`**: More lenient matching (might track different objects as same)

## Visual Features

✅ **Newly counted objects**: Thicker green/blue/purple boxes with `[NEW]` label
✅ **Persistent objects**: Normal thickness boxes (already counted)
✅ **Two separate overlays**: "In Frame" vs "TOTAL COUNTED"
✅ **Color coding**: Blue=People, Purple=Bicycles, Green=Vehicles
