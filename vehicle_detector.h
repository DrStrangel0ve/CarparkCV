#ifndef VEHICLE_DETECTOR_H
#define VEHICLE_DETECTOR_H

#include <stdint.h>
#include <stdbool.h>

// --- CONFIGURATION ---
// #define BASELINE_DEPTH_CM 220.0f    // REMOVED: Now calculated dynamically
#define VEHICLE_VARIANCE_THRESH 3070.0f // From ML model: Variance <= 3070 is a Vehicle
#define MIN_DIP_DURATION_MS 200        // Minimum duration to filter noise (0.2s)
#define MAX_SAMPLES 500                // Max buffer size (20s @ 40ms rate)
#define CALIBRATION_SAMPLES 50         // Samples to establish baseline (2s @ 40ms)
#define DIP_THRESHOLD_OFFSET 15.0f     // Must drop 15cm below baseline to trigger dip

// Result types
typedef enum {
    DETECTION_NONE = 0,
    DETECTION_VEHICLE,
    DETECTION_OTHER,
    DETECTION_CALIBRATING // System is learning the baseline
} DetectionResult;

// Detector State Struct
typedef struct {
    // Calibration State
    bool is_calibrated;
    float baseline_cm;
    float calibration_sum;
    int calibration_count;

    // Detection State
    bool in_dip;
    float depth_buffer[MAX_SAMPLES];
    int sample_count;
    uint32_t start_time_ms;
} VehicleDetector;

// Function Prototypes
void VehicleDetector_Init(VehicleDetector* detector);
DetectionResult VehicleDetector_Process(VehicleDetector* detector, float distance_cm, uint32_t current_time_ms);

#endif // VEHICLE_DETECTOR_H
