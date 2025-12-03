#include "vehicle_detector.h"

/**
 * Initialize the detector structure.
 */
void VehicleDetector_Init(VehicleDetector* detector) {
    // Calibration Init
    detector->is_calibrated = false;
    detector->baseline_cm = 0.0f;
    detector->calibration_sum = 0.0f;
    detector->calibration_count = 0;

    // Detection Init
    detector->in_dip = false;
    detector->sample_count = 0;
    detector->start_time_ms = 0;
}

/**
 * Calculate variance of the data buffer.
 * Matches the logic used in the Python ML model (Population Variance).
 */
static float calculate_variance(float* data, int count) {
    if (count < 2) return 0.0f;

    // 1. Calculate Mean
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        sum += data[i];
    }
    float mean = sum / count;

    // 2. Calculate Sum of Squared Differences
    float sq_diff_sum = 0.0f;
    for (int i = 0; i < count; i++) {
        float diff = data[i] - mean;
        sq_diff_sum += diff * diff;
    }

    // 3. Return Variance
    return sq_diff_sum / count;
}

/**
 * Main processing function. Call this every time you get a new sensor reading.
 * 
 * @param detector Pointer to the detector state struct
 * @param distance_cm The current distance reading in centimeters
 * @param current_time_ms The current system time in milliseconds
 * @return DetectionResult (NONE, VEHICLE, OTHER, or CALIBRATING)
 */
DetectionResult VehicleDetector_Process(VehicleDetector* detector, float distance_cm, uint32_t current_time_ms) {
    // --- 1. CALIBRATION PHASE ---
    if (!detector->is_calibrated) {
        detector->calibration_sum += distance_cm;
        detector->calibration_count++;

        if (detector->calibration_count >= CALIBRATION_SAMPLES) {
            detector->baseline_cm = detector->calibration_sum / detector->calibration_count;
            detector->is_calibrated = true;
        }
        return DETECTION_CALIBRATING;
    }

    // --- 2. DETECTION PHASE ---
    // Calculate dynamic threshold (Baseline - Offset)
    float dip_threshold = detector->baseline_cm - DIP_THRESHOLD_OFFSET;

    if (!detector->in_dip) {
        // STATE: IDLE (Waiting for object)
        
        if (distance_cm < dip_threshold) {
            // Transition -> IN DIP
            detector->in_dip = true;
            detector->sample_count = 0;
            detector->start_time_ms = current_time_ms;
            
            // Record first sample
            if (detector->sample_count < MAX_SAMPLES) {
                detector->depth_buffer[detector->sample_count++] = distance_cm;
            }
        }
        return DETECTION_NONE;
        
    } else {
        // STATE: IN DIP (Recording object)
        
        if (distance_cm < dip_threshold) {
            // Still in dip, continue recording
            if (detector->sample_count < MAX_SAMPLES) {
                detector->depth_buffer[detector->sample_count++] = distance_cm;
            } else {
                // Buffer overflow protection (reset if object stays too long > 20s)
                detector->in_dip = false;
                detector->sample_count = 0;
            }
            return DETECTION_NONE;
            
        } else {
            // Dip Ended (Distance went back above baseline threshold)
            detector->in_dip = false;
            uint32_t duration = current_time_ms - detector->start_time_ms;

            // 1. Filter Noise (Too short)
            if (duration < MIN_DIP_DURATION_MS) {
                return DETECTION_NONE;
            }

            // 2. Run Analysis Algorithm
            float variance = calculate_variance(detector->depth_buffer, detector->sample_count);

            // 3. Classify based on ML Threshold
            if (variance <= VEHICLE_VARIANCE_THRESH) {
                return DETECTION_VEHICLE;
            } else {
                return DETECTION_OTHER;
            }
        }
    }
}
