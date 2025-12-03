#include <stdio.h>
#include "vehicle_detector.h"

// Mock function to simulate time passing (in a real MCU, use millis() or similar)
static uint32_t mock_time_ms = 0;

int main() {
    printf("Starting RAK11300 Vehicle Detection Simulation...\n");

    // 1. Initialize Detector
    VehicleDetector detector;
    VehicleDetector_Init(&detector);

    // 2. Simulate Data Stream: Baseline -> Car -> Baseline -> Person -> Baseline
    // Car: Flat top (low variance)
    // Person: Irregular (high variance)
    float sensor_stream[] = {
        230, 230, 230,              // Baseline
        150, 148, 152, 150, 149,    // Car (Vehicle)
        230, 230,                   // Baseline
        180, 60, 120, 70, 150,      // Person (Other)
        230, 230                    // Baseline
    };

    int stream_len = sizeof(sensor_stream) / sizeof(float);

    // 3. Run Loop
    for (int i = 0; i < stream_len; i++) {
        float dist = sensor_stream[i];
        
        // Call the processor
        DetectionResult result = VehicleDetector_Process(&detector, dist, mock_time_ms);

        // Handle Result
        if (result == DETECTION_CALIBRATING) {
            printf("[Time %d ms] Calibrating... (Sample %d/%d)\n", mock_time_ms, detector.calibration_count, CALIBRATION_SAMPLES);
        } else if (result == DETECTION_VEHICLE) {
            printf("[Time %d ms] DETECTED: VEHICLE (Car/Truck)\n", mock_time_ms);
        } else if (result == DETECTION_OTHER) {
            printf("[Time %d ms] DETECTED: OTHER OBJECT (Person/Bike)\n", mock_time_ms);
        }

        // Advance time by 40ms (simulating sensor rate)
        mock_time_ms += 40;
    }

    return 0;
}
