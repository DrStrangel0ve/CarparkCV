#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// Constants
#define DIST_VAR 4800.0
#define GRAD_THRESH 35.0
#define DIP_THRESH 10.0
#define MIN_DIP_TIME_SEC 0.5
#define MAX_DIP_TIME_SEC 10.0
#define MERGE_GAP_SEC 0.2
#define MIN_VEH_HEIGHT 30.0
#define SENSOR_INTERVAL_SEC 0.044
#define BASELINE 225.0

// Calibration
#define CALIBRATION_DURATION_MS 60000
#define MAX_DIST_MM 5000 // Max distance for histogram

// Buffer size for a single vehicle event (10s @ ~20Hz = 200 samples, 500 is safe)
#define MAX_BUFFER_SIZE 5000

typedef struct {
    long long time;
    double distance;
} DataPoint;

typedef struct {
    long long time;
    double duration;
    double min_depth;
    double variance;
    double avg_gradient;
    char type[20];
} VehicleResult;

typedef enum {
    STATE_WAIT_FOR_TRIGGER,
    STATE_CALIBRATING,
    STATE_IDLE,
    STATE_IN_DIP,
    STATE_WAITING_FOR_MERGE
} DetectionState;

typedef struct {
    DetectionState state;
    DataPoint buffer[MAX_BUFFER_SIZE];
    int buffer_count;
    long long last_dip_time;
    DataPoint prev_point;
    int has_prev_point;
    
    // Calibration
    long long calibration_start_time;
    uint16_t calibration_histogram[MAX_DIST_MM];
    
    // Configurable thresholds (can be set at runtime if needed)
    double baseline;
    double dip_threshold;
} VehicleDetector;

// --- Helper Functions ---

double calculate_variance(DataPoint *data, int n) {
    if (n < 2) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[i].distance;
    }
    double mean = sum / n;
    double sq_diff_sum = 0.0;
    for (int i = 0; i < n; i++) {
        sq_diff_sum += (data[i].distance - mean) * (data[i].distance - mean);
    }
    return sq_diff_sum / n;
}

double calculate_avg_gradient(DataPoint *data, int n) {
    if (n < 2) return 0.0;
    double sum_abs_diff = 0.0;
    for (int i = 0; i < n - 1; i++) {
        sum_abs_diff += fabs(data[i+1].distance - data[i].distance);
    }
    return sum_abs_diff / (n - 1);
}

// --- Core Logic ---

void rak_init(VehicleDetector *detector) {
    detector->state = STATE_WAIT_FOR_TRIGGER;
    detector->buffer_count = 0;
    detector->has_prev_point = 0;
    detector->baseline = BASELINE; // Default, will be overwritten
    detector->dip_threshold = BASELINE - DIP_THRESH;
    memset(detector->calibration_histogram, 0, sizeof(detector->calibration_histogram));
}

void rak_message_received(VehicleDetector *detector, long long current_time_ms) {
    if (detector->state == STATE_WAIT_FOR_TRIGGER) {
        detector->state = STATE_CALIBRATING;
        detector->calibration_start_time = current_time_ms;
        printf("Calibration started at %lld ms\n", current_time_ms);
        // Clear histogram just in case
        memset(detector->calibration_histogram, 0, sizeof(detector->calibration_histogram));
    }
}

// Returns 1 if a vehicle was detected, 0 otherwise.
// If 1, 'result' is populated.
int rak_process_sample(VehicleDetector *detector, long long time_ms, double distance, VehicleResult *result) {
    DataPoint current_point = {time_ms, distance};
    int vehicle_detected = 0;

    // Always update prev_point at the end, but we might need it now.
    
    switch (detector->state) {
        case STATE_WAIT_FOR_TRIGGER:
            // Do nothing until triggered
            break;

        case STATE_CALIBRATING:
            if (time_ms - detector->calibration_start_time < CALIBRATION_DURATION_MS) {
                int d = (int)(distance + 0.5);
                if (d >= 0 && d < MAX_DIST_MM) {
                    if (detector->calibration_histogram[d] < 65535) {
                        detector->calibration_histogram[d]++;
                    }
                }
            } else {
                // Finish calibration
                int mode_dist = 0;
                int max_count = -1;
                for (int i = 0; i < MAX_DIST_MM; i++) {
                    if (detector->calibration_histogram[i] > max_count) {
                        max_count = detector->calibration_histogram[i];
                        mode_dist = i;
                    }
                }
                detector->baseline = (double)mode_dist;
                detector->dip_threshold = detector->baseline - DIP_THRESH;
                detector->state = STATE_IDLE;
                printf("Calibration Complete. Baseline: %.2f\n", detector->baseline);
            }
            break;

        case STATE_IDLE:
            if (distance < detector->dip_threshold) {
                // Start of dip
                detector->state = STATE_IN_DIP;
                detector->buffer_count = 0;
                
                // Add previous point if available (context)
                if (detector->has_prev_point) {
                    if (detector->buffer_count < MAX_BUFFER_SIZE) {
                        detector->buffer[detector->buffer_count++] = detector->prev_point;
                    }
                }
                
                // Add current point
                if (detector->buffer_count < MAX_BUFFER_SIZE) {
                    detector->buffer[detector->buffer_count++] = current_point;
                }
                
                detector->last_dip_time = time_ms;
            }
            break;

        case STATE_IN_DIP:
            // Add to buffer
            if (detector->buffer_count < MAX_BUFFER_SIZE) {
                detector->buffer[detector->buffer_count++] = current_point;
            }
            
            if (distance < detector->dip_threshold) {
                detector->last_dip_time = time_ms;
            } else {
                // Went above threshold, wait to see if it's just a gap
                detector->state = STATE_WAITING_FOR_MERGE;
            }
            break;

        case STATE_WAITING_FOR_MERGE:
            // Add to buffer (we include the gap points until we decide to close)
            if (detector->buffer_count < MAX_BUFFER_SIZE) {
                detector->buffer[detector->buffer_count++] = current_point;
            }

            if (distance < detector->dip_threshold) {
                // Dip continues
                detector->state = STATE_IN_DIP;
                detector->last_dip_time = time_ms;
            } else {
                // Still above threshold
                double time_gap = (time_ms - detector->last_dip_time) / 1000.0;
                
                if (time_gap > MERGE_GAP_SEC) {
                    // Gap too long, close event
                    // The buffer currently includes the gap points. 
                    // The original logic included them up to the break point.
                    
                    // Analyze buffer
                    double duration_sec = (detector->buffer[detector->buffer_count-1].time - detector->buffer[0].time) / 1000.0;
                    
                    // Find min depth
                    double min_val = detector->buffer[0].distance;
                    for(int k=1; k<detector->buffer_count; k++) {
                        if(detector->buffer[k].distance < min_val) min_val = detector->buffer[k].distance;
                    }

                    int height_condition = (min_val < (detector->baseline - MIN_VEH_HEIGHT));
                    int time_condition = (duration_sec >= MIN_DIP_TIME_SEC && duration_sec <= MAX_DIP_TIME_SEC);

                    if (time_condition && height_condition) {
                        double variance = calculate_variance(detector->buffer, detector->buffer_count);
                        double avg_gradient = calculate_avg_gradient(detector->buffer, detector->buffer_count);

                        result->time = detector->buffer[0].time;
                        result->duration = duration_sec;
                        result->min_depth = min_val;
                        result->variance = variance;
                        result->avg_gradient = avg_gradient;

                        if (variance < DIST_VAR && avg_gradient < GRAD_THRESH) {
                            strcpy(result->type, "Vehicle");
                            vehicle_detected = 1; // Signal detection
                        } else {
                            strcpy(result->type, "Person/Bike");
                            // We can choose to return 1 here if we want to report non-vehicles too
                            // For now, let's print it but maybe not count it as a "Vehicle" return?
                            // The user asked "determine if cars are going through".
                            // Let's return 1 but the type distinguishes it.
                            vehicle_detected = 1; 
                        }
                    }

                    // Reset
                    detector->state = STATE_IDLE;
                    detector->buffer_count = 0;
                }
            }
            break;
    }

    detector->prev_point = current_point;
    detector->has_prev_point = 1;
    
    return vehicle_detected;
}

// --- Main for Testing (Simulates Live Feed) ---

int main() {
    const char *file_path = "RAK_DATA_F2025_Test2.TXT";
    long long RAK_START_TIME_MS = 6800 + 465170;

    FILE *file = fopen(file_path, "r");
    if (!file) {
        printf("File %s not found.\n", file_path);
        return 1;
    }

    printf("Processing %s in LIVE simulation mode...\n", file_path);
    printf("Filtering data starting from %lld ms\n", RAK_START_TIME_MS);
    printf("Baseline: %.2f, Dip Threshold: %.2f\n", BASELINE, BASELINE - DIP_THRESH);
    printf("\n============================================================\n");
    printf("LIVE DETECTION LOG\n");
    printf("============================================================\n");
    printf("%-15s %-15s %-10s %-10s %-10s %-10s\n", "Time (ms)", "Duration (s)", "Min Depth", "Variance", "Gradient", "Type");
    printf("--------------------------------------------------------------------------------\n");

    VehicleDetector detector;
    rak_init(&detector);
    
    // Simulate receiving a message to start calibration immediately
    rak_message_received(&detector, RAK_START_TIME_MS);
    
    VehicleResult result;
    int total_vehicles = 0;

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        long long t;
        double d;
        if (sscanf(line, "%lld,%lf", &t, &d) != 2) {
             if (sscanf(line, "%lld %lf", &t, &d) != 2) continue; 
        }

        if (t >= RAK_START_TIME_MS) {
            // Feed sample to detector
            if (rak_process_sample(&detector, t, d, &result)) {
                printf("%-15lld %-15.3f %-10.2f %-10.2f %-10.2f %-10s\n", 
                    result.time, 
                    result.duration, 
                    result.min_depth, 
                    result.variance, 
                    result.avg_gradient, 
                    result.type);
                
                if (strcmp(result.type, "Vehicle") == 0) {
                    total_vehicles++;
                }
            }
        }
    }

    fclose(file);
    printf("--------------------------------------------------------------------------------\n");
    printf("Total Vehicles Detected: %d\n", total_vehicles);

    return 0;
}
