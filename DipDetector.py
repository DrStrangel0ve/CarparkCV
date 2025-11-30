import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load csv file called detection_results_with_depth.csv
filename = r"C:\Users\Arnav\Downloads\detection_results_with_depth.csv"

# Try to load the file, handle if it doesn't exist (for testing purposes)
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"File '{filename}' not found. Please ensure the file exists.")
    # For the purpose of the script, we stop here if file is missing, 
    # or we could generate dummy data. I'll assume the user will provide the file.
    exit()

# The csv file contains these columns im interested in : 'people_in_frame', 'bicycles_in_frame', 'vehicles_in_frame', 'ultrawide_depth_m'
# We also need a time reference. Assuming 'timestamp' column exists or using index.
if 'timestamp' not in df.columns:
    # Attempt to find a time-like column or default to index assuming 30fps
    # print("Warning: 'timestamp' column not found. Assuming 30 FPS for duration calculations.")
    df['timestamp'] = df.index / 30.0

# Find dips. Dip starts when ultrasonic val goes below 200 and then goes back above 200. 
# Take one preceding and one following frame as well. Dip has to be at least 0.2 seconds.

threshold = 200
depth_col = 'ultrawide_depth_m'

dips = []
in_dip = False
start_idx = -1

# Iterate through the dataframe
for i in range(len(df)):
    val = df.iloc[i][depth_col]
    
    # Check for dip start
    if not in_dip and val < threshold:
        in_dip = True
        start_idx = i
    
    # Check for dip end
    elif in_dip and val >= threshold:
        in_dip = False
        # The dip strictly is from start_idx to i-1
        # We want to include one preceding (start_idx - 1) and one following (i)
        
        dip_start_idx = max(0, start_idx - 1)
        dip_end_idx = min(len(df) - 1, i)
        
        # Extract the dip data
        dip_data = df.iloc[dip_start_idx : dip_end_idx + 1]
        
        # Calculate duration
        duration = dip_data['timestamp'].iloc[-1] - dip_data['timestamp'].iloc[0]
        
        # Dip has to be at least 0.2 seconds
        if duration >= 0.2:
            # If there is a vehicle in frame > 0, assign type 'V'. 
            # Else, if there is a bicycle in frame > 0, assign type 'B'. 
            # Else, if there is a person in frame > 0, assign type 'P'. 
            # Else assign type 'N'.
            
            dip_type = 'N'
            if (dip_data['vehicles_in_frame'] > 0).any():
                dip_type = 'V'
            elif (dip_data['bicycles_in_frame'] > 0).any():
                dip_type = 'B'
            elif (dip_data['people_in_frame'] > 0).any():
                dip_type = 'P'
            
            # Also find these stats for each dip: total duration, arc length, derivative sign changes
            
            # Arc length: sum of sqrt(dt^2 + dy^2)
            dt = np.diff(dip_data['timestamp'])
            dy = np.diff(dip_data[depth_col])
            arc_length = np.sum(np.sqrt(dt**2 + dy**2))
            
            # Derivative sign changes (like how many times does it go from pos to neg)
            # We look at the sign of the derivative (dy/dt). Since dt > 0, sign of dy is enough.
            # We count how many times the sign changes.
            if len(dy) > 0:
                signs = np.sign(dy)
                # Filter out zeros to see true crossings if needed, or just diff
                # A change is when sign[k] != sign[k-1]
                # We can filter zeros to handle flat spots
                signs_nonzero = signs[signs != 0]
                if len(signs_nonzero) > 1:
                    sign_changes = np.sum(np.diff(signs_nonzero) != 0)
                else:
                    sign_changes = 0
            else:
                sign_changes = 0

            max_depth = 225 - min(dip_data[depth_col])
            
            # Extract only the features we need for the ML model
            depth_values = dip_data[depth_col].values
            
            # Rate of change metrics
            depth_diff = np.diff(depth_values)
            max_slope = np.max(np.abs(depth_diff)) if len(depth_diff) > 0 else 0
            
            # Smoothness (variance of second derivative)
            if len(depth_diff) > 1:
                second_deriv = np.diff(depth_diff)
                smoothness = np.std(second_deriv)
            else:
                smoothness = 0
            
            # Calculate vehicle confidence based on metrics (old method - kept for comparison)
            vehicle_confidence = 0.0
            
            # Duration score (vehicles avg 1.32s, pedestrians avg 0.67s)
            if duration > 1.0:
                vehicle_confidence += 0.4  # Strong indicator
            elif duration > 0.8:
                vehicle_confidence += 0.2
            
            # Max depth score (vehicles avg 151.6m, pedestrians avg 179.4m)
            if max_depth < 160:
                vehicle_confidence += 0.35  # Strong indicator
            elif max_depth < 170:
                vehicle_confidence += 0.15
            
            # Derivative sign changes score (vehicles avg 2.76, pedestrians avg 5.65)
            if sign_changes < 4:
                vehicle_confidence += 0.25  # Smooth profile
            elif sign_changes < 6:
                vehicle_confidence += 0.10
            
            # Cap confidence at 1.0
            vehicle_confidence = min(1.0, vehicle_confidence)

            dips.append({
                'start_index': dip_start_idx,
                'end_index': dip_end_idx,
                'type': dip_type,
                'duration': duration,
                'arc_length': arc_length,
                'derivative_sign_changes': sign_changes,
                'max_depth': max_depth,
                'smoothness': smoothness,
                'max_slope': max_slope,
                'vehicle_confidence': vehicle_confidence
            })

# Create a DataFrame for the results
results_df = pd.DataFrame(dips)

# Display or save results
print(f"Found {len(results_df)} dips.")
print(results_df)

# Create a table for average, min and max for each feature by the type of detection
print("\n" + "="*80)
print("Summary Statistics by Detection Type")
print("="*80)

features = ['duration', 'arc_length', 'derivative_sign_changes', 'max_depth']

for feature in features:
    print(f"\n{feature.upper()}:")
    stats_by_type = results_df.groupby('type')[feature].agg(['mean', 'min', 'max'])
    stats_by_type.columns = ['Average', 'Min', 'Max']
    print(stats_by_type)

# Save to csv
results_df.to_csv('dip_analysis_results.csv', index=False)
print(f"\nResults saved to 'dip_analysis_results.csv'")

# ============================================================================
# LOGISTIC REGRESSION MODEL FOR VEHICLE DETECTION
# ============================================================================
print("\n" + "="*80)
print("Training Logistic Regression Model for Vehicle Classification")
print("="*80)

# Use only the most important features (fewer calculations)
feature_cols = [
    'duration',                    # Strong positive indicator for vehicles
    'derivative_sign_changes',     # Vehicles have fewer sign changes
    'max_depth',                   # Vehicles have shallower depth
    'smoothness',                  # Vehicles have smoother profiles
    'max_slope'                    # Vehicles have gentler transitions
]

# Create binary target: 1 if vehicle (type 'V'), 0 otherwise (excluding first vehicle)
results_df['is_vehicle'] = (results_df['type'] == 'V').astype(int)

# Remove the first vehicle (index 0) as per user request
model_data = results_df[results_df.index != 0].copy()

# Handle any NaN values that might exist
X = model_data[feature_cols].fillna(0)
y = model_data['is_vehicle']

print(f"\nDataset (excluding first vehicle): {len(X)} samples, {len(feature_cols)} features")
print(f"Features used: {', '.join(feature_cols)}")
print(f"Vehicles: {y.sum()}, Non-vehicles: {(y == 0).sum()}")
print(f"Vehicle percentage: {y.sum() / len(y) * 100:.1f}%")

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model with more iterations (epochs) and balanced weights
model = LogisticRegression(
    random_state=42, 
    max_iter=5000,  # Increased from 1000 to allow more training iterations
    class_weight='balanced',
    solver='lbfgs'  # Better for smaller datasets
)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# Evaluate model
print(f"\nTrain Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")

print(f"\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)
print(f"False Positives (People classified as Vehicles): {cm[0, 1]}")
print(f"False Negatives (Vehicles classified as People): {cm[1, 0]}")

print(f"\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=['Non-Vehicle (P/B/N)', 'Vehicle (V)']))

# Feature importance (coefficient magnitudes)
print(f"\nFeature Importance (by coefficient magnitude):")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)
print(feature_importance)

# Add model predictions to results dataframe
all_X_scaled = scaler.transform(results_df[feature_cols].fillna(0))
results_df['predicted_vehicle'] = model.predict(all_X_scaled)
results_df['vehicle_probability'] = model.predict_proba(results_df[feature_cols].fillna(0).values)[:, 1]

# Create output with selected columns for readability
output_df = results_df[[
    'start_index', 'end_index', 'type', 'predicted_vehicle', 'vehicle_probability',
    'duration', 'derivative_sign_changes', 'max_depth', 'smoothness', 'max_slope',
    'vehicle_confidence'
]].copy()

# Rename columns for clarity in output
output_df.rename(columns={
    'predicted_vehicle': 'ML_Predicted_Vehicle',
    'vehicle_probability': 'ML_Vehicle_Probability',
    'vehicle_confidence': 'Rule_Based_Confidence'
}, inplace=True)

# Save to both CSV and Excel
output_df.to_csv('dip_analysis_results_with_ml.csv', index=False)
print(f"\nCSV results saved to 'dip_analysis_results_with_ml.csv'")

# Install openpyxl if needed and save to Excel
try:
    import time
    # Give a small delay and try to save
    time.sleep(0.5)
    output_df.to_excel('dip_analysis_results_with_ml.xlsx', index=False, engine='openpyxl')
    print(f"Excel results saved to 'dip_analysis_results_with_ml.xlsx'")
except PermissionError:
    print("Warning: Could not save Excel file (file may be open in Excel). CSV saved successfully.")
except ImportError:
    print("Note: openpyxl not installed. Install it to save Excel files.")
    print("Run: pip install openpyxl")

# Show summary of ML predictions vs actual types
print(f"\n" + "="*80)
print("ML Model Predictions Summary")
print("="*80)
print("\nActual vs Predicted (high confidence predictions):")
high_conf = results_df[results_df['vehicle_probability'] > 0.7]
print(f"High confidence vehicle predictions: {high_conf['predicted_vehicle'].sum()}")
print(f"  - True vehicles (V): {len(high_conf[high_conf['type'] == 'V'])}")
print(f"  - Other types: {len(high_conf[high_conf['type'] != 'V'])}")


