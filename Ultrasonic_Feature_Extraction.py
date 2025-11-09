"""
Comprehensive Feature Extraction from Ultrasonic Distance Graphs
For car detection using dip patterns, rise/fall characteristics, and timing
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class UltrasonicFeatureExtractor:
    """
    Extract rich features from ultrasonic distance data to characterize car passage
    
    Key insight: A car creates a DIP in the distance graph:
    - Baseline: Ground to sensor when no car present
    - Dip: Car reduces distance (body reflects signal better)
    - Shape: How quickly distance changes (fall rate, minimum, rise rate)
    - Timing: Duration of dip, spacing between dips (multiple sensors?)
    """
    
    def __init__(self, window_size: int = 50, baseline_window: int = 100):
        """
        Args:
            window_size: Number of samples to analyze at a time
            baseline_window: Samples to use for establishing baseline distance
        """
        self.window_size = window_size
        self.baseline_window = baseline_window
    
    # ============================================================================
    # BASELINE & NORMALIZATION FEATURES
    # ============================================================================
    
    def calculate_baseline_distance(self, distance_array: np.ndarray) -> float:
        """
        Estimate baseline (ground) distance when no car present
        Uses median of entire dataset (cars are brief events)
        """
        return np.median(distance_array)
    
    def calculate_adaptive_baseline(self, distance_array: np.ndarray, window_idx: int) -> float:
        """
        Calculate local baseline around a window (for slowly drifting sensor)
        Uses samples before the window
        """
        start_idx = max(0, window_idx - self.baseline_window)
        local_data = distance_array[start_idx:window_idx]
        if len(local_data) > 0:
            return np.median(local_data)
        return np.median(distance_array)
    
    def normalize_to_baseline(self, distance_array: np.ndarray, baseline: float) -> np.ndarray:
        """
        Normalize distances relative to baseline
        Positive values = dip (car present), negative = noise/fluctuation
        """
        return baseline - distance_array  # Inverted so car = positive dip
    
    # ============================================================================
    # DIP DEPTH & AMPLITUDE FEATURES
    # ============================================================================
    
    def extract_dip_depth_features(self, window: np.ndarray, baseline: float) -> Dict:
        """
        Features describing how deep the dip is (how close car got to sensor)
        """
        normalized = baseline - window  # Car = positive value
        
        return {
            'dip_depth_max': np.max(normalized),           # Deepest point
            'dip_depth_min': np.min(normalized),           # Shallowest point
            'dip_depth_mean': np.mean(normalized),         # Average depth during window
            'dip_depth_std': np.std(normalized),           # Consistency of depth
            'dip_depth_median': np.median(normalized),     # Robust measure
            'dip_depth_range': np.max(normalized) - np.min(normalized),  # Variation
            'dip_depth_q25': np.percentile(normalized, 25),
            'dip_depth_q75': np.percentile(normalized, 75),
            'dip_depth_iqr': np.percentile(normalized, 75) - np.percentile(normalized, 25),
        }
    
    # ============================================================================
    # TEMPORAL FEATURES (Time-based characteristics)
    # ============================================================================
    
    def extract_temporal_features(self, window: np.ndarray, baseline: float, 
                                 time_delta: float = 0.02) -> Dict:
        """
        Features describing how dip evolves over time
        
        Args:
            window: Distance array for this time window
            baseline: Baseline distance (no car)
            time_delta: Time between samples (in seconds, typically 20ms for 50Hz sensor)
        """
        normalized = baseline - window
        
        # First derivative: rate of change (velocity of distance change)
        velocity = np.diff(normalized) / time_delta
        
        # Second derivative: acceleration of distance change
        if len(velocity) > 1:
            acceleration = np.diff(velocity) / time_delta
        else:
            acceleration = np.array([0])
        
        return {
            # Velocity features (rate of dip change)
            'velocity_mean': np.mean(velocity),
            'velocity_std': np.std(velocity),
            'velocity_max': np.max(velocity),
            'velocity_min': np.min(velocity),
            'velocity_range': np.max(velocity) - np.min(velocity),
            
            # Acceleration features (how quickly depth changes)
            'acceleration_mean': np.mean(acceleration) if len(acceleration) > 0 else 0,
            'acceleration_std': np.std(acceleration) if len(acceleration) > 0 else 0,
            'acceleration_max': np.max(acceleration) if len(acceleration) > 0 else 0,
            'acceleration_min': np.min(acceleration) if len(acceleration) > 0 else 0,
            
            # Trend over window (is it getting deeper or shallower?)
            'trend': normalized[-1] - normalized[0],
            'trend_ratio': (normalized[-1] - normalized[0]) / (np.mean(normalized) + 1e-6),
        }
    
    # ============================================================================
    # SHAPE/MORPHOLOGY FEATURES (Dip profile characteristics)
    # ============================================================================
    
    def extract_shape_features(self, window: np.ndarray, baseline: float) -> Dict:
        """
        Features describing the SHAPE of the dip
        (Is it sharp V-shaped? Smooth U-shaped? Multi-peaked?)
        """
        normalized = baseline - window
        
        # Find peaks and valleys in the normalized signal
        peaks, peak_props = signal.find_peaks(normalized, height=0)
        valleys, valley_props = signal.find_peaks(-normalized)
        
        # Skewness and kurtosis (shape metrics)
        from scipy.stats import skew, kurtosis
        
        # Symmetry: is dip symmetric or skewed?
        skewness = skew(normalized)
        kurt = kurtosis(normalized)
        
        # Peakedness: how sharp is the dip?
        # Sharp dips have high kurtosis
        
        return {
            'num_peaks': len(peaks),                        # How many local maxima?
            'num_valleys': len(valleys),                    # How many local minima?
            'skewness': skewness,                           # -1 to 1, 0 = symmetric
            'kurtosis': kurt,                               # >3 = sharp, <3 = flat
            'peak_prominence_mean': np.mean(peak_props.get('prominences', [0])) if len(peaks) > 0 else 0,
            'valley_depth_mean': np.mean(valley_props.get('prominences', [0])) if len(valleys) > 0 else 0,
        }
    
    def extract_curvature_features(self, window: np.ndarray, baseline: float, 
                                   time_delta: float = 0.02) -> Dict:
        """
        Features describing curvature and smoothness of the dip
        """
        normalized = baseline - window
        
        # Velocity and acceleration for curvature calc
        if len(normalized) < 2:
            return {}
        
        v = np.diff(normalized) / time_delta
        if len(v) < 2:
            return {}
        
        a = np.diff(v) / time_delta
        
        # Curvature estimation: |dv/dt| / (1 + v^2)^1.5
        denom = (1 + v**2) ** 1.5
        curvature = np.abs(a) / (denom + 1e-6)
        
        return {
            'curvature_mean': np.mean(curvature),
            'curvature_std': np.std(curvature),
            'curvature_max': np.max(curvature),
            'smoothness': 1 / (1 + np.mean(curvature)),     # Higher = smoother
        }
    
    # ============================================================================
    # RISE/FALL RATE FEATURES (How quickly car enters and leaves)
    # ============================================================================
    
    def extract_rise_fall_features(self, window: np.ndarray, baseline: float, 
                                  time_delta: float = 0.02) -> Dict:
        """
        Features specifically for the FALLING phase (car approaching)
        and RISING phase (car leaving)
        """
        normalized = baseline - window
        
        # Identify descending vs ascending segments
        velocity = np.diff(normalized)
        
        descending_mask = velocity < 0  # Getting deeper (car approaching)
        ascending_mask = velocity > 0   # Getting shallower (car leaving)
        
        fall_velocity = velocity[descending_mask]
        rise_velocity = velocity[ascending_mask]
        
        return {
            # FALL phase (car approaching) - velocity should be negative
            'fall_rate_mean': np.mean(fall_velocity) if len(fall_velocity) > 0 else 0,
            'fall_rate_max': np.min(fall_velocity) if len(fall_velocity) > 0 else 0,  # Most negative
            'fall_rate_std': np.std(fall_velocity) if len(fall_velocity) > 0 else 0,
            'fall_duration_samples': np.sum(descending_mask),
            'fall_asymmetry': np.sum(descending_mask) / max(len(velocity), 1),  # What % is falling?
            
            # RISE phase (car leaving) - velocity should be positive
            'rise_rate_mean': np.mean(rise_velocity) if len(rise_velocity) > 0 else 0,
            'rise_rate_max': np.max(rise_velocity) if len(rise_velocity) > 0 else 0,
            'rise_rate_std': np.std(rise_velocity) if len(rise_velocity) > 0 else 0,
            'rise_duration_samples': np.sum(ascending_mask),
            'rise_asymmetry': np.sum(ascending_mask) / max(len(velocity), 1),  # What % is rising?
            
            # Asymmetry: Is fall faster than rise? (typical for cars)
            'fall_rise_asymmetry': np.sum(descending_mask) - np.sum(ascending_mask),
        }
    
    # ============================================================================
    # FREQUENCY & OSCILLATION FEATURES
    # ============================================================================
    
    def extract_frequency_features(self, window: np.ndarray, baseline: float, 
                                  sampling_rate: float = 50.0) -> Dict:
        """
        Features describing oscillations and frequency content
        (Can distinguish sensor noise from real car signals)
        """
        normalized = baseline - window
        
        # Power spectral density
        if len(normalized) > 4:
            frequencies, psd = signal.welch(normalized, fs=sampling_rate, nperseg=min(len(normalized), 16))
            
            # Spectral centroid (where is the energy concentrated?)
            spectral_centroid = np.sum(frequencies * psd) / (np.sum(psd) + 1e-6)
            
            # Spectral spread
            spectral_spread = np.sqrt(np.sum((frequencies - spectral_centroid)**2 * psd) / (np.sum(psd) + 1e-6))
        else:
            spectral_centroid = 0
            spectral_spread = 0
        
        # Autocorrelation (periodicity)
        if len(normalized) > 10:
            autocorr = np.correlate(normalized - np.mean(normalized), normalized - np.mean(normalized), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # First zero crossing
            zero_crossings = np.sum(np.diff(np.sign(autocorr)) != 0)
        else:
            zero_crossings = 0
        
        return {
            'spectral_centroid': spectral_centroid,
            'spectral_spread': spectral_spread,
            'zero_crossings': zero_crossings,
            'oscillation_energy': np.sum(np.abs(np.diff(normalized))),  # Total variation
        }
    
    # ============================================================================
    # STATISTICAL DISTRIBUTION FEATURES
    # ============================================================================
    
    def extract_distribution_features(self, window: np.ndarray, baseline: float) -> Dict:
        """
        Features describing statistical properties of the dip
        """
        normalized = baseline - window
        
        # Moments
        mean = np.mean(normalized)
        std = np.std(normalized)
        
        # Entropy (disorder in signal)
        # Normalize to histogram
        hist, _ = np.histogram(normalized, bins=10, range=(normalized.min(), normalized.max() + 1e-6))
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        return {
            'mean': mean,
            'std': std,
            'variance': np.var(normalized),
            'cv': std / (np.abs(mean) + 1e-6),             # Coefficient of variation
            'entropy': entropy,                             # Signal disorder
            'dynamic_range': np.max(normalized) - np.min(normalized),
            'rms': np.sqrt(np.mean(normalized**2)),        # Root mean square
        }
    
    # ============================================================================
    # COMPARISON FEATURES (How different from baseline?)
    # ============================================================================
    
    def extract_anomaly_features(self, window: np.ndarray, baseline: float) -> Dict:
        """
        Features measuring how much this window deviates from baseline
        """
        normalized = baseline - window
        
        # How much does this window deviate?
        below_baseline = np.sum(normalized > 0)  # Samples that are deeper than baseline
        above_baseline = np.sum(normalized <= 0)
        
        return {
            'anomaly_score': np.sum(np.abs(normalized)),    # Total deviation
            'anomaly_prevalence': below_baseline / len(normalized),  # What % is dip?
            'max_depth_ratio': np.max(normalized) / (baseline + 1e-6),  # How deep relative to baseline?
            'dip_probability': np.sum(normalized > np.std(normalized)) / len(normalized),  # Outlier %
        }
    
    # ============================================================================
    # COMPOSITE FEATURES (Multiple sensor patterns)
    # ============================================================================
    
    def extract_multi_peak_features(self, window: np.ndarray, baseline: float) -> Dict:
        """
        Features for detecting multi-dip patterns (if car passes multiple sensors)
        or if a train/truck creates multiple peaks
        """
        normalized = baseline - window
        
        # Find local minima
        valleys, valley_props = signal.find_peaks(-normalized, prominence=np.std(normalized) * 0.5)
        
        num_dips = len(valleys)
        
        features = {
            'num_dips': num_dips,
            'dips_spacing_mean': 0,
            'dips_spacing_std': 0,
            'dips_depth_variation': 0,
        }
        
        if len(valleys) > 1:
            spacings = np.diff(valleys)
            features['dips_spacing_mean'] = np.mean(spacings)
            features['dips_spacing_std'] = np.std(spacings)
            
            # Depth variation between dips
            depths = -normalized[valleys]
            features['dips_depth_variation'] = np.std(depths) / (np.mean(depths) + 1e-6)
        
        return features
    
    # ============================================================================
    # ENERGY & POWER FEATURES
    # ============================================================================
    
    def extract_energy_features(self, window: np.ndarray, baseline: float, 
                               time_delta: float = 0.02) -> Dict:
        """
        Features measuring the energy/intensity of the signal change
        """
        normalized = baseline - window
        
        # Energy
        energy = np.sum(normalized**2)
        
        # Power (energy per unit time)
        power = energy / (len(normalized) * time_delta)
        
        # Integrated absolute value (area under curve)
        auc = np.sum(np.abs(normalized)) * time_delta
        
        return {
            'energy': energy,
            'power': power,
            'area_under_curve': auc,
            'peak_to_average_ratio': np.max(np.abs(normalized)) / (np.mean(np.abs(normalized)) + 1e-6),
        }
    
    # ============================================================================
    # MASTER FEATURE EXTRACTION
    # ============================================================================
    
    def extract_all_features(self, window: np.ndarray, baseline: float,
                            time_delta: float = 0.02, sampling_rate: float = 50.0) -> Dict:
        """
        Extract ALL features from a window
        """
        all_features = {}
        
        all_features.update(self.extract_dip_depth_features(window, baseline))
        all_features.update(self.extract_temporal_features(window, baseline, time_delta))
        all_features.update(self.extract_shape_features(window, baseline))
        all_features.update(self.extract_curvature_features(window, baseline, time_delta))
        all_features.update(self.extract_rise_fall_features(window, baseline, time_delta))
        all_features.update(self.extract_frequency_features(window, baseline, sampling_rate))
        all_features.update(self.extract_distribution_features(window, baseline))
        all_features.update(self.extract_anomaly_features(window, baseline))
        all_features.update(self.extract_multi_peak_features(window, baseline))
        all_features.update(self.extract_energy_features(window, baseline, time_delta))
        
        return all_features
    
    def extract_features_from_data(self, distance_array: np.ndarray, 
                                  labels: np.ndarray = None,
                                  time_delta: float = 0.02,
                                  sampling_rate: float = 50.0) -> pd.DataFrame:
        """
        Extract features from entire distance array, sliding window style
        
        Args:
            distance_array: Full distance measurements
            labels: Optional labels (1=car, 0=no car) for validation
            time_delta: Time between samples
            sampling_rate: Samples per second
            
        Returns:
            DataFrame with all features for each window
        """
        baseline = self.calculate_baseline_distance(distance_array)
        features_list = []
        
        for i in range(len(distance_array) - self.window_size):
            window = distance_array[i:i+self.window_size]
            features = self.extract_all_features(window, baseline, time_delta, sampling_rate)
            features['window_idx'] = i
            
            if labels is not None:
                features['label'] = labels[i + self.window_size // 2]
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)


# ============================================================================
# VISUALIZATION & ANALYSIS
# ============================================================================

def visualize_feature_extraction(distance_array: np.ndarray, labels: np.ndarray,
                                window_size: int = 50, output_file: str = "features_viz.png"):
    """
    Visualize extracted features and how they relate to car presence
    """
    extractor = UltrasonicFeatureExtractor(window_size=window_size)
    features_df = extractor.extract_features_from_data(distance_array, labels)
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    
    # Plot 1: Raw distance data with labels
    ax = axes[0, 0]
    car_mask = labels == 1
    no_car_mask = labels == 0
    ax.plot(np.where(car_mask)[0], distance_array[car_mask], 'r.', label='Car', alpha=0.6)
    ax.plot(np.where(no_car_mask)[0], distance_array[no_car_mask], 'b.', label='No Car', alpha=0.6)
    ax.set_title('Raw Distance Data')
    ax.set_ylabel('Distance (cm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Dip depth
    ax = axes[0, 1]
    car_features = features_df[features_df['label'] == 1]
    no_car_features = features_df[features_df['label'] == 0]
    ax.hist([car_features['dip_depth_max'], no_car_features['dip_depth_max']], 
            label=['Car', 'No Car'], alpha=0.7)
    ax.set_title('Max Dip Depth Distribution')
    ax.set_xlabel('Depth (cm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Fall rate
    ax = axes[0, 2]
    ax.hist([car_features['fall_rate_max'].abs(), no_car_features['fall_rate_max'].abs()], 
            label=['Car', 'No Car'], alpha=0.7)
    ax.set_title('Fall Rate Distribution')
    ax.set_xlabel('Rate (cm/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Skewness
    ax = axes[1, 0]
    ax.hist([car_features['skewness'], no_car_features['skewness']], 
            label=['Car', 'No Car'], alpha=0.7)
    ax.set_title('Signal Skewness')
    ax.set_xlabel('Skewness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Anomaly score
    ax = axes[1, 1]
    ax.hist([car_features['anomaly_score'], no_car_features['anomaly_score']], 
            label=['Car', 'No Car'], alpha=0.7, bins=20)
    ax.set_title('Anomaly Score')
    ax.set_xlabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Dip prevalence
    ax = axes[1, 2]
    ax.hist([car_features['anomaly_prevalence'], no_car_features['anomaly_prevalence']], 
            label=['Car', 'No Car'], alpha=0.7)
    ax.set_title('Dip Prevalence (% below baseline)')
    ax.set_xlabel('Prevalence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 7: Velocity mean
    ax = axes[2, 0]
    ax.hist([car_features['velocity_mean'], no_car_features['velocity_mean']], 
            label=['Car', 'No Car'], alpha=0.7)
    ax.set_title('Mean Velocity')
    ax.set_xlabel('Velocity (cm/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Spectral centroid
    ax = axes[2, 1]
    ax.hist([car_features['spectral_centroid'], no_car_features['spectral_centroid']], 
            label=['Car', 'No Car'], alpha=0.7)
    ax.set_title('Spectral Centroid')
    ax.set_xlabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Entropy
    ax = axes[2, 2]
    ax.hist([car_features['entropy'], no_car_features['entropy']], 
            label=['Car', 'No Car'], alpha=0.7)
    ax.set_title('Signal Entropy')
    ax.set_xlabel('Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 10: Smoothness
    ax = axes[3, 0]
    ax.hist([car_features['smoothness'], no_car_features['smoothness']], 
            label=['Car', 'No Car'], alpha=0.7)
    ax.set_title('Signal Smoothness')
    ax.set_xlabel('Smoothness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 11: Energy
    ax = axes[3, 1]
    ax.hist([car_features['energy'], no_car_features['energy']], 
            label=['Car', 'No Car'], alpha=0.7, bins=20)
    ax.set_title('Signal Energy')
    ax.set_xlabel('Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 12: Feature importance summary
    ax = axes[3, 2]
    feature_stats = {}
    for col in features_df.columns:
        if col not in ['window_idx', 'label']:
            car_mean = car_features[col].mean()
            no_car_mean = no_car_features[col].mean()
            # Discriminative power (how different are they?)
            discriminative = abs(car_mean - no_car_mean) / (max(abs(car_mean), abs(no_car_mean)) + 1e-6)
            feature_stats[col] = discriminative
    
    top_features = sorted(feature_stats.items(), key=lambda x: x[1], reverse=True)[:10]
    names = [f[0][:20] for f in top_features]
    scores = [f[1] for f in top_features]
    ax.barh(range(len(names)), scores)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Discriminative Power')
    ax.set_title('Top 10 Features')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved to {output_file}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("COMPREHENSIVE ULTRASONIC FEATURE EXTRACTION")
    print("="*70)
    
    # Generate synthetic data for demonstration
    print("\nGenerating synthetic ultrasonic data with car passages...")
    
    np.random.seed(42)
    
    # Baseline distance (no car)
    baseline = 150  # 150 cm
    
    # Generate time series
    n_samples = 5000
    time = np.arange(n_samples) * 0.02  # 50 Hz sampling (20 ms intervals)
    
    # Add noise to baseline
    distance = baseline + np.random.normal(0, 0.5, n_samples)
    
    # Add car passages (dips)
    labels = np.zeros(n_samples)
    
    # Car passage 1: 500-800
    dip1 = baseline - 30 * np.exp(-((np.arange(300) - 150)**2) / (150**2))  # Gaussian dip
    distance[500:800] = dip1 + np.random.normal(0, 0.5, 300)
    labels[500:800] = 1
    
    # Car passage 2: 1500-1900
    dip2 = baseline - 40 * np.exp(-((np.arange(400) - 200)**2) / (200**2))  # Deeper dip
    distance[1500:1900] = dip2 + np.random.normal(0, 0.5, 400)
    labels[1500:1900] = 1
    
    # Car passage 3: 3000-3300 (sharp dip)
    dip3 = baseline - 35 * (1 / (1 + ((np.arange(300) - 150) / 50)**2))  # Lorentzian dip
    distance[3000:3300] = dip3 + np.random.normal(0, 0.5, 300)
    labels[3000:3300] = 1
    
    print(f"Synthetic data: {n_samples} samples, {np.sum(labels)} car samples")
    
    # Extract features
    print("\nExtracting features...")
    extractor = UltrasonicFeatureExtractor(window_size=50)
    features_df = extractor.extract_features_from_data(distance, labels, time_delta=0.02, sampling_rate=50)
    
    print(f"Extracted {len(features_df)} windows with {len(features_df.columns)} features")
    print(f"\nFeature columns:")
    for i, col in enumerate(features_df.columns):
        if col not in ['window_idx', 'label']:
            print(f"  {i+1:2d}. {col}")
    
    # Show statistics
    print(f"\n{'Feature':<30} {'Car Mean':<12} {'No-Car Mean':<12} {'Difference':<12}")
    print("-" * 66)
    
    car_data = features_df[features_df['label'] == 1]
    no_car_data = features_df[features_df['label'] == 0]
    
    for col in sorted(features_df.columns):
        if col not in ['window_idx', 'label']:
            car_mean = car_data[col].mean()
            no_car_mean = no_car_data[col].mean()
            diff = car_mean - no_car_mean
            print(f"{col:<30} {car_mean:<12.4f} {no_car_mean:<12.4f} {diff:<12.4f}")
    
    # Visualize
    print("\nCreating visualizations...")
    visualize_feature_extraction(distance, labels, window_size=50)
    
    print("\n" + "="*70)
    print("Feature extraction complete!")
    print("="*70)
    print("\nKey findings:")
    print("- Each window is characterized by ~50 different features")
    print("- Features capture: depth, shape, timing, frequency, energy")
    print("- Use these features to train ML classifier")
