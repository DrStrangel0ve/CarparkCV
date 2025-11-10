"""
Peak Alignment Module: Aligns CV Detection Peaks with Ultrasonic Sensor Dips
=============================================================================

The RAK ultrasonic data shows "dips" (when objects are closer = lower distance values).
The CV data shows "peaks" (when more objects are detected).

This module provides multiple strategies to align these peaks across the two datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


class PeakAligner:
    """
    Aligns peaks between CV detection data and ultrasonic sensor dips.
    """
    
    def __init__(self, cv_csv_path: str, rakdata_csv_path: str):
        """
        Initialize with CV detection and RAK ultrasonic sensor data.
        
        Args:
            cv_csv_path: Path to detection_results.csv
            rakdata_csv_path: Path to RAK_DATA_F2025_FEATURES.csv (already processed features)
        """
        self.cv_data = pd.read_csv(cv_csv_path)
        self.rakdata = pd.read_csv(rakdata_csv_path)
        
        print("=" * 80)
        print("PEAK ALIGNMENT: CV DETECTIONS vs ULTRASONIC SENSOR DIPS")
        print("=" * 80)
        print()
        
        print(f"CV Data loaded:")
        print(f"  Shape: {self.cv_data.shape}")
        print(f"  Time range: {self.cv_data['time_seconds'].min():.2f}s to {self.cv_data['time_seconds'].max():.2f}s")
        print(f"  Duration: {self.cv_data['time_seconds'].max() - self.cv_data['time_seconds'].min():.2f}s")
        print()
        
        print(f"RAK Sensor Data loaded:")
        print(f"  Shape: {self.rakdata.shape}")
        print(f"  Dips detected: {len(self.rakdata)}")
        print()
        
        self.cv_peaks = None
        self.rakdata_peaks = None
        self.alignment_offset = None
        self.merged_aligned = None
    
    def extract_cv_peaks(self, column: str = 'vehicles_in_frame', 
                        prominence_ratio: float = 0.3,
                        distance_frames: int = 10) -> Tuple[np.ndarray, Dict]:
        """
        Extract peaks from CV detection data.
        
        Args:
            column: Which column to analyze ('vehicles_in_frame', 'people_in_frame', 'total_in_frame', etc.)
            prominence_ratio: Peak prominence as ratio of max value (0-1)
            distance_frames: Minimum distance between peaks in frames
        
        Returns:
            Tuple of (peak_indices, peak_info_dict)
        """
        data = self.cv_data[column].values
        
        if data.max() == 0:
            print(f"⚠ Warning: Column '{column}' has no detections (all zeros)")
            return np.array([]), {}
        
        # Calculate prominence threshold
        prominence_threshold = data.max() * prominence_ratio
        
        # Find peaks
        peaks, properties = find_peaks(
            data,
            prominence=prominence_threshold,
            distance=distance_frames
        )
        
        print(f"CV Peak Detection ({column}):")
        print(f"  Column max: {data.max()}")
        print(f"  Peaks found: {len(peaks)}")
        print(f"  Peak indices: {peaks}")
        print(f"  Peak values: {data[peaks]}")
        print(f"  Peak times (s): {self.cv_data.iloc[peaks]['time_seconds'].values}")
        print()
        
        self.cv_peaks = peaks
        return peaks, {
            'data': data,
            'peaks': peaks,
            'properties': properties,
            'column': column
        }
    
    def extract_rakdata_peaks(self, time_column: str = 'start_time',
                              depth_column: str = 'max_depth') -> Tuple[List[float], Dict]:
        """
        Extract dip peaks (events) from RAK ultrasonic data.
        RAK data already contains detected dips as rows, so we use start_time as peaks.
        
        Args:
            time_column: Column containing start time of each dip
            depth_column: Column showing depth of dip
        
        Returns:
            Tuple of (dip_times, dip_info_dict)
        """
        dip_times = self.rakdata[time_column].values
        dip_depths = self.rakdata[depth_column].values
        
        print(f"RAK Dip Detection:")
        print(f"  Total dips: {len(dip_times)}")
        print(f"  Dip times (raw units): {dip_times[:10]}...")  # First 10
        print(f"  Dip depths: min={dip_depths.min()}, max={dip_depths.max()}, avg={dip_depths.mean():.2f}")
        print()
        
        self.rakdata_peaks = dip_times
        return dip_times, {
            'times': dip_times,
            'depths': dip_depths,
            'count': len(dip_times)
        }
    
    def estimate_time_scale(self) -> float:
        """
        Estimate the conversion factor from RAK raw time units to video seconds.
        Assumes first and last events happen roughly at same relative positions.
        
        Returns:
            Scale factor (video_seconds / rak_time_units)
        """
        if self.cv_peaks is None or self.rakdata_peaks is None:
            raise ValueError("Must call extract_cv_peaks() and extract_rakdata_peaks() first")
        
        cv_duration = self.cv_data['time_seconds'].max() - self.cv_data['time_seconds'].min()
        rak_duration = self.rakdata_peaks.max() - self.rakdata_peaks.min()
        
        scale = cv_duration / rak_duration if rak_duration > 0 else 1.0
        
        print(f"Time Scale Estimation:")
        print(f"  CV duration: {cv_duration:.2f}s")
        print(f"  RAK duration (raw units): {rak_duration}")
        print(f"  Estimated scale: {scale:.6f} s/unit")
        print()
        
        return scale
    
    def align_by_cross_correlation(self) -> float:
        """
        Align peaks using cross-correlation to find optimal time offset.
        
        Returns:
            Optimal time offset (seconds) to shift RAK data to align with CV data
        """
        if self.cv_peaks is None or self.rakdata_peaks is None:
            raise ValueError("Must call extract_cv_peaks() and extract_rakdata_peaks() first")
        
        print("Cross-Correlation Alignment:")
        
        # Create binary signals (1 where peaks exist, 0 elsewhere)
        cv_signal = np.zeros(len(self.cv_data))
        cv_signal[self.cv_peaks] = 1
        
        # Create RAK signal based on time alignment
        rak_signal = np.zeros(len(self.cv_data))
        
        # Try different time scales
        best_offset = 0
        best_correlation = -1
        best_scale = 1.0
        
        for scale in np.linspace(0.5, 2.0, 30):
            # Convert RAK times to CV times
            rak_times_converted = self.rakdata_peaks * scale + self.cv_data['time_seconds'].min()
            
            # Find which CV frames these correspond to
            for rak_time in rak_times_converted:
                closest_idx = (self.cv_data['time_seconds'] - rak_time).abs().idxmin()
                if 0 <= closest_idx < len(rak_signal):
                    rak_signal[closest_idx] = 1
            
            # Compute correlation
            correlation = np.sum(cv_signal * rak_signal) / (np.sum(cv_signal) + np.sum(rak_signal) + 1e-6)
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_scale = scale
        
        print(f"  Best time scale: {best_scale:.6f} s/unit")
        print(f"  Cross-correlation score: {best_correlation:.4f}")
        print()
        
        self.alignment_offset = best_scale
        return best_scale
    
    def align_by_manual_offset(self, offset_seconds: float) -> None:
        """
        Manually specify time offset for alignment.
        
        Args:
            offset_seconds: Time offset to apply to RAK data
        """
        self.alignment_offset = offset_seconds
        print(f"Manual alignment offset set to: {offset_seconds:.2f}s")
        print()
    
    def create_aligned_dataset(self, offset_seconds: float = None) -> pd.DataFrame:
        """
        Create merged dataset with aligned timestamps.
        
        Args:
            offset_seconds: Time offset (if None, uses previously calculated offset)
        
        Returns:
            Merged DataFrame with both CV and aligned RAK data
        """
        if offset_seconds is not None:
            self.alignment_offset = offset_seconds
        
        if self.alignment_offset is None:
            raise ValueError("No alignment offset set. Call align_by_cross_correlation() or align_by_manual_offset() first")
        
        print(f"Creating aligned dataset with offset {self.alignment_offset:.2f}s...")
        
        # Start with CV data
        merged = self.cv_data.copy()
        
        # Add columns for each RAK dip
        for i, (rak_time, rak_depth) in enumerate(zip(
            self.rakdata['start_time'],
            self.rakdata['max_depth']
        )):
            # RAK time is already in seconds, apply scale offset
            cv_time = rak_time * self.alignment_offset
            
            # Find closest CV frame
            closest_idx = (merged['time_seconds'] - cv_time).abs().idxmin()
            
            # Add marker at this point
            if 'rak_dips' not in merged.columns:
                merged['rak_dips'] = 0
            merged.loc[closest_idx, 'rak_dips'] += 1
        
        # Add RAK depth column (interpolate to each CV frame)
        merged['rak_depth'] = np.interp(
            merged['time_seconds'],
            self.rakdata['start_time'] * self.alignment_offset,
            self.rakdata['max_depth']
        )
        
        self.merged_aligned = merged
        print(f"✓ Aligned dataset created with {len(merged)} rows")
        print()
        
        return merged
    
    def plot_alignment(self, output_path: Optional[str] = "peak_alignment.png") -> None:
        """
        Visualize peak alignment between CV and RAK data.
        
        Args:
            output_path: Path to save plot (None = don't save)
        """
        if self.merged_aligned is None:
            raise ValueError("No aligned data. Call create_aligned_dataset() first")
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        
        time = self.merged_aligned['time_seconds']
        
        # Plot 1: CV vehicles
        ax1 = axes[0]
        ax1.plot(time, self.merged_aligned['vehicles_in_frame'], 'g-', linewidth=2, label='Vehicles in Frame')
        ax1.scatter(time.iloc[self.cv_peaks], 
                   self.merged_aligned['vehicles_in_frame'].iloc[self.cv_peaks], 
                   color='red', s=100, marker='v', label='CV Peaks', zorder=5)
        ax1.set_ylabel('Vehicles Detected', fontsize=11, fontweight='bold')
        ax1.set_title('Peak Alignment: CV Detections vs. Ultrasonic Sensor Dips', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Plot 2: RAK depth (inverted to show as peaks)
        ax2 = axes[1]
        rak_depth_max = self.merged_aligned['rak_depth'].max()
        ax2.fill_between(time, 0, rak_depth_max - self.merged_aligned['rak_depth'], 
                        alpha=0.5, color='blue', label='Ultrasonic Dip Depth')
        ax2.plot(time, rak_depth_max - self.merged_aligned['rak_depth'], 'b-', linewidth=2)
        
        # Mark RAK dips
        dip_times = self.rakdata['start_time'] * self.alignment_offset
        dip_depths = rak_depth_max - self.rakdata['max_depth']
        ax2.scatter(dip_times, dip_depths, color='red', s=100, marker='^', label='RAK Dips', zorder=5)
        
        ax2.set_ylabel('Dip Magnitude', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        
        # Plot 3: Overlay both
        ax3 = axes[2]
        ax3_twin = ax3.twinx()
        
        # Normalize both signals to 0-1 for comparison
        cv_norm = (self.merged_aligned['vehicles_in_frame'] - self.merged_aligned['vehicles_in_frame'].min()) / \
                  (self.merged_aligned['vehicles_in_frame'].max() - self.merged_aligned['vehicles_in_frame'].min() + 1e-6)
        rak_norm = (rak_depth_max - self.merged_aligned['rak_depth']) / (rak_depth_max + 1e-6)
        
        ax3.plot(time, cv_norm, 'g-', linewidth=2.5, label='CV Vehicles (normalized)', alpha=0.8)
        ax3_twin.plot(time, rak_norm, 'b-', linewidth=2.5, label='RAK Dips (normalized)', alpha=0.8)
        
        ax3.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('CV Signal (normalized)', color='g', fontsize=10, fontweight='bold')
        ax3_twin.set_ylabel('RAK Signal (normalized)', color='b', fontsize=10, fontweight='bold')
        ax3.set_title('Normalized Signal Overlay', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='y', labelcolor='g')
        ax3_twin.tick_params(axis='y', labelcolor='b')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Alignment plot saved to: {output_path}")
        
        plt.show()
        print()
    
    def calculate_correlation_score(self) -> float:
        """
        Calculate Pearson correlation between aligned signals.
        
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if self.merged_aligned is None:
            raise ValueError("No aligned data. Call create_aligned_dataset() first")
        
        # Normalize both signals
        cv_vehicles = self.merged_aligned['vehicles_in_frame']
        rak_depth = self.merged_aligned['rak_depth']
        
        # Invert RAK (dip = high value, so invert to match detection peaks)
        rak_inverted = rak_depth.max() - rak_depth
        
        correlation = cv_vehicles.corr(rak_inverted)
        
        print(f"Correlation Score (CV Vehicles vs. RAK Dips):")
        print(f"  Pearson correlation: {correlation:.4f}")
        print()
        
        return correlation
    
    def create_rak_centric_report(self) -> pd.DataFrame:
        """
        Create a RAK-centric report with a single detection type column.
        
        Returns:
            DataFrame with RAK data as base + 'detection_type' column
            detection_type values: 'Vehicle', 'People', 'Bicycle', or 'None'
        """
        if self.merged_aligned is None:
            raise ValueError("No aligned data. Call create_aligned_dataset() first")
        
        print("Creating RAK-centric report with detection types...")
        
        # Start with RAK data
        rak_report = self.rakdata.copy()
        
        # Add detection type column for each dip
        detection_types = []
        
        for idx, row in rak_report.iterrows():
            rak_time = row['start_time']
            
            # RAK time is already in seconds, apply scale offset
            cv_time = rak_time * self.alignment_offset
            
            # Find closest CV frame
            closest_cv_idx = (self.cv_data['time_seconds'] - cv_time).abs().idxmin()
            
            # Get detections at this time
            cv_row = self.cv_data.iloc[closest_cv_idx]
            
            vehicles = int(cv_row['vehicles_in_frame'])
            people = int(cv_row['people_in_frame'])
            bicycles = int(cv_row['bicycles_in_frame'])
            
            # Determine primary detection type (priority: Vehicle > People > Bicycle > None)
            if vehicles > 0:
                detection_type = 'Vehicle'
            elif people > 0:
                detection_type = 'People'
            elif bicycles > 0:
                detection_type = 'Bicycle'
            else:
                detection_type = 'None'
            
            detection_types.append(detection_type)
        
        # Add detection type column to RAK report
        rak_report['detection_type'] = detection_types
        
        print(f"✓ RAK-centric report created with {len(rak_report)} dip events")
        print(f"  Detection type counts:")
        print(f"    Vehicle: {sum(1 for d in detection_types if d == 'Vehicle')}")
        print(f"    People: {sum(1 for d in detection_types if d == 'People')}")
        print(f"    Bicycle: {sum(1 for d in detection_types if d == 'Bicycle')}")
        print(f"    None: {sum(1 for d in detection_types if d == 'None')}")
        print()
        
        return rak_report
    
    def save_rak_centric_data(self, output_path: str = "rak_with_detections.csv") -> None:
        """Save RAK-centric report to CSV."""
        rak_report = self.create_rak_centric_report()
        rak_report.to_csv(output_path, index=False)
        print(f"✓ RAK-centric report saved to: {output_path}")
        print()
        return rak_report
    
    def save_aligned_data(self, output_path: str = "aligned_cv_rak.csv") -> None:
        """Save aligned dataset to CSV."""
        if self.merged_aligned is None:
            raise ValueError("No aligned data. Call create_aligned_dataset() first")
        
        self.merged_aligned.to_csv(output_path, index=False)
        print(f"✓ Aligned data saved to: {output_path}")
        print()
    
    def get_summary(self) -> Dict:
        """Get alignment summary statistics."""
        if self.merged_aligned is None:
            raise ValueError("No aligned data. Call create_aligned_dataset() first")
        
        return {
            'alignment_offset': self.alignment_offset,
            'cv_peaks_count': len(self.cv_peaks) if self.cv_peaks is not None else 0,
            'rak_dips_count': len(self.rakdata_peaks) if self.rakdata_peaks is not None else 0,
            'correlation': self.calculate_correlation_score(),
            'total_rows': len(self.merged_aligned),
            'time_range_seconds': (self.merged_aligned['time_seconds'].max() - 
                                  self.merged_aligned['time_seconds'].min())
        }


def main():
    """Example usage."""
    
    # Paths to your data
    cv_csv = "detection_results.csv"
    rak_csv = "RAK_DATA_F2025.csv"
    
    try:
        # Initialize aligner
        aligner = PeakAligner(cv_csv, rak_csv)
        
        # Extract peaks from CV data
        cv_peaks, cv_info = aligner.extract_cv_peaks(column='vehicles_in_frame', prominence_ratio=0.3)
        
        # Extract peaks from RAK data
        rak_peaks, rak_info = aligner.extract_rakdata_peaks()
        
        # Estimate time scale
        time_scale = aligner.estimate_time_scale()
        
        # Align using cross-correlation
        offset = aligner.align_by_cross_correlation()
        
        # Create aligned dataset
        aligned_data = aligner.create_aligned_dataset()
        
        # Calculate and display correlation
        correlation = aligner.calculate_correlation_score()
        
        # Create visualization
        aligner.plot_alignment("peak_alignment.png")
        
        # Save aligned data
        aligner.save_aligned_data("aligned_cv_rak.csv")
        
        # Save RAK-centric report with detections
        rak_report = aligner.save_rak_centric_data("rak_with_detections.csv")
        
        print("=" * 80)
        print("RAK-CENTRIC REPORT PREVIEW (first 10 rows)")
        print("=" * 80)
        print(rak_report[['start_time', 'end_time', 'max_depth', 'detection_type']].head(10))
        print()
        
        # Print summary
        summary = aligner.get_summary()
        print("=" * 80)
        print("ALIGNMENT SUMMARY")
        print("=" * 80)
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print("=" * 80)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
