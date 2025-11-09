"""
Data Fusion Module: Combines CV Detection Data with Ultrasonic Sensor Data
============================================================================

This script reads two CSV files:
1. CV Processing results (from CV_Vehicle_tracking.py)
2. Ultrasonic sensor data (distance/time measurements)

It aligns the timestamps, merges the data, and provides visualization and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class DataFusion:
    """
    Fuses CV detection data with ultrasonic sensor measurements.
    Handles timestamp alignment and provides analysis/visualization.
    """
    
    def __init__(self, cv_csv_path: str, sensor_csv_path: str):
        """
        Initialize the DataFusion object with paths to both CSV files.
        
        Args:
            cv_csv_path: Path to CV detection results CSV (from CV_Vehicle_tracking.py)
            sensor_csv_path: Path to ultrasonic sensor data CSV
        """
        self.cv_csv_path = cv_csv_path
        self.sensor_csv_path = sensor_csv_path
        self.cv_data = None
        self.sensor_data = None
        self.merged_data = None
        
        print("=" * 70)
        print("DATA FUSION: CV DETECTION + ULTRASONIC SENSOR")
        print("=" * 70)
        print()
        
        # Load both CSV files
        self._load_cv_data()
        self._load_sensor_data()
    
    def _load_cv_data(self):
        """Load and validate CV detection CSV."""
        try:
            self.cv_data = pd.read_csv(self.cv_csv_path)
            print(f"✓ Loaded CV data from: {self.cv_csv_path}")
            print(f"  Rows: {len(self.cv_data)}, Columns: {len(self.cv_data.columns)}")
            print(f"  Time range: {self.cv_data['time_seconds'].min():.2f}s to {self.cv_data['time_seconds'].max():.2f}s")
            print(f"  Expected columns: time_seconds, frame_number, people_in_frame, bicycles_in_frame, vehicles_in_frame, etc.")
            print()
        except FileNotFoundError:
            raise FileNotFoundError(f"CV CSV file not found: {self.cv_csv_path}")
        except Exception as e:
            raise Exception(f"Error loading CV CSV: {e}")
    
    def _load_sensor_data(self):
        """Load and validate ultrasonic sensor CSV."""
        try:
            self.sensor_data = pd.read_csv(self.sensor_csv_path)
            print(f"✓ Loaded sensor data from: {self.sensor_csv_path}")
            print(f"  Rows: {len(self.sensor_data)}, Columns: {len(self.sensor_data.columns)}")
            
            # Try to identify time column
            time_col = self._find_time_column()
            distance_col = self._find_distance_column()
            
            if time_col:
                print(f"  Time column: '{time_col}'")
                print(f"  Time range: {self.sensor_data[time_col].min():.2f}s to {self.sensor_data[time_col].max():.2f}s")
            if distance_col:
                print(f"  Distance column: '{distance_col}'")
                print(f"  Distance range: {self.sensor_data[distance_col].min():.2f} to {self.sensor_data[distance_col].max():.2f}")
            print()
        except FileNotFoundError:
            raise FileNotFoundError(f"Sensor CSV file not found: {self.sensor_csv_path}")
        except Exception as e:
            raise Exception(f"Error loading sensor CSV: {e}")
    
    def _find_time_column(self) -> Optional[str]:
        """
        Automatically detect time column in sensor data.
        Looks for common column names.
        """
        time_candidates = ['time', 'time_seconds', 'timestamp', 'time_s', 't', 
                          'Time', 'Timestamp', 'TIME', 'TIMESTAMP']
        for col in time_candidates:
            if col in self.sensor_data.columns:
                return col
        return None
    
    def _find_distance_column(self) -> Optional[str]:
        """
        Automatically detect distance column in sensor data.
        Looks for common column names.
        """
        distance_candidates = ['distance', 'distance_cm', 'distance_m', 'dist', 
                              'Distance', 'DISTANCE', 'dist_cm', 'range']
        for col in distance_candidates:
            if col in self.sensor_data.columns:
                return col
        return None
    
    def merge(self, time_column: Optional[str] = None, 
              distance_column: Optional[str] = None,
              method: str = 'nearest') -> pd.DataFrame:
        """
        Merge CV and sensor data based on timestamp alignment.
        
        Args:
            time_column: Name of time column in sensor data (auto-detected if None)
            distance_column: Name of distance column in sensor data (auto-detected if None)
            method: 'nearest' (default) or 'interpolate'
                   'nearest': Match each CV row to nearest sensor timestamp
                   'interpolate': Interpolate sensor values at each CV timestamp
        
        Returns:
            Merged DataFrame with both CV and sensor data
        """
        if time_column is None:
            time_column = self._find_time_column()
            if time_column is None:
                raise ValueError("Could not auto-detect time column. Please specify 'time_column' parameter.")
        
        if distance_column is None:
            distance_column = self._find_distance_column()
            if distance_column is None:
                raise ValueError("Could not auto-detect distance column. Please specify 'distance_column' parameter.")
        
        print(f"Merging data using '{method}' method...")
        print(f"  CV time column: 'time_seconds'")
        print(f"  Sensor time column: '{time_column}'")
        print(f"  Sensor distance column: '{distance_column}'")
        print()
        
        # Prepare sensor data
        sensor_prep = self.sensor_data[[time_column, distance_column]].copy()
        sensor_prep.columns = ['sensor_time', 'distance']
        sensor_prep = sensor_prep.sort_values('sensor_time').reset_index(drop=True)
        
        if method == 'nearest':
            self.merged_data = self._merge_nearest(sensor_prep)
        elif method == 'interpolate':
            self.merged_data = self._merge_interpolate(sensor_prep)
        else:
            raise ValueError(f"Unknown merge method: {method}. Use 'nearest' or 'interpolate'.")
        
        print(f"✓ Merged successfully!")
        print(f"  Total rows: {len(self.merged_data)}")
        print(f"  Columns: {list(self.merged_data.columns)}")
        print()
        
        return self.merged_data
    
    def _merge_nearest(self, sensor_prep: pd.DataFrame) -> pd.DataFrame:
        """Merge by finding nearest sensor timestamp for each CV timestamp."""
        merged = self.cv_data.copy()
        
        distances = []
        for cv_time in merged['time_seconds']:
            # Find nearest sensor time
            idx = (sensor_prep['sensor_time'] - cv_time).abs().idxmin()
            distances.append(sensor_prep.loc[idx, 'distance'])
        
        merged['distance'] = distances
        return merged
    
    def _merge_interpolate(self, sensor_prep: pd.DataFrame) -> pd.DataFrame:
        """Merge by interpolating sensor values at each CV timestamp."""
        merged = self.cv_data.copy()
        
        # Interpolate distance values at CV timestamps
        interpolated_distances = np.interp(
            merged['time_seconds'],
            sensor_prep['sensor_time'],
            sensor_prep['distance']
        )
        
        merged['distance'] = interpolated_distances
        return merged
    
    def align_time_offset(self, offset_seconds: float) -> pd.DataFrame:
        """
        Manually adjust time offset between CV and sensor data.
        Useful if recordings didn't start at the same time.
        
        Args:
            offset_seconds: Time offset to add to sensor data (positive = shift sensor later)
        
        Returns:
            Adjusted merged DataFrame
        """
        if self.merged_data is None:
            raise ValueError("Must call merge() first before adjusting time offset.")
        
        print(f"Adjusting time offset by {offset_seconds:+.2f} seconds...")
        
        # Shift sensor times and re-merge
        sensor_prep = self.sensor_data[[self._find_time_column(), self._find_distance_column()]].copy()
        time_col = self._find_time_column()
        dist_col = self._find_distance_column()
        
        sensor_prep.columns = ['sensor_time', 'distance']
        sensor_prep['sensor_time'] = sensor_prep['sensor_time'] + offset_seconds
        sensor_prep = sensor_prep.sort_values('sensor_time').reset_index(drop=True)
        
        self.merged_data = self._merge_nearest(sensor_prep)
        print(f"✓ Time offset applied!")
        print()
        
        return self.merged_data
    
    def get_merged_data(self) -> pd.DataFrame:
        """Return the merged dataset."""
        if self.merged_data is None:
            raise ValueError("No merged data available. Call merge() first.")
        return self.merged_data
    
    def save_merged_data(self, output_path: str = "merged_data.csv"):
        """
        Save merged data to CSV file.
        
        Args:
            output_path: Output CSV file path
        """
        if self.merged_data is None:
            raise ValueError("No merged data available. Call merge() first.")
        
        self.merged_data.to_csv(output_path, index=False)
        print(f"✓ Merged data saved to: {output_path}")
        print()
    
    def plot_comparison(self, output_path: Optional[str] = "fusion_comparison.png"):
        """
        Create a comprehensive comparison plot of CV detections vs sensor distance.
        
        Args:
            output_path: Path to save the plot (None = don't save)
        """
        if self.merged_data is None:
            raise ValueError("No merged data available. Call merge() first.")
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        time = self.merged_data['time_seconds']
        
        # Plot 1: Cumulative vehicle counts
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        
        ax1.plot(time, self.merged_data['total_vehicles_counted'], 'g-', linewidth=2, label='Vehicles (Cumulative)')
        ax1.fill_between(time, self.merged_data['total_vehicles_counted'], alpha=0.2, color='green')
        ax1_twin.plot(time, self.merged_data['distance'], 'r-', linewidth=2, label='Distance (Sensor)')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Cumulative Vehicles Counted', color='g')
        ax1_twin.set_ylabel('Distance (cm/m)', color='r')
        ax1.set_title('Vehicle Detections vs. Sensor Distance Over Time', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor='g')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        # Plot 2: Vehicles in frame vs distance
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        
        ax2.bar(time, self.merged_data['vehicles_in_frame'], width=0.05, alpha=0.6, 
                color='green', label='Vehicles in Frame')
        ax2_twin.plot(time, self.merged_data['distance'], 'r-', linewidth=2, label='Distance')
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Vehicles Visible', color='g')
        ax2_twin.set_ylabel('Distance (cm/m)', color='r')
        ax2.set_title('Vehicles Visible vs. Sensor Distance', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        
        # Plot 3: All detection types
        ax3 = axes[2]
        ax3_twin = ax3.twinx()
        
        ax3.plot(time, self.merged_data['people_in_frame'], 'b-', linewidth=1.5, 
                alpha=0.7, label='People')
        ax3.plot(time, self.merged_data['bicycles_in_frame'], 'm-', linewidth=1.5, 
                alpha=0.7, label='Bicycles')
        ax3.plot(time, self.merged_data['vehicles_in_frame'], 'g-', linewidth=1.5, 
                alpha=0.7, label='Vehicles')
        ax3_twin.plot(time, self.merged_data['distance'], 'r-', linewidth=2.5, label='Distance')
        
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Objects Visible in Frame', color='black')
        ax3_twin.set_ylabel('Distance (cm/m)', color='r')
        ax3.set_title('All Detection Types vs. Sensor Distance', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        ax3_twin.tick_params(axis='y', labelcolor='r')
        
        # Plot 4: Total objects vs distance
        ax4 = axes[3]
        ax4_twin = ax4.twinx()
        
        ax4.plot(time, self.merged_data['total_in_frame'], 'orange', linewidth=2, 
                label='Total Objects in Frame')
        ax4.fill_between(time, self.merged_data['total_in_frame'], alpha=0.2, color='orange')
        ax4_twin.plot(time, self.merged_data['distance'], 'r-', linewidth=2.5, label='Distance')
        
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Total Objects Visible', color='orange')
        ax4_twin.set_ylabel('Distance (cm/m)', color='r')
        ax4.set_title('Total Objects vs. Sensor Distance', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='y', labelcolor='orange')
        ax4_twin.tick_params(axis='y', labelcolor='r')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison plot saved to: {output_path}")
        
        plt.show()
        print()
    
    def correlation_analysis(self) -> dict:
        """
        Calculate correlation between detection counts and sensor distance.
        
        Returns:
            Dictionary of correlation statistics
        """
        if self.merged_data is None:
            raise ValueError("No merged data available. Call merge() first.")
        
        print("=" * 70)
        print("CORRELATION ANALYSIS")
        print("=" * 70)
        
        correlations = {
            'people_vs_distance': self.merged_data['people_in_frame'].corr(self.merged_data['distance']),
            'bicycles_vs_distance': self.merged_data['bicycles_in_frame'].corr(self.merged_data['distance']),
            'vehicles_vs_distance': self.merged_data['vehicles_in_frame'].corr(self.merged_data['distance']),
            'total_vs_distance': self.merged_data['total_in_frame'].corr(self.merged_data['distance']),
        }
        
        print("\nPearson Correlation Coefficients (with Distance):")
        print(f"  People vs Distance:       {correlations['people_vs_distance']:+.4f}")
        print(f"  Bicycles vs Distance:     {correlations['bicycles_vs_distance']:+.4f}")
        print(f"  Vehicles vs Distance:     {correlations['vehicles_vs_distance']:+.4f}")
        print(f"  Total Objects vs Distance: {correlations['total_vs_distance']:+.4f}")
        print()
        print("  Interpretation:")
        print("    -1.0 to -0.7: Strong negative correlation (inverse relationship)")
        print("    -0.7 to -0.3: Moderate negative correlation")
        print("    -0.3 to  0.3: Weak or no correlation")
        print("     0.3 to  0.7: Moderate positive correlation")
        print("     0.7 to  1.0: Strong positive correlation")
        print()
        print("=" * 70)
        print()
        
        return correlations
    
    def summary_statistics(self):
        """Print summary statistics of merged data."""
        if self.merged_data is None:
            raise ValueError("No merged data available. Call merge() first.")
        
        print("=" * 70)
        print("MERGED DATA SUMMARY STATISTICS")
        print("=" * 70)
        print()
        
        print("Time Range:")
        print(f"  Start: {self.merged_data['time_seconds'].min():.2f}s")
        print(f"  End:   {self.merged_data['time_seconds'].max():.2f}s")
        print(f"  Duration: {self.merged_data['time_seconds'].max() - self.merged_data['time_seconds'].min():.2f}s")
        print()
        
        print("Distance Statistics:")
        print(f"  Min: {self.merged_data['distance'].min():.2f}")
        print(f"  Max: {self.merged_data['distance'].max():.2f}")
        print(f"  Mean: {self.merged_data['distance'].mean():.2f}")
        print(f"  Std Dev: {self.merged_data['distance'].std():.2f}")
        print()
        
        print("Detection Statistics:")
        print(f"  Frames with people: {(self.merged_data['people_in_frame'] > 0).sum()} / {len(self.merged_data)}")
        print(f"  Frames with bicycles: {(self.merged_data['bicycles_in_frame'] > 0).sum()} / {len(self.merged_data)}")
        print(f"  Frames with vehicles: {(self.merged_data['vehicles_in_frame'] > 0).sum()} / {len(self.merged_data)}")
        print(f"  Frames with any objects: {(self.merged_data['total_in_frame'] > 0).sum()} / {len(self.merged_data)}")
        print()
        
        print("Cumulative Counts:")
        print(f"  Total people counted: {int(self.merged_data['total_people_counted'].iloc[-1])}")
        print(f"  Total bicycles counted: {int(self.merged_data['total_bicycles_counted'].iloc[-1])}")
        print(f"  Total vehicles counted: {int(self.merged_data['total_vehicles_counted'].iloc[-1])}")
        print()
        print("=" * 70)
        print()


def main():
    """
    Example usage of DataFusion class.
    """
    # Example paths - modify these to your actual file paths
    cv_csv = "detection_results.csv"  # From CV_Vehicle_tracking.py
    sensor_csv = "sensor_data.csv"     # Your ultrasonic sensor data
    
    try:
        # Initialize data fusion
        fusion = DataFusion(cv_csv, sensor_csv)
        
        # Merge the data (auto-detects columns)
        merged = fusion.merge(method='interpolate')
        
        # Optionally adjust time offset if recordings didn't start together
        # merged = fusion.align_time_offset(offset_seconds=5.0)
        
        # Save merged data
        fusion.save_merged_data("merged_cv_sensor_data.csv")
        
        # Print analysis
        fusion.summary_statistics()
        fusion.correlation_analysis()
        
        # Create comparison plot
        fusion.plot_comparison("cv_sensor_comparison.png")
        
        print("\n✓ Data fusion complete!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure both CSV files exist and provide correct paths:")
        print(f"  - CV CSV: {cv_csv}")
        print(f"  - Sensor CSV: {sensor_csv}")
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    main()
