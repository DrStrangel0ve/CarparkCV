import pandas as pd
import numpy as np
from scipy import integrate
from pathlib import Path


class DipAnalyzer:
    """
    Analyzes depth dips in video frame data and generates statistical metrics.
    """
    
    def __init__(self, csv_path, baseline_depth, min_duration=0.2, merge_gap=0.3, column_mapping=None):
        """
        Initialize the DipAnalyzer.
        
        Args:
            csv_path (str): Path to input CSV file
            baseline_depth (float): Depth threshold below which a dip is detected
            min_duration (float): Minimum duration in seconds to keep a dip
            merge_gap (float): Maximum gap in seconds to merge adjacent dips
            column_mapping (dict): Custom column name mapping
        """
        self.csv_path = csv_path
        self.baseline_depth = baseline_depth
        self.min_duration = min_duration
        self.merge_gap = merge_gap
        self.df = None
        self.dips = []
        
        # Default column mapping
        self.col_map = {
            'time': 'time_seconds',
            'people': 'people_in_frame',
            'bicycles': 'bicycles_in_frame',
            'vehicles': 'vehicles_in_frame',
            'depth': 'ultrawide_depth_m'
        }
        
        # Override with custom mapping if provided
        if column_mapping:
            self.col_map.update(column_mapping)
        
        self.load_data()
    
    def load_data(self):
        """Load and validate CSV data."""
        self.df = pd.read_csv(self.csv_path)
        
        # Validate required columns
        required_cols = list(self.col_map.values())
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"Loaded {len(self.df)} frames from {self.csv_path}")
    
    def detect_dips(self):
        """
        Detect dips where depth goes below baseline and returns above baseline.
        Returns list of tuples: (start_idx, end_idx)
        """
        depth_col = self.col_map['depth']
        depth = self.df[depth_col].values
        
        # Create boolean array for below baseline
        below_baseline = depth < self.baseline_depth
        
        # Find transitions
        dips = []
        in_dip = False
        start_idx = None
        
        for i in range(len(below_baseline)):
            if below_baseline[i] and not in_dip:
                # Start of dip (frame before this is x-1)
                start_idx = max(0, i - 1)
                in_dip = True
            elif not below_baseline[i] and in_dip:
                # End of dip (current frame is y)
                end_idx = i
                dips.append((start_idx, end_idx))
                in_dip = False
        
        # Handle case where data ends during a dip
        if in_dip and start_idx is not None:
            dips.append((start_idx, len(self.df) - 1))
        
        # Post-process dips
        self.dips = self.merge_and_filter_dips(dips)
        print(f"Detected {len(self.dips)} dips (merged from {len(dips)} raw events)")
        return self.dips

    def merge_and_filter_dips(self, raw_dips):
        """Merge close dips and filter out short ones."""
        if not raw_dips:
            return []
            
        time_col = self.col_map['time']
        times = self.df[time_col].values
        
        # 1. Merge close dips
        merged_dips = []
        if raw_dips:
            current_start, current_end = raw_dips[0]
            
            for i in range(1, len(raw_dips)):
                next_start, next_end = raw_dips[i]
                
                # Calculate gap duration
                gap = times[next_start] - times[current_end]
                
                if gap < self.merge_gap:
                    # Merge: extend current end to next end
                    current_end = next_end
                else:
                    # Save current and start new
                    merged_dips.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
            
            merged_dips.append((current_start, current_end))
        
        # 2. Filter short dips
        final_dips = []
        for start, end in merged_dips:
            duration = times[end] - times[start]
            if duration >= self.min_duration:
                final_dips.append((start, end))
                
        return final_dips
    
    def classify_object(self, dip_slice):
        """
        Classify the object type based on dip data.
        V: Vehicle (vehicles > 0)
        B: Bicycle (bicycles > 0, no vehicles)
        P: Person (people > 0, no bicycles/vehicles)
        N: None/Nothing
        """
        vehicles = dip_slice[self.col_map['vehicles']].max()
        bicycles = dip_slice[self.col_map['bicycles']].max()
        people = dip_slice[self.col_map['people']].max()
        
        if vehicles > 0:
            return 'V'
        elif bicycles > 0:
            return 'B'
        elif people > 0:
            return 'P'
        else:
            return 'N'
    
    def calculate_line_integral(self, depth_values):
        """
        Calculate line integral (area under curve) of depth values.
        Uses trapezoidal integration.
        """
        if len(depth_values) < 2:
            return 0.0
        
        # Normalize indices to [0, 1] for proper integration
        x = np.linspace(0, 1, len(depth_values))
        integral = integrate.trapezoid(depth_values, x)
        return float(integral)
    
    def calculate_depth_gradient(self, depth_values):
        """Calculate average gradient (rate of change) of depth."""
        if len(depth_values) < 2:
            return 0.0
        
        gradients = np.diff(depth_values)
        return float(np.mean(np.abs(gradients)))

    def calculate_max_depth_gradient(self, depth_values):
        """Calculate maximum gradient (rate of change) of depth."""
        if len(depth_values) < 2:
            return 0.0
        
        gradients = np.diff(depth_values)
        return float(np.max(np.abs(gradients)))
    
    def calculate_depth_variance(self, depth_values):
        """Calculate variance of depth values."""
        return float(np.var(depth_values))

    def calculate_depth_fourth_moment(self, depth_values):
        """Calculate the 4th moment of depth values (punishes outliers more)."""
        if len(depth_values) < 1:
            return 0.0
        mean_val = np.mean(depth_values)
        fourth_moment = np.mean((depth_values - mean_val) ** 4)
        return float(fourth_moment)
    
    def calculate_dip_duration(self, time_values):
        """Calculate dip duration in seconds."""
        if len(time_values) < 2:
            return 0.0
        return float(time_values.iloc[-1] - time_values.iloc[0])
    
    def calculate_shape_metrics(self, depth_values, time_values):
        """Calculate shape-based metrics for object classification."""
        if len(depth_values) < 3:
            return {
                'dip_height': 0.0,
                'fill_factor': 0.0,
                'plateau_variance': 0.0,
                'shape_category': 'Undefined'
            }

        # 1. Dip Height (Ground - Top of Object)
        # Assuming max_depth represents the ground/baseline level during this window
        ground_level = np.max(depth_values)
        min_depth = np.min(depth_values)
        dip_height = ground_level - min_depth
        
        if dip_height <= 0:
            return {
                'dip_height': 0.0,
                'fill_factor': 0.0,
                'plateau_variance': 0.0,
                'shape_category': 'Noise'
            }

        # 2. Fill Factor (Boxiness)
        # Area of the "object" (space between ground and depth curve)
        # We use normalized x [0,1] to match previous integral style
        object_profile = ground_level - depth_values
        x = np.linspace(0, 1, len(depth_values))
        actual_area = integrate.trapezoid(object_profile, x)
        
        # Bounding box area in this normalized space is just dip_height * 1.0
        bounding_area = dip_height
        
        fill_factor = actual_area / bounding_area if bounding_area > 0 else 0

        # 3. Plateau Variance (Flatness of top)
        # Take middle 50% of the frames
        n_samples = len(depth_values)
        start_mid = int(n_samples * 0.25)
        end_mid = int(n_samples * 0.75)
        
        if end_mid > start_mid:
            middle_section = depth_values[start_mid:end_mid]
            plateau_variance = np.var(middle_section)
        else:
            plateau_variance = 0.0

        # 4. Categorization
        # Heuristics for vehicles:
        # - Boxy (High fill factor)
        # - Flat top (Low variance)
        
        is_boxy = fill_factor > 0.65  # Rectangle is 1.0, Triangle is 0.5. Cars are > 0.7 usually.
        is_flat = plateau_variance < 0.05  # Assuming meters. 5cm variance? 
        
        if is_boxy and is_flat:
            category = 'Boxy_Flat (Likely Vehicle)'
        elif is_boxy:
            category = 'Boxy_Irregular'
        elif is_flat:
            category = 'Peaked_Flat'
        else:
            category = 'Peaked_Irregular (Likely Pedestrian/Bike)'

        return {
            'dip_height': dip_height,
            'fill_factor': fill_factor,
            'plateau_variance': plateau_variance,
            'shape_category': category
        }

    def analyze_dips(self):
        """
        Analyze each detected dip and return results dataframe.
        """
        if not self.dips:
            self.detect_dips()
        
        results = []
        
        for start_idx, end_idx in self.dips:
            dip_slice = self.df.iloc[start_idx:end_idx + 1]
            
            time_col = self.col_map['time']
            depth_col = self.col_map['depth']
            
            time_values = dip_slice[time_col]
            depth_values = dip_slice[depth_col].values
            
            # Calculate metrics
            dip_time = float(time_values.iloc[0])
            max_depth = float(depth_values.max())
            avg_depth = float(depth_values.mean())
            obj_type = self.classify_object(dip_slice)
            
            # Additional metrics
            dip_duration = self.calculate_dip_duration(time_values)
            line_integral = self.calculate_line_integral(depth_values)
            depth_gradient = self.calculate_depth_gradient(depth_values)
            max_depth_gradient = self.calculate_max_depth_gradient(depth_values)
            depth_variance = self.calculate_depth_variance(depth_values)
            depth_fourth_moment = self.calculate_depth_fourth_moment(depth_values)
            min_depth = float(depth_values.min())
            
            # Shape metrics
            shape_metrics = self.calculate_shape_metrics(depth_values, time_values)
            
            # Count objects at various thresholds
            avg_vehicles = float(dip_slice[self.col_map['vehicles']].mean())
            avg_bicycles = float(dip_slice[self.col_map['bicycles']].mean())
            avg_people = float(dip_slice[self.col_map['people']].mean())
            
            result = {
                'time_of_dip_seconds': dip_time,
                'max_depth': max_depth,
                'min_depth': min_depth,
                'avg_depth': avg_depth,
                'object': obj_type,
                'dip_duration_seconds': dip_duration,
                'line_integral': line_integral,
                'avg_depth_gradient': depth_gradient,
                'max_depth_gradient': max_depth_gradient,
                'depth_variance': depth_variance,
                'depth_fourth_moment': depth_fourth_moment,
                'avg_vehicles': avg_vehicles,
                'avg_bicycles': avg_bicycles,
                'avg_people': avg_people,
                'frame_count': end_idx - start_idx + 1,
                'start_frame_idx': start_idx,
                'end_frame_idx': end_idx
            }
            
            # Add shape metrics to result
            result.update(shape_metrics)
            
            results.append(result)
        
        self.results_df = pd.DataFrame(results)
        return self.results_df
    
    def save_results(self, output_path):
        """Save analysis results to CSV file."""
        if self.results_df is None or len(self.results_df) == 0:
            print("No dips analyzed. Run analyze_dips() first.")
            return
        
        self.results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    def get_summary_statistics(self):
        """Print summary statistics of detected dips."""
        if self.results_df is None or len(self.results_df) == 0:
            print("No dips analyzed.")
            return
        
        print("\n" + "="*60)
        print("DIP ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total dips detected: {len(self.results_df)}")
        print(f"Average dip duration: {self.results_df['dip_duration_seconds'].mean():.2f} seconds")
        print(f"Average max depth: {self.results_df['max_depth'].mean():.2f}")
        print(f"Average depth variance: {self.results_df['depth_variance'].mean():.2f}")
        
        # Object type distribution
        obj_dist = self.results_df['object'].value_counts()
        print("\nObject Type Distribution:")
        for obj, count in obj_dist.items():
            print(f"  {obj}: {count} ({count/len(self.results_df)*100:.1f}%)")
        
        print("\nDepth Statistics:")
        print(f"  Min: {self.results_df['min_depth'].min():.2f}")
        print(f"  Max: {self.results_df['max_depth'].max():.2f}")
        print(f"  Mean: {self.results_df['avg_depth'].mean():.2f}")
        print("="*60 + "\n")


def main():
    """Main entry point."""
    # Configuration parameters
    input_csv = "rak_data_with_interpolated_cv.csv"
    baseline = 220
    min_duration = 0.2
    merge_gap = 0.3
    output_csv = "dip_analysis_results.csv"
    
    # Validate input file
    if not input_csv:
        print("Error: input_csv parameter is empty")
        return
    
    if not Path(input_csv).exists():
        print(f"Error: Input file '{input_csv}' not found")
        return
    
    # Set output path if not specified
    if not output_csv:
        input_path = Path(input_csv)
        output_csv = str(input_path.parent / f"{input_path.stem}_dips.csv")
    
    try:
        # Create analyzer and run analysis
        analyzer = DipAnalyzer(input_csv, baseline, min_duration=min_duration, merge_gap=merge_gap)
        analyzer.detect_dips()
        analyzer.analyze_dips()
        analyzer.save_results(output_csv)
        analyzer.get_summary_statistics()
        
        print(f"\nAnalysis complete! Results saved to {output_csv}")
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == '__main__':
    main()
