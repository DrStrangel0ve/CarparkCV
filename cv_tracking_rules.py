"""
CV Tracking Ruleset
Derived from Dip Analysis (dip_analyzer.py and dip_classifier.py)

This module defines the rules for validating vehicle detections in the Computer Vision pipeline,
based on the statistical properties observed in the depth sensor analysis.

INSIGHTS FROM DIP ANALYSIS (ML Model Results):
1. Target Count: The ground truth vehicle count is 24 (down from 28).
   We need stricter rules to filter out the 4 false positives.

2. Feature Importance (Random Forest):
   - depth_variance (0.37): #1 Predictor. Vehicles have very stable depth.
   - plateau_variance (0.21): #2 Predictor. Vehicles have flat tops.
   - avg_depth_gradient (0.19): #3 Predictor. Vehicles have smooth transitions.
   - fill_factor (0.08): #4 Predictor. Vehicles are boxy.
   - dip_duration (0.01): Low importance, but useful for basic noise filtering.

3. Decision Tree Rule:
   - avg_depth_gradient <= 21.08 -> Vehicle.
   - This confirms that "smoothness" and "stability" are the key differentiators.

TRANSLATION TO CV RULES:
- Stability (Variance): We must strictly enforce bounding box stability. 
  Vehicles are rigid; their bounding box area shouldn't fluctuate much.
  We will tighten 'max_size_variance' to 0.15 (was 0.2).
- Boxiness (Aspect Ratio): Still relevant but less critical than stability.
- Duration: Kept at 1.0s as a baseline filter.
"""

import numpy as np

class TrackingRules:
    def __init__(self, fps=30.0):
        self.fps = fps
        
        # Rules derived from Dip Analysis
        self.VEHICLE_RULES = {
            # Duration: Basic noise filter
            'min_duration_seconds': 1.0,
            
            # Aspect Ratio: "Boxiness" (Fill Factor)
            # Vehicles are generally wider than they are tall (or at least not thin vertical lines)
            'min_aspect_ratio': 0.3, 
            'max_aspect_ratio': 3.0,
            
            # Stability: "Variance" (Depth/Plateau Variance)
            # This is the MOST IMPORTANT feature according to ML.
            # We use Coefficient of Variation (std / mean) of the bounding box area.
            # Tightened to 0.15 to reduce false positives (target count 24).
            'max_size_variance': 0.15
        }
        
        self.PERSON_RULES = {
            'min_duration_seconds': 0.5, # People can appear/disappear quickly
            'min_aspect_ratio': 0.1,
            'max_aspect_ratio': 1.5,     # People are generally taller than wide
            'max_size_variance': 0.5     # People change shape more (walking)
        }

    def is_valid_vehicle(self, track_history):
        """
        Check if a track history represents a valid vehicle based on rules.
        
        Args:
            track_history (list): List of detection dicts {'bbox': (x1,y1,x2,y2), 'frame': int, ...}
        
        Returns:
            bool: True if valid, False otherwise
        """
        if not track_history:
            return False
            
        # 1. Duration Check
        start_frame = track_history[0]['frame']
        end_frame = track_history[-1]['frame']
        duration_seconds = (end_frame - start_frame) / self.fps
        
        if duration_seconds < self.VEHICLE_RULES['min_duration_seconds']:
            return False
            
        # 2. Aspect Ratio Check (Average)
        aspect_ratios = []
        areas = []
        for det in track_history:
            x1, y1, x2, y2 = det['bbox']
            w = x2 - x1
            h = y2 - y1
            if h > 0:
                aspect_ratios.append(w / h)
            areas.append(w * h)
            
        avg_ar = np.mean(aspect_ratios)
        if not (self.VEHICLE_RULES['min_aspect_ratio'] <= avg_ar <= self.VEHICLE_RULES['max_aspect_ratio']):
            return False
            
        # 3. Stability Check (Area Variance)
        # We use Coefficient of Variation (std / mean) to be scale-invariant
        if len(areas) > 2:
            area_cv = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
            if area_cv > self.VEHICLE_RULES['max_size_variance']:
                return False
                
        return True

    def is_valid_person(self, track_history):
        """Check if track is a valid person."""
        if not track_history:
            return False
            
        start_frame = track_history[0]['frame']
        end_frame = track_history[-1]['frame']
        duration_seconds = (end_frame - start_frame) / self.fps
        
        if duration_seconds < self.PERSON_RULES['min_duration_seconds']:
            return False
            
        return True
