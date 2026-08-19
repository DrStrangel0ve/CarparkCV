import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix
import argparse
from pathlib import Path

def train_classifier(input_csv, output_csv=None):
    """
    Train a classifier to identify vehicles based on dip characteristics.
    """
    # 1. Load Data
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 2. Prepare Target and Features
    # Target: Is it a vehicle? (V)
    # We treat 'V' as positive class (1), everything else ('P', 'B', 'N') as negative (0)
    df['is_vehicle'] = (df['object'] == 'V').astype(int)
    
    # Features to use for classification
    feature_cols = [
        'dip_duration_seconds',
        'min_depth',
        'avg_depth',
        'line_integral',
        'avg_depth_gradient',
        'depth_variance',
        'dip_height',
        'fill_factor',
        'plateau_variance'
    ]
    
    X = df[feature_cols]
    y = df['is_vehicle']
    
    # Handle any NaN values
    X = X.fillna(0)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Vehicle count: {y.sum()}")
    print(f"Non-vehicle count: {len(y) - y.sum()}")
    
    # 3. Train/Test Split
    # If dataset is small, we might just train on all for analysis, but let's do a split
    # If very small (< 20 samples), we'll just use the whole set for "analysis" purposes
    if len(df) < 20:
        print("Dataset too small for split. Training on full dataset for feature analysis.")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 4. Train Random Forest (with Regularization)
    # Reduced n_estimators, limited max_depth, increased min_samples_leaf to prevent overfitting
    rf_model = RandomForestClassifier(
        n_estimators=50, 
        max_depth=5, 
        min_samples_leaf=3,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # 5. Train Decision Tree (for Interpretability - Highly Regularized)
    dt_model = DecisionTreeClassifier(
        max_depth=3, 
        min_samples_leaf=5,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    
    # 6. Evaluate with Cross Validation
    # Use Stratified K-Fold to ensure class balance in folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='accuracy')
    
    y_pred = rf_model.predict(X_test)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE (Random Forest)")
    print("="*60)
    print(f"Cross-Validation Accuracy (5-fold): {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    print("\nTest Set Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Vehicle', 'Vehicle']))
    
    # 7. Feature Importance Analysis
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*60)
    print("WHAT MAKES A VEHICLE? (Feature Importance)")
    print("="*60)
    print(feature_importance)
    
    # 8. Decision Rules
    print("\n" + "="*60)
    print("ROBUST RULES (Regularized Decision Tree)")
    print("="*60)
    tree_rules = export_text(dt_model, feature_names=feature_cols)
    print(tree_rules)
    
    # 9. Save Predictions
    # We use the regularized model for predictions
    df['ml_vehicle_prob'] = rf_model.predict_proba(X)[:, 1]
    df['ml_prediction'] = rf_model.predict(X)
    df['ml_prediction_label'] = df['ml_prediction'].map({1: 'Vehicle', 0: 'Non-Vehicle'})
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nResults with ML predictions saved to {output_csv}")
        
    return df, feature_importance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ML model to classify vehicles from dip data')
    parser.add_argument('--input', default='dip_analysis_results.csv', help='Input CSV file')
    parser.add_argument('--output', default='dip_analysis_results_with_ml.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found.")
    else:
        train_classifier(args.input, args.output)
