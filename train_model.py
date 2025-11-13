"""
Predictive Maintenance Model Training
Random Forest Classifier for 30-day failure prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def load_features(filepath='features_training_data.csv'):
    """Load engineered features"""
    print("Loading feature data...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} observations")
    return df


def prepare_data(df):
    """Split features and target, create train/test sets"""
    # Define feature columns (exclude metadata and target)
    feature_cols = [
        'current_mileage',
        'age_years',
        'days_since_oil_change',
        'days_since_brake_service',
        'days_since_transmission_service',
        'miles_since_oil_change',
        'miles_since_brake_service',
        'maintenance_compliance_score',
        'total_failures_history',
        'failures_last_year',
        'days_since_last_failure',
        'avg_failure_cost_history',
        'avg_oil_pressure_30d',
        'avg_coolant_temp_30d',
        'avg_load_weight_30d',
        'oil_pressure_variability',
        'coolant_temp_variability',
        'heavy_load_percentage',
        'avg_daily_miles'
    ]

    X = df[feature_cols]
    y = df['failure_next_30_days']

    # Handle any missing values
    X = X.fillna(0)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution:")
    print(f"  No failure: {(y == 0).sum()} ({(y == 0).mean() * 100:.1f}%)")
    print(f"  Failure: {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%)")

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {len(X_train)} observations")
    print(f"Test set: {len(X_test)} observations")

    return X_train, X_test, y_train, y_test, feature_cols


def train_model(X_train, y_train):
    """Train Random Forest classifier"""
    print("\nTraining Random Forest model...")

    # Random Forest with balanced class weights
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',  # Handle imbalanced classes
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Model training complete!")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_cols):
    """Comprehensive model evaluation"""
    print("\nMODEL EVALUATION")

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]

    # Training performance
    print("\nTraining Set")
    print(classification_report(y_train, y_pred_train, target_names=['No Failure', 'Failure']))

    # Test performance
    print("\nTest Set")
    print(classification_report(y_test, y_pred_test, target_names=['No Failure', 'Failure']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print("\nConfusion Matrix (Test):")
    print(f"                 Predicted No  Predicted Yes")
    print(f"Actual No        {cm[0, 0]:>12}  {cm[0, 1]:>13}")
    print(f"Actual Yes       {cm[1, 0]:>12}  {cm[1, 1]:>13}")

    # ROC AUC
    roc_auc_train = roc_auc_score(y_train, y_pred_proba_train)
    roc_auc_test = roc_auc_score(y_test, y_pred_proba_test)

    print(f"\nROC AUC Score:")
    print(f"  Training: {roc_auc_train:.3f}")
    print(f"  Test: {roc_auc_test:.3f}")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features")
    print(importance_df.head(10).to_string(index=False))

    return importance_df


def plot_feature_importance(importance_df, top_n=15):
    """Visualize feature importance"""
    plt.figure(figsize=(10, 6))

    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top Feature Importances - Predictive Maintenance Model')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\nFeature importance plot saved: feature_importance.png")


def plot_roc_curve(model, X_test, y_test):
    """Plot ROC curve"""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Failure Prediction Model')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print("ROC curve saved: roc_curve.png")


def save_model(model, filename='predictive_maintenance_model.pkl'):
    """Save trained model to disk"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved: {filename}")


def calculate_cost_benefit(y_test, y_pred_proba, threshold=0.5):
    """Calculate business value of predictive maintenance"""
    # Cost assumptions
    PREVENTIVE_MAINTENANCE_COST = 500  # Planned maintenance
    BREAKDOWN_COST = 2500  # Average emergency repair
    DOWNTIME_COST = 3000  # Lost revenue from downtime

    TOTAL_BREAKDOWN_COST = BREAKDOWN_COST + DOWNTIME_COST

    # Apply threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate outcomes
    true_positives = ((y_test == 1) & (y_pred == 1)).sum()
    false_positives = ((y_test == 0) & (y_pred == 1)).sum()
    false_negatives = ((y_test == 1) & (y_pred == 0)).sum()
    true_negatives = ((y_test == 0) & (y_pred == 0)).sum()

    # Cost analysis
    # TP: Caught failure early - pay preventive maintenance, avoid breakdown
    cost_tp = true_positives * PREVENTIVE_MAINTENANCE_COST
    savings_tp = true_positives * TOTAL_BREAKDOWN_COST

    # FP: False alarm - unnecessary preventive maintenance
    cost_fp = false_positives * PREVENTIVE_MAINTENANCE_COST

    # FN: Missed failure - full breakdown cost
    cost_fn = false_negatives * TOTAL_BREAKDOWN_COST

    # TN: Correctly predicted no failure - no cost
    cost_tn = 0

    total_cost_with_model = cost_tp + cost_fp + cost_fn + cost_tn
    total_cost_no_model = (true_positives + false_negatives) * TOTAL_BREAKDOWN_COST

    savings = total_cost_no_model - total_cost_with_model

    print("\nCOST-BENEFIT ANALYSIS")
    print(f"\nThreshold: {threshold}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (caught failures): {true_positives}")
    print(f"  False Positives (false alarms): {false_positives}")
    print(f"  False Negatives (missed failures): {false_negatives}")
    print(f"  True Negatives (correct no-failure): {true_negatives}")

    print(f"\nCost Breakdown:")
    print(f"  Cost from preventive maintenance (TP): ${cost_tp:,.0f}")
    print(f"  Cost from false alarms (FP): ${cost_fp:,.0f}")
    print(f"  Cost from missed failures (FN): ${cost_fn:,.0f}")
    print(f"  TOTAL COST WITH MODEL: ${total_cost_with_model:,.0f}")

    print(f"\nBaseline (no model):")
    print(f"  All failures result in breakdowns: ${total_cost_no_model:,.0f}")

    print(f"\nNET SAVINGS: ${savings:,.0f}")
    print(f"ROI: {(savings / total_cost_with_model) * 100:.1f}%")

    return {
        'savings': savings,
        'cost_with_model': total_cost_with_model,
        'cost_no_model': total_cost_no_model,
        'roi_percent': (savings / total_cost_with_model) * 100
    }


def main():
    """Main training pipeline"""
    # Load data
    df = load_features()

    # Prepare train/test sets
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    importance_df = evaluate_model(model, X_train, X_test, y_train, y_test, feature_cols)

    # Visualizations
    plot_feature_importance(importance_df)
    plot_roc_curve(model, X_test, y_test)

    # Cost-benefit analysis
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    cost_benefit = calculate_cost_benefit(y_test, y_pred_proba, threshold=0.5)

    # Test different thresholds
    print("\nTHRESHOLD SENSITIVITY")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        result = calculate_cost_benefit(y_test, y_pred_proba, threshold=threshold)
        print(f"Threshold {threshold}: Savings = ${result['savings']:,.0f}")

    # Save model
    save_model(model)

    print("\nTRAINING COMPLETE")
    return model, importance_df


if __name__ == "__main__":
    model, importance_df = main()