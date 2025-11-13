"""
Generate Current Fleet Risk Predictions
Creates predictions for all active vehicles for Tableau visualization
"""

import pandas as pd
import numpy as np
import pickle
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('NEON_HOST'),
    'database': os.getenv('NEON_DATABASE'),
    'user': os.getenv('NEON_USER'),
    'password': os.getenv('NEON_PASSWORD'),
    'sslmode': 'require'
}


def load_model(filepath='predictive_maintenance_model.pkl'):
    """Load trained model"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def get_current_fleet_features():
    """Generate features for current fleet state"""
    conn = psycopg2.connect(**DB_CONFIG)

    # Get all data
    vehicles = pd.read_sql("SELECT * FROM vehicles WHERE in_service = TRUE", conn)
    maintenance = pd.read_sql("SELECT * FROM maintenance_events", conn)
    failures = pd.read_sql("SELECT * FROM failure_events", conn)
    telemetry = pd.read_sql("SELECT * FROM telemetry_readings", conn)

    conn.close()

    # Convert dates
    vehicles['purchase_date'] = pd.to_datetime(vehicles['purchase_date'])
    maintenance['event_date'] = pd.to_datetime(maintenance['event_date'])
    failures['failure_date'] = pd.to_datetime(failures['failure_date'])
    telemetry['reading_date'] = pd.to_datetime(telemetry['reading_date'])

    current_date = pd.Timestamp.now()

    features_list = []

    for _, vehicle in vehicles.iterrows():
        vehicle_id = vehicle['vehicle_id']

        # Vehicle characteristics
        age = current_date.year - vehicle['year']
        current_mileage = vehicle['current_mileage']

        # Maintenance recency
        def days_since_maintenance(event_type):
            vm = maintenance[
                (maintenance['vehicle_id'] == vehicle_id) &
                (maintenance['event_type'] == event_type)
                ]
            if len(vm) == 0:
                return 9999
            return (current_date - vm['event_date'].max()).days

        def miles_since_maintenance(event_type):
            vm = maintenance[
                (maintenance['vehicle_id'] == vehicle_id) &
                (maintenance['event_type'] == event_type)
                ]
            if len(vm) == 0:
                return 999999
            return current_mileage - vm['mileage_at_event'].max()

        days_since_oil = days_since_maintenance('oil_change')
        days_since_brakes = days_since_maintenance('brake_service')
        days_since_trans = days_since_maintenance('transmission_service')
        miles_since_oil = miles_since_maintenance('oil_change')
        miles_since_brakes = miles_since_maintenance('brake_service')

        # Maintenance compliance score
        oil_overdue = max(0, (miles_since_oil - 25000) / 25000)
        brake_overdue = max(0, (miles_since_brakes - 60000) / 60000)
        compliance_score = max(0, 100 - (oil_overdue + brake_overdue) * 50)

        # Failure history
        vf = failures[failures['vehicle_id'] == vehicle_id]
        total_failures = len(vf)
        failures_last_year = len(vf[vf['failure_date'] >= current_date - timedelta(days=365)])
        avg_failure_cost = vf['repair_cost'].mean() if len(vf) > 0 else 0
        days_since_last_failure = (current_date - vf['failure_date'].max()).days if len(vf) > 0 else 9999

        # Telemetry (last 30 days)
        start_date = current_date - timedelta(days=30)
        vt = telemetry[
            (telemetry['vehicle_id'] == vehicle_id) &
            (telemetry['reading_date'] >= start_date)
            ]

        if len(vt) > 0:
            avg_oil_pressure = vt['oil_pressure_psi'].mean()
            avg_coolant_temp = vt['coolant_temp_f'].mean()
            avg_load_weight = vt['load_weight_lbs'].mean()
            oil_pressure_std = vt['oil_pressure_psi'].std()
            coolant_temp_std = vt['coolant_temp_f'].std()
            heavy_load_pct = (vt['load_weight_lbs'] > 60000).mean() * 100
            avg_daily_miles = vt['miles_driven'].mean()
        else:
            avg_oil_pressure = 0
            avg_coolant_temp = 0
            avg_load_weight = 0
            oil_pressure_std = 0
            coolant_temp_std = 0
            heavy_load_pct = 0
            avg_daily_miles = 0

        features_list.append({
            'vehicle_id': vehicle_id,
            'make': vehicle['make'],
            'model': vehicle['model'],
            'year': vehicle['year'],
            'current_mileage': current_mileage,
            'age_years': age,
            'days_since_oil_change': days_since_oil,
            'days_since_brake_service': days_since_brakes,
            'days_since_transmission_service': days_since_trans,
            'miles_since_oil_change': miles_since_oil,
            'miles_since_brake_service': miles_since_brakes,
            'maintenance_compliance_score': compliance_score,
            'total_failures_history': total_failures,
            'failures_last_year': failures_last_year,
            'days_since_last_failure': days_since_last_failure,
            'avg_failure_cost_history': avg_failure_cost,
            'avg_oil_pressure_30d': avg_oil_pressure,
            'avg_coolant_temp_30d': avg_coolant_temp,
            'avg_load_weight_30d': avg_load_weight,
            'oil_pressure_variability': oil_pressure_std,
            'coolant_temp_variability': coolant_temp_std,
            'heavy_load_percentage': heavy_load_pct,
            'avg_daily_miles': avg_daily_miles
        })

    return pd.DataFrame(features_list)


def calculate_risk_tier(probability):
    """Categorize risk level"""
    if probability >= 0.7:
        return 'Critical'
    elif probability >= 0.5:
        return 'High'
    elif probability >= 0.3:
        return 'Medium'
    else:
        return 'Low'


def generate_recommendations(row):
    """Generate maintenance recommendations based on features"""
    recommendations = []

    if row['miles_since_oil_change'] > 20000:
        recommendations.append('Schedule oil change (overdue)')
    elif row['miles_since_oil_change'] > 15000:
        recommendations.append('Schedule oil change soon')

    if row['miles_since_brake_service'] > 55000:
        recommendations.append('Schedule brake inspection (overdue)')
    elif row['miles_since_brake_service'] > 45000:
        recommendations.append('Schedule brake inspection soon')

    if row['days_since_transmission_service'] > 180:
        recommendations.append('Transmission service recommended')

    if row['oil_pressure_variability'] > 10:
        recommendations.append('Oil pressure irregularities detected')

    if row['coolant_temp_variability'] > 15:
        recommendations.append('Coolant system check recommended')

    if row['heavy_load_percentage'] > 70:
        recommendations.append('High usage pattern - increase inspection frequency')

    if len(recommendations) == 0:
        recommendations.append('No immediate action required')

    return ' | '.join(recommendations)


def main():
    """Generate predictions for current fleet"""
    print("Loading trained model...")
    model = load_model()

    print("Extracting current fleet features...")
    fleet_df = get_current_fleet_features()

    print(f"Generating predictions for {len(fleet_df)} vehicles...")

    # Feature columns for model
    feature_cols = [
        'current_mileage', 'age_years',
        'days_since_oil_change', 'days_since_brake_service', 'days_since_transmission_service',
        'miles_since_oil_change', 'miles_since_brake_service',
        'maintenance_compliance_score',
        'total_failures_history', 'failures_last_year', 'days_since_last_failure',
        'avg_failure_cost_history',
        'avg_oil_pressure_30d', 'avg_coolant_temp_30d', 'avg_load_weight_30d',
        'oil_pressure_variability', 'coolant_temp_variability',
        'heavy_load_percentage', 'avg_daily_miles'
    ]

    X = fleet_df[feature_cols].fillna(0)

    # Generate predictions
    fleet_df['failure_probability'] = model.predict_proba(X)[:, 1]
    fleet_df['predicted_failure'] = model.predict(X)
    fleet_df['risk_tier'] = fleet_df['failure_probability'].apply(calculate_risk_tier)

    # Generate recommendations
    fleet_df['recommendations'] = fleet_df.apply(generate_recommendations, axis=1)

    # Calculate cost impact
    PREVENTIVE_COST = 500
    BREAKDOWN_COST = 5500  # repair + downtime

    fleet_df['estimated_intervention_cost'] = fleet_df.apply(
        lambda row: PREVENTIVE_COST if row['predicted_failure'] == 1 else 0,
        axis=1
    )

    fleet_df['estimated_savings'] = fleet_df.apply(
        lambda row: (BREAKDOWN_COST - PREVENTIVE_COST) if row['predicted_failure'] == 1 else 0,
        axis=1
    )

    # Sort by risk
    fleet_df = fleet_df.sort_values('failure_probability', ascending=False)

    # Export for Tableau
    output_path = 'fleet_risk_predictions.csv'
    fleet_df.to_csv(output_path, index=False)

    print(f"\nFLEET RISK SUMMARY")
    print(f"Total vehicles: {len(fleet_df)}")
    print(f"\nRisk Distribution:")
    print(fleet_df['risk_tier'].value_counts())
    print(f"\nVehicles requiring intervention: {fleet_df['predicted_failure'].sum()}")
    print(f"Total estimated intervention cost: ${fleet_df['estimated_intervention_cost'].sum():,.0f}")
    print(f"Total estimated savings from prevention: ${fleet_df['estimated_savings'].sum():,.0f}")

    print(f"\nTOP 10 HIGHEST RISK VEHICLES")
    print(fleet_df[['vehicle_id', 'make', 'model', 'risk_tier', 'failure_probability', 'recommendations']].head(
        10).to_string(index=False))

    print(f"\nPredictions saved to: {output_path}")

    return fleet_df


if __name__ == "__main__":
    predictions = main()