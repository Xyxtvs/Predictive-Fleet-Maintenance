"""
Feature Engineering for Predictive Maintenance
Transforms raw operational data into predictive signals
"""

import psycopg2
import pandas as pd
import numpy as np
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


def connect_db():
    return psycopg2.connect(**DB_CONFIG)


def extract_base_data():
    """Pull all data from database"""
    conn = connect_db()

    print("Extracting vehicles...")
    vehicles = pd.read_sql("SELECT * FROM vehicles", conn)

    print("Extracting maintenance events...")
    maintenance = pd.read_sql("SELECT * FROM maintenance_events", conn)

    print("Extracting failure events...")
    failures = pd.read_sql("SELECT * FROM failure_events", conn)

    print("Extracting telemetry...")
    telemetry = pd.read_sql("SELECT * FROM telemetry_readings", conn)

    conn.close()

    return vehicles, maintenance, failures, telemetry


def calculate_days_since_maintenance(vehicle_id, current_date, maintenance_df, event_type):
    """Calculate days since last maintenance of specific type"""
    vehicle_maint = maintenance_df[
        (maintenance_df['vehicle_id'] == vehicle_id) &
        (maintenance_df['event_type'] == event_type) &
        (maintenance_df['event_date'] <= current_date)
        ]

    if len(vehicle_maint) == 0:
        return 9999  # Never serviced

    last_service = vehicle_maint['event_date'].max()
    return (current_date - last_service).days


def calculate_miles_since_maintenance(vehicle_id, current_mileage, maintenance_df, event_type):
    """Calculate miles since last maintenance of specific type"""
    vehicle_maint = maintenance_df[
        (maintenance_df['vehicle_id'] == vehicle_id) &
        (maintenance_df['event_type'] == event_type) &
        (maintenance_df['mileage_at_event'] <= current_mileage)
        ]

    if len(vehicle_maint) == 0:
        return 999999  # Never serviced

    last_service_mileage = vehicle_maint['mileage_at_event'].max()
    return current_mileage - last_service_mileage


def get_telemetry_features(vehicle_id, reference_date, telemetry_df, days_back=30):
    """Calculate rolling telemetry averages"""
    start_date = reference_date - timedelta(days=days_back)

    vehicle_telem = telemetry_df[
        (telemetry_df['vehicle_id'] == vehicle_id) &
        (telemetry_df['reading_date'] >= start_date) &
        (telemetry_df['reading_date'] <= reference_date)
        ]

    if len(vehicle_telem) == 0:
        return {
            'avg_oil_pressure': 0,
            'avg_coolant_temp': 0,
            'avg_load_weight': 0,
            'oil_pressure_std': 0,
            'coolant_temp_std': 0,
            'heavy_load_pct': 0,
            'avg_daily_miles': 0
        }

    heavy_load_threshold = 60000  # lbs

    return {
        'avg_oil_pressure': vehicle_telem['oil_pressure_psi'].mean(),
        'avg_coolant_temp': vehicle_telem['coolant_temp_f'].mean(),
        'avg_load_weight': vehicle_telem['load_weight_lbs'].mean(),
        'oil_pressure_std': vehicle_telem['oil_pressure_psi'].std(),
        'coolant_temp_std': vehicle_telem['coolant_temp_f'].std(),
        'heavy_load_pct': (vehicle_telem['load_weight_lbs'] > heavy_load_threshold).mean() * 100,
        'avg_daily_miles': vehicle_telem['miles_driven'].mean()
    }


def get_failure_history(vehicle_id, reference_date, failures_df):
    """Calculate historical failure metrics"""
    vehicle_failures = failures_df[
        (failures_df['vehicle_id'] == vehicle_id) &
        (failures_df['failure_date'] < reference_date)
        ]

    return {
        'total_failures': len(vehicle_failures),
        'failures_last_year': len(vehicle_failures[
                                      vehicle_failures['failure_date'] >= reference_date - timedelta(days=365)
                                      ]),
        'avg_failure_cost': vehicle_failures['repair_cost'].mean() if len(vehicle_failures) > 0 else 0,
        'days_since_last_failure': (
            (reference_date - vehicle_failures['failure_date'].max()).days
            if len(vehicle_failures) > 0 else 9999
        )
    }


def create_training_features(vehicles, maintenance, failures, telemetry):
    """
    Create feature matrix for model training
    For each vehicle-date combination, calculate features and label
    """
    print("\nBuilding feature matrix...")

    # Convert ALL date columns to pandas datetime
    vehicles['purchase_date'] = pd.to_datetime(vehicles['purchase_date'])
    maintenance['event_date'] = pd.to_datetime(maintenance['event_date'])
    failures['failure_date'] = pd.to_datetime(failures['failure_date'])
    telemetry['reading_date'] = pd.to_datetime(telemetry['reading_date'])

    # Generate observation dates (monthly snapshots)
    start_date = telemetry['reading_date'].min()
    end_date = pd.Timestamp.now()
    observation_dates = pd.date_range(start=start_date, end=end_date, freq='30D')

    features_list = []

    for vehicle_id in vehicles['vehicle_id'].values:
        vehicle_info = vehicles[vehicles['vehicle_id'] == vehicle_id].iloc[0]

        for obs_date in observation_dates:
            # Skip if before vehicle purchase
            if obs_date < vehicle_info['purchase_date']:
                continue

            # Estimate current mileage at observation date
            days_in_service = (obs_date - vehicle_info['purchase_date']).days
            miles_per_day = (vehicle_info['current_mileage'] - vehicle_info['initial_mileage']) / max(
                (pd.Timestamp.now() - vehicle_info['purchase_date']).days, 1
            )
            current_mileage = vehicle_info['initial_mileage'] + int(days_in_service * miles_per_day)

            # --- FEATURES ---

            # Vehicle characteristics
            age = obs_date.year - vehicle_info['year']

            # Maintenance recency features
            days_since_oil = calculate_days_since_maintenance(vehicle_id, obs_date, maintenance, 'oil_change')
            days_since_brakes = calculate_days_since_maintenance(vehicle_id, obs_date, maintenance, 'brake_service')
            days_since_trans = calculate_days_since_maintenance(vehicle_id, obs_date, maintenance,
                                                                'transmission_service')

            miles_since_oil = calculate_miles_since_maintenance(vehicle_id, current_mileage, maintenance, 'oil_change')
            miles_since_brakes = calculate_miles_since_maintenance(vehicle_id, current_mileage, maintenance,
                                                                   'brake_service')

            # Telemetry features
            telem_features = get_telemetry_features(vehicle_id, obs_date, telemetry, days_back=30)

            # Failure history
            failure_features = get_failure_history(vehicle_id, obs_date, failures)

            # Maintenance compliance score (0-100)
            oil_overdue = max(0, (miles_since_oil - 25000) / 25000)
            brake_overdue = max(0, (miles_since_brakes - 60000) / 60000)
            compliance_score = max(0, 100 - (oil_overdue + brake_overdue) * 50)

            # --- TARGET LABEL ---
            # Did failure occur in next 30 days?
            future_failures = failures[
                (failures['vehicle_id'] == vehicle_id) &
                (failures['failure_date'] > obs_date) &
                (failures['failure_date'] <= obs_date + timedelta(days=30))
                ]
            failure_occurred = 1 if len(future_failures) > 0 else 0

            # Build feature row
            feature_row = {
                'vehicle_id': vehicle_id,
                'observation_date': obs_date,
                'current_mileage': current_mileage,
                'age_years': age,
                'days_since_oil_change': days_since_oil,
                'days_since_brake_service': days_since_brakes,
                'days_since_transmission_service': days_since_trans,
                'miles_since_oil_change': miles_since_oil,
                'miles_since_brake_service': miles_since_brakes,
                'maintenance_compliance_score': compliance_score,
                'total_failures_history': failure_features['total_failures'],
                'failures_last_year': failure_features['failures_last_year'],
                'days_since_last_failure': failure_features['days_since_last_failure'],
                'avg_failure_cost_history': failure_features['avg_failure_cost'],
                'avg_oil_pressure_30d': telem_features['avg_oil_pressure'],
                'avg_coolant_temp_30d': telem_features['avg_coolant_temp'],
                'avg_load_weight_30d': telem_features['avg_load_weight'],
                'oil_pressure_variability': telem_features['oil_pressure_std'],
                'coolant_temp_variability': telem_features['coolant_temp_std'],
                'heavy_load_percentage': telem_features['heavy_load_pct'],
                'avg_daily_miles': telem_features['avg_daily_miles'],
                'failure_next_30_days': failure_occurred  # TARGET
            }

            features_list.append(feature_row)

    df = pd.DataFrame(features_list)

    print(f"\nFeature matrix created: {len(df)} observations")
    print(f"Failure rate in dataset: {df['failure_next_30_days'].mean() * 100:.2f}%")

    return df


def main():
    """Extract data and build feature matrix"""
    print("Starting feature engineering...")

    vehicles, maintenance, failures, telemetry = extract_base_data()

    print(f"\nLoaded:")
    print(f"  - {len(vehicles)} vehicles")
    print(f"  - {len(maintenance)} maintenance events")
    print(f"  - {len(failures)} failure events")
    print(f"  - {len(telemetry)} telemetry readings")

    features_df = create_training_features(vehicles, maintenance, failures, telemetry)

    # Save to CSV for model training
    output_path = 'features_training_data.csv'
    features_df.to_csv(output_path, index=False)
    print(f"\nFeatures saved to: {output_path}")

    # Summary stats
    print("\n=== Feature Summary ===")
    print(f"Total observations: {len(features_df)}")
    print(f"Unique vehicles: {features_df['vehicle_id'].nunique()}")
    print(f"Date range: {features_df['observation_date'].min()} to {features_df['observation_date'].max()}")
    print(f"Failures (next 30d): {features_df['failure_next_30_days'].sum()}")
    print(f"Failure rate: {features_df['failure_next_30_days'].mean() * 100:.2f}%")

    return features_df


if __name__ == "__main__":
    df = main()