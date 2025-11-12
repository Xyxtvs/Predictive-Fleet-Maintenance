"""
Fleet Maintenance Data Generator
Creates realistic synthetic data for predictive maintenance modeling
"""

import psycopg2
import random
from datetime import datetime, timedelta
import numpy as np
from typing import List, Tuple

# Database connection
DB_CONFIG = {
    'host': 'ep-weathered-heart-adxqd82a-pooler.c-2.us-east-1.aws.neon.tech',
    'database': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_Rrh0OAMk7VFi',
    'sslmode': 'require'
}

# Fleet composition
TRUCK_MAKES = [
    ('Freightliner', 'Cascadia'),
    ('Kenworth', 'T680'),
    ('Peterbilt', '579'),
    ('Volvo', 'VNL'),
    ('International', 'LT')
]

# Maintenance intervals (miles)
MAINTENANCE_INTERVALS = {
    'oil_change': (15000, 25000),
    'brake_service': (40000, 60000),
    'tire_rotation': (25000, 35000),
    'transmission_service': (100000, 150000),
    'coolant_flush': (150000, 200000),
    'air_filter': (20000, 30000),
    'fuel_filter': (30000, 40000),
    'def_system': (25000, 35000),
    'inspection': (10000, 15000)
}

# Failure probability factors
BASE_FAILURE_RATE = 0.02  # 2% chance per month baseline
AGE_MULTIPLIER = 0.15  # +15% per year of age
MILEAGE_MULTIPLIER = 0.00001  # +1% per 100k miles
MISSED_MAINTENANCE_MULTIPLIER = 2.5  # 2.5x if maintenance overdue

def connect_db():
    """Connect to PostgreSQL database"""
    return psycopg2.connect(**DB_CONFIG)

def generate_vehicles(num_vehicles: int = 100) -> List[dict]:
    """Generate fleet of commercial vehicles"""
    vehicles = []
    start_date = datetime(2019, 1, 1)

    for i in range(1, num_vehicles + 1):
        make, model = random.choice(TRUCK_MAKES)
        year = random.randint(2015, 2023)
        purchase_date = start_date + timedelta(days=random.randint(0, 1825))  # 5 years

        # Initial mileage based on year (used trucks)
        age_at_purchase = purchase_date.year - year
        initial_mileage = age_at_purchase * random.randint(80000, 120000)

        # Current mileage (average 10k miles/month in service)
        months_in_service = (datetime.now() - purchase_date).days / 30
        miles_added = int(months_in_service * random.randint(8000, 12000))
        current_mileage = initial_mileage + miles_added

        # Engine hours (roughly 1 hour per 50 miles for highway)
        engine_hours = int(current_mileage / random.randint(45, 55))

        vehicles.append({
            'vehicle_id': i,
            'make': make,
            'model': model,
            'year': year,
            'initial_mileage': initial_mileage,
            'current_mileage': current_mileage,
            'engine_hours': engine_hours,
            'purchase_date': purchase_date,
            'in_service': True
        })

    return vehicles

def generate_maintenance_events(vehicles: List[dict]) -> Tuple[List[dict], dict]:
    """Generate realistic maintenance history"""
    events = []
    event_id = 1
    vehicle_last_service = {}  # Separate tracking dict

    for vehicle in vehicles:
        current_date = vehicle['purchase_date']
        current_mileage = vehicle['initial_mileage']
        end_mileage = vehicle['current_mileage']

        # Track last service for each type
        last_service = {service: current_mileage for service in MAINTENANCE_INTERVALS.keys()}

        while current_mileage < end_mileage:
            # Advance 1-2 weeks
            days_forward = random.randint(7, 14)
            current_date += timedelta(days=days_forward)
            miles_driven = random.randint(5000, 8000)
            current_mileage += miles_driven

            # Check if any maintenance is due
            for service_type, (min_interval, max_interval) in MAINTENANCE_INTERVALS.items():
                interval = random.randint(min_interval, max_interval)
                miles_since_last = current_mileage - last_service[service_type]

                # 80% compliance: sometimes maintenance gets skipped
                if miles_since_last >= interval and random.random() < 0.8:
                    cost = get_maintenance_cost(service_type)
                    labor_hours = random.uniform(1.0, 4.0)

                    events.append({
                        'event_id': event_id,
                        'vehicle_id': vehicle['vehicle_id'],
                        'event_date': current_date,
                        'event_type': service_type,
                        'mileage_at_event': current_mileage,
                        'cost': cost,
                        'labor_hours': labor_hours,
                        'scheduled': True
                    })

                    last_service[service_type] = current_mileage
                    event_id += 1

        # Store separately - don't contaminate vehicle dict
        vehicle_last_service[vehicle['vehicle_id']] = last_service

    return events, vehicle_last_service

def get_maintenance_cost(service_type: str) -> float:
    """Return realistic maintenance costs"""
    costs = {
        'oil_change': random.uniform(150, 250),
        'brake_service': random.uniform(800, 1200),
        'tire_rotation': random.uniform(100, 150),
        'transmission_service': random.uniform(400, 600),
        'coolant_flush': random.uniform(200, 350),
        'air_filter': random.uniform(80, 120),
        'fuel_filter': random.uniform(90, 140),
        'def_system': random.uniform(250, 400),
        'inspection': random.uniform(150, 250)
    }
    return round(costs.get(service_type, 200), 2)

def calculate_failure_probability(vehicle: dict, current_date: datetime, current_mileage: int, last_service: dict) -> float:
    """Calculate failure probability based on vehicle condition"""
    # Base rate
    prob = BASE_FAILURE_RATE

    # Age factor
    age = current_date.year - vehicle['year']
    prob *= (1 + AGE_MULTIPLIER * age)

    # Mileage factor
    prob *= (1 + MILEAGE_MULTIPLIER * current_mileage)

    # Maintenance compliance factor
    for service_type, last_mileage in last_service.items():
        miles_since = current_mileage - last_mileage
        max_interval = MAINTENANCE_INTERVALS[service_type][1]

        if miles_since > max_interval * 1.2:  # 20% overdue
            prob *= MISSED_MAINTENANCE_MULTIPLIER

    return min(prob, 0.5)  # Cap at 50%

def generate_failure_events(vehicles: List[dict], vehicle_last_service: dict) -> List[dict]:
    """Generate realistic failure events based on vehicle condition"""
    failures = []
    failure_id = 1

    FAILURE_TYPES = {
        'engine': {'severity_dist': [0.3, 0.5, 0.2], 'cost_range': (2000, 8000), 'downtime': (24, 120)},
        'transmission': {'severity_dist': [0.2, 0.5, 0.3], 'cost_range': (3000, 12000), 'downtime': (48, 168)},
        'brakes': {'severity_dist': [0.5, 0.4, 0.1], 'cost_range': (800, 3000), 'downtime': (8, 48)},
        'electrical': {'severity_dist': [0.6, 0.3, 0.1], 'cost_range': (500, 2500), 'downtime': (4, 24)},
        'coolant': {'severity_dist': [0.4, 0.4, 0.2], 'cost_range': (800, 4000), 'downtime': (12, 72)},
        'exhaust': {'severity_dist': [0.5, 0.4, 0.1], 'cost_range': (1000, 3500), 'downtime': (8, 48)},
        'suspension': {'severity_dist': [0.6, 0.3, 0.1], 'cost_range': (600, 2000), 'downtime': (8, 36)},
        'tires': {'severity_dist': [0.7, 0.2, 0.1], 'cost_range': (400, 1500), 'downtime': (2, 12)}
    }

    for vehicle in vehicles:
        current_date = vehicle['purchase_date']
        current_mileage = vehicle['initial_mileage']
        end_date = datetime.now()
        end_mileage = vehicle['current_mileage']

        # Get maintenance history for this vehicle
        last_service = vehicle_last_service.get(vehicle['vehicle_id'], {})

        while current_date < end_date and current_mileage < end_mileage:
            # Advance time
            days_forward = random.randint(20, 40)
            current_date += timedelta(days=days_forward)
            miles_driven = random.randint(8000, 12000)
            current_mileage += miles_driven

            # Check for failure
            failure_prob = calculate_failure_probability(vehicle, current_date, current_mileage, last_service)

            if random.random() < failure_prob:
                failure_type = random.choice(list(FAILURE_TYPES.keys()))
                failure_info = FAILURE_TYPES[failure_type]

                # Determine severity
                severity = np.random.choice(['minor', 'major', 'critical'], p=failure_info['severity_dist'])

                # Costs scale with severity
                severity_multiplier = {'minor': 0.6, 'major': 1.0, 'critical': 1.5}[severity]
                base_cost = random.uniform(*failure_info['cost_range'])
                repair_cost = round(base_cost * severity_multiplier, 2)

                # Downtime and indirect costs
                min_down, max_down = failure_info['downtime']
                downtime_hours = int(random.uniform(min_down, max_down) * severity_multiplier)
                tow_cost = random.uniform(200, 600) if severity in ['major', 'critical'] else 0
                lost_revenue = downtime_hours * random.uniform(80, 150)  # Lost revenue per hour

                failures.append({
                    'failure_id': failure_id,
                    'vehicle_id': vehicle['vehicle_id'],
                    'failure_date': current_date,
                    'failure_type': failure_type,
                    'mileage_at_failure': current_mileage,
                    'severity': severity,
                    'downtime_hours': downtime_hours,
                    'repair_cost': repair_cost,
                    'tow_cost': round(tow_cost, 2),
                    'lost_revenue': round(lost_revenue, 2)
                })

                failure_id += 1

    return failures

def generate_telemetry(vehicles: List[dict]) -> List[dict]:
    """Generate daily telemetry readings"""
    telemetry = []
    reading_id = 1

    for vehicle in vehicles:
        current_date = vehicle['purchase_date']
        end_date = datetime.now()

        # Generate readings every 3-5 days (not every day for realism)
        while current_date < end_date:
            days_forward = random.randint(3, 5)
            current_date += timedelta(days=days_forward)

            # Realistic operating parameters
            oil_pressure = random.randint(45, 65)  # Normal range
            coolant_temp = random.randint(180, 210)  # Normal operating temp
            engine_rpm = random.randint(1200, 1600)  # Highway cruise
            load_weight = random.randint(35000, 75000)  # Loaded vs empty
            avg_speed = random.randint(55, 68)
            miles_driven = random.randint(400, 650)
            idle_hours = random.uniform(0.5, 3.0)

            telemetry.append({
                'reading_id': reading_id,
                'vehicle_id': vehicle['vehicle_id'],
                'reading_date': current_date,
                'oil_pressure_psi': oil_pressure,
                'coolant_temp_f': coolant_temp,
                'engine_rpm_avg': engine_rpm,
                'load_weight_lbs': load_weight,
                'avg_speed_mph': avg_speed,
                'miles_driven': miles_driven,
                'idle_hours': round(idle_hours, 1)
            })

            reading_id += 1

    return telemetry

def insert_data(conn, vehicles, maintenance_events, failure_events, telemetry):
    """Batch insert all generated data"""
    cur = conn.cursor()

    print("Inserting vehicles...")
    vehicle_query = """
        INSERT INTO vehicles (vehicle_id, make, model, year, initial_mileage, 
                             current_mileage, engine_hours, purchase_date, in_service)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cur.executemany(vehicle_query, [tuple(v.values()) for v in vehicles])

    print("Inserting maintenance events...")
    maint_query = """
        INSERT INTO maintenance_events (event_id, vehicle_id, event_date, event_type,
                                       mileage_at_event, cost, labor_hours, scheduled)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    cur.executemany(maint_query, [(e['event_id'], e['vehicle_id'], e['event_date'],
                                    e['event_type'], e['mileage_at_event'], e['cost'],
                                    e['labor_hours'], e['scheduled']) for e in maintenance_events])

    print("Inserting failure events...")
    failure_query = """
        INSERT INTO failure_events (failure_id, vehicle_id, failure_date, failure_type,
                                   mileage_at_failure, severity, downtime_hours,
                                   repair_cost, tow_cost, lost_revenue)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cur.executemany(failure_query, [tuple(f.values()) for f in failure_events])

    print("Inserting telemetry readings...")
    telem_query = """
        INSERT INTO telemetry_readings (reading_id, vehicle_id, reading_date,
                                       oil_pressure_psi, coolant_temp_f, engine_rpm_avg,
                                       load_weight_lbs, avg_speed_mph, miles_driven, idle_hours)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cur.executemany(telem_query, [tuple(t.values()) for t in telemetry])

    conn.commit()
    cur.close()

def main():
    """Generate and insert all synthetic data"""
    print("Generating fleet data...")
    vehicles = generate_vehicles(100)

    print("Generating maintenance history...")
    maintenance_events, vehicle_last_service = generate_maintenance_events(vehicles)

    print("Generating failure events...")
    failure_events = generate_failure_events(vehicles, vehicle_last_service)

    print("Generating telemetry readings...")
    telemetry = generate_telemetry(vehicles)

    print(f"\nGenerated:")
    print(f"  - {len(vehicles)} vehicles")
    print(f"  - {len(maintenance_events)} maintenance events")
    print(f"  - {len(failure_events)} failure events")
    print(f"  - {len(telemetry)} telemetry readings")

    print("\nConnecting to database...")
    conn = connect_db()

    print("Inserting data...")
    insert_data(conn, vehicles, maintenance_events, failure_events, telemetry)

    print("\nData generation complete!")

    # Verify
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM vehicles")
    print(f"Vehicles in DB: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM maintenance_events")
    print(f"Maintenance events in DB: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM failure_events")
    print(f"Failure events in DB: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM telemetry_readings")
    print(f"Telemetry readings in DB: {cur.fetchone()[0]}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()