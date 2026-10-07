"""
feed_db.py: load the rideshare CSV files into PostgreSQL.

Run once (or whenever you want to reset the data):  uv run feed_db.py

What it does, in order:
  1. Connect to Postgres using the settings in .env
  2. Create the `rideshare` schema and its 5 tables (if they don't exist yet)
  3. Empty the tables (TRUNCATE), so every run starts clean
  4. Bulk-load each CSV from data/ with Postgres COPY
  5. Print the row count of each table
  6. COMMIT everything in one transaction

Learning notes:
- Schema: a namespace inside a database (like a folder for tables). Our tables
  live in `rideshare`, so their full names are rideshare.users, rideshare.rides, ...
  Postgres looks in `public` by default, which is why the agent must write
  `rideshare.payments` (or set search_path).
- Transaction: autocommit is OFF, so nothing is saved until conn.commit() at the
  end. If anything fails halfway, no partial data is saved.
- COPY is much faster than many INSERTs: Postgres reads the whole file in one go.
"""

# write our data to the postgres database, so that we can query it later. We will use the feed_db.py script to do this.

import os
import csv
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
load_dotenv()  # read the .env file into os.environ

# Default to Postgres' standard port if .env doesn't set one.
if 'POSTGRES_PORT' not in os.environ:
    os.environ['POSTGRES_PORT'] = '5432'

# ============================================================
# CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": os.environ['POSTGRES_HOST'],
    "port": int(os.environ['POSTGRES_PORT']),
    "database": os.environ['POSTGRES_DB'],
    "user": os.environ['POSTGRES_USER'],
    "password": os.environ['POSTGRES_PASSWORD'],
}

CSV_DIR = "data"        # folder with users.csv, vehicles.csv, rides.csv, payments.csv, ratings.csv
SCHEMA = "rideshare"    # Postgres schema that holds all the tables


# ============================================================
# DATABASE CONNECTION
# ============================================================

conn = psycopg2.connect(**DB_CONFIG)  # ** unpacks the dict into keyword arguments
conn.autocommit = False               # group everything into one transaction (see conn.commit() at the end)

cursor = conn.cursor()

print("Connected to PostgreSQL")


# ============================================================
# CREATE TABLES
# ============================================================

# Data model (-> means "references"):
#
#   users  (riders AND drivers, told apart by user_type)
#     ^-- vehicles.driver_id
#     ^-- rides.rider_id, rides.driver_id
#     ^-- payments.user_id
#     ^-- ratings.rider_id, ratings.driver_id
#   rides
#     ^-- payments.ride_id
#     ^-- ratings.ride_id
#
# SQL notes:
#   - PRIMARY KEY: unique ID for each row
#   - FOREIGN KEY ... REFERENCES: the value must exist in the other table
#     (e.g. a ride's driver_id must be a real user)
#   - UNIQUE / NOT NULL / CHECK: rules Postgres enforces on every insert
#   - DECIMAL(9,6): 9 digits in total, 6 after the decimal point (good for GPS coordinates)
#   - IF NOT EXISTS: safe to re-run the script, it won't fail if the table is already there
#   - Tables are created parents-first (users before rides before payments), because a
#     foreign key can only point to a table that already exists.
#   - INDEX (at the end): like a book's index, it lets Postgres find rows by a column
#     without scanning the whole table. Added on the columns used most in JOINs
#     (the *_id foreign keys) and filters (requested_at, status).

create_tables_sql = """

CREATE SCHEMA IF NOT EXISTS rideshare;

-- =========================================================
-- USERS
-- =========================================================

CREATE TABLE IF NOT EXISTS rideshare.users (
    user_id INTEGER PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    phone VARCHAR(50),
    city VARCHAR(100),
    province VARCHAR(50),
    user_type VARCHAR(20) NOT NULL,
    signup_date DATE,
    is_active BOOLEAN
);


-- =========================================================
-- VEHICLES
-- =========================================================

CREATE TABLE IF NOT EXISTS rideshare.vehicles (
    vehicle_id INTEGER PRIMARY KEY,
    driver_id INTEGER NOT NULL,
    make VARCHAR(50),
    model VARCHAR(50),
    year INTEGER,
    license_plate VARCHAR(20) UNIQUE,
    color VARCHAR(30),
    is_active BOOLEAN,

    CONSTRAINT fk_vehicle_driver
        FOREIGN KEY (driver_id)
        REFERENCES rideshare.users(user_id)
);


-- =========================================================
-- RIDES
-- =========================================================

CREATE TABLE IF NOT EXISTS rideshare.rides (
    ride_id INTEGER PRIMARY KEY,

    rider_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,

    requested_at TIMESTAMP,
    pickup_time TIMESTAMP,
    dropoff_time TIMESTAMP,

    pickup_latitude DECIMAL(9,6),
    pickup_longitude DECIMAL(9,6),

    dropoff_latitude DECIMAL(9,6),
    dropoff_longitude DECIMAL(9,6),

    distance_km DECIMAL(10,2),
    fare DECIMAL(10,2),
    surge_multiplier DECIMAL(4,2),

    status VARCHAR(30),
    cancellation_reason VARCHAR(100),

    CONSTRAINT fk_ride_rider
        FOREIGN KEY (rider_id)
        REFERENCES rideshare.users(user_id),

    CONSTRAINT fk_ride_driver
        FOREIGN KEY (driver_id)
        REFERENCES rideshare.users(user_id)
);


-- =========================================================
-- PAYMENTS
-- =========================================================

CREATE TABLE IF NOT EXISTS rideshare.payments (
    payment_id INTEGER PRIMARY KEY,

    ride_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,

    amount DECIMAL(10,2),

    payment_method VARCHAR(50),
    payment_status VARCHAR(30),

    transaction_id VARCHAR(100) UNIQUE,
    payment_time TIMESTAMP,

    CONSTRAINT fk_payment_ride
        FOREIGN KEY (ride_id)
        REFERENCES rideshare.rides(ride_id),

    CONSTRAINT fk_payment_user
        FOREIGN KEY (user_id)
        REFERENCES rideshare.users(user_id)
);


-- =========================================================
-- RATINGS
-- =========================================================

CREATE TABLE IF NOT EXISTS rideshare.ratings (
    rating_id INTEGER PRIMARY KEY,

    ride_id INTEGER NOT NULL,
    rider_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,

    rating INTEGER,
    comment TEXT,
    rated_at TIMESTAMP,

    CONSTRAINT fk_rating_ride
        FOREIGN KEY (ride_id)
        REFERENCES rideshare.rides(ride_id),

    CONSTRAINT fk_rating_rider
        FOREIGN KEY (rider_id)
        REFERENCES rideshare.users(user_id),

    CONSTRAINT fk_rating_driver
        FOREIGN KEY (driver_id)
        REFERENCES rideshare.users(user_id),

    CONSTRAINT chk_rating
        CHECK (rating BETWEEN 1 AND 5)
);


-- =========================================================
-- INDEXES
-- =========================================================

CREATE INDEX IF NOT EXISTS idx_vehicles_driver_id
ON rideshare.vehicles(driver_id);

CREATE INDEX IF NOT EXISTS idx_rides_rider_id
ON rideshare.rides(rider_id);

CREATE INDEX IF NOT EXISTS idx_rides_driver_id
ON rideshare.rides(driver_id);

CREATE INDEX IF NOT EXISTS idx_rides_requested_at
ON rideshare.rides(requested_at);

CREATE INDEX IF NOT EXISTS idx_rides_status
ON rideshare.rides(status);

CREATE INDEX IF NOT EXISTS idx_payments_ride_id
ON rideshare.payments(ride_id);

CREATE INDEX IF NOT EXISTS idx_payments_user_id
ON rideshare.payments(user_id);

CREATE INDEX IF NOT EXISTS idx_ratings_ride_id
ON rideshare.ratings(ride_id);

CREATE INDEX IF NOT EXISTS idx_ratings_driver_id
ON rideshare.ratings(driver_id);

"""

cursor.execute(create_tables_sql)

print("Tables created successfully")


# ============================================================
# CLEAR EXISTING DATA
# ============================================================

# This runs on every execution, so each run fully reloads the CSV data.
# Without it, a second run would fail on duplicate primary keys.
# Comment it out if you want to keep existing rows.
# CASCADE also empties any table that references these ones through a foreign key.

cursor.execute("""
    TRUNCATE TABLE
        rideshare.ratings,
        rideshare.payments,
        rideshare.rides,
        rideshare.vehicles,
        rideshare.users
    CASCADE;
""")



# ============================================================
# LOAD CSV USING POSTGRES COPY
# ============================================================

def load_csv(table_name, csv_file, columns):
    """
    Bulk-load one CSV file into rideshare.<table_name> using COPY.

    Args:
        table_name: target table, e.g. "users"
        csv_file:   file name inside CSV_DIR, e.g. "users.csv"
        columns:    column names in the same order as the CSV header

    Notes:
        - sql.SQL(...).format(sql.Identifier(...)) builds the query safely: it
          quotes table/column names properly, so they can't break or inject SQL
          (unlike plain f-strings).
        - HEADER TRUE skips the first line; NULL '' turns empty cells into NULL.
        - copy_expert streams the open file to Postgres through COPY ... FROM STDIN.
    """

    file_path = os.path.join(CSV_DIR, csv_file)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"CSV file not found: {file_path}"
        )

    copy_sql = sql.SQL("""
        COPY {} ({})
        FROM STDIN
        WITH (
            FORMAT CSV,
            HEADER TRUE,
            DELIMITER ',',
            NULL ''
        )
    """).format(
        sql.Identifier(SCHEMA, table_name),
        sql.SQL(", ").join(
            sql.Identifier(column)
            for column in columns
        )
    )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        cursor.copy_expert(
            copy_sql,
            file
        )

    print(f"Loaded {csv_file}")


# ============================================================
# LOAD USERS
# ============================================================

# Load order matters: parents first (users), then the tables that reference them,
# otherwise the foreign keys would reject the rows.

load_csv(
    "users",
    "users.csv",
    [
        "user_id",
        "first_name",
        "last_name",
        "email",
        "phone",
        "city",
        "province",
        "user_type",
        "signup_date",
        "is_active",
    ],
)


# ============================================================
# LOAD VEHICLES
# ============================================================

load_csv(
    "vehicles",
    "vehicles.csv",
    [
        "vehicle_id",
        "driver_id",
        "make",
        "model",
        "year",
        "license_plate",
        "color",
        "is_active",
    ],
)


# ============================================================
# LOAD RIDES
# ============================================================

load_csv(
    "rides",
    "rides.csv",
    [
        "ride_id",
        "rider_id",
        "driver_id",
        "requested_at",
        "pickup_time",
        "dropoff_time",
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
        "distance_km",
        "fare",
        "surge_multiplier",
        "status",
        "cancellation_reason",
    ],
)


# ============================================================
# LOAD PAYMENTS
# ============================================================

load_csv(
    "payments",
    "payments.csv",
    [
        "payment_id",
        "ride_id",
        "user_id",
        "amount",
        "payment_method",
        "payment_status",
        "transaction_id",
        "payment_time",
    ],
)


# ============================================================
# LOAD RATINGS
# ============================================================

load_csv(
    "ratings",
    "ratings.csv",
    [
        "rating_id",
        "ride_id",
        "rider_id",
        "driver_id",
        "rating",
        "comment",
        "rated_at",
    ],
)


# ============================================================
# VERIFY RECORD COUNTS
# ============================================================

tables = [
    "users",
    "vehicles",
    "rides",
    "payments",
    "ratings",
]

print("\nRecord counts:")
print("-" * 40)

for table in tables:

    cursor.execute(
        sql.SQL(
            "SELECT COUNT(*) FROM {}.{}"
        ).format(
            sql.Identifier(SCHEMA),
            sql.Identifier(table)
        )
    )

    count = cursor.fetchone()[0]  # fetchone() returns one row as a tuple: (count,)

    # :<15 left-aligns in 15 chars, :>10, right-aligns in 10 chars with thousands separators
    print(f"{table:<15} {count:>10,}")


# ============================================================
# COMMIT
# ============================================================

# Save everything at once. Until this line, other connections can't see the new data.

conn.commit()

print("\nData loaded successfully!")
print("Transaction committed.")


# ============================================================
# CLOSE CONNECTION
# ============================================================

cursor.close()
conn.close()

print("PostgreSQL connection closed.")