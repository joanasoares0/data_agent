"""
Database helper for the data agent.

Wraps a PostgreSQL connection and turns the structure of a schema
(its tables, columns and column types) into plain text. That text is
passed to the LLM as context, so it knows which tables and columns exist
before it writes a SQL query.

Example output of schema_details("rideshare"):

    Database Schema: rideshare

    Table: users
      Column: user_id, Data Type: integer
      Column: first_name, Data Type: character varying
      ...
"""

import psycopg2


class DatabaseUtil:

    def __init__(self, db_config):
        # db_config is a dict with the connection settings, e.g.
        # {"host": ..., "port": ..., "database": ..., "user": ..., "password": ...}
        # (same shape as DB_CONFIG in feed_db.py).
        self.db_url = db_config

        # Open one connection when the object is created and reuse it for every query.
        # If it fails (wrong password, database not running, ...) we print the error
        # and keep connection = None instead of crashing at import time.
        try:
            self.connection = psycopg2.connect(**db_config)  # ** unpacks the dict into keyword arguments
        except Exception as e:
            print(f"Error connecting to database: {e}")
            self.connection = None

    def schema_details(self, schema_name):
        """Return a text description of every table and column in `schema_name`."""

        try:
            schema_info_context = ""

            # A cursor is the object that sends SQL to the database and holds the results.
            connection = self.connection
            cursor = connection.cursor()

            # Header line of the description.
            schema_info_context = f"Database Schema: {schema_name}\n"

            # 1) List the tables in the schema.
            # information_schema is a set of built-in views that Postgres provides
            # to describe the database itself (tables, columns, constraints, ...).
            cursor.execute(f"""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = '{schema_name}';
            """)
            # fetchall() returns a list of tuples, one per row: [("users",), ("rides",), ...]
            tables_list = cursor.fetchall()

            for table in tables_list:
                table_name = table[0]  # each row is a 1-item tuple, so take the first element
                schema_info_context += f"\nTable: {table_name}\n"

                # 2) For each table, list its columns and their data types.
                cursor.execute(f"""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = '{table_name}';
                """)
                # Rows look like: [("user_id", "integer"), ("email", "character varying"), ...]
                columns_list = cursor.fetchall()

                for column in columns_list:
                    column_name = column[0]
                    data_type = column[1]
                    schema_info_context += f"  Column: {column_name}, Data Type: {data_type}\n"

                # Adding sample data for each table
                cursor.execute(f"""
                    SELECT *
                    FROM {schema_name}.{table_name}
                    LIMIT 5;"""
                    )
                sample_data = cursor.fetchall()
                schema_info_context += f"  Sample Data:\n"
                for row in sample_data:
                    schema_info_context += f"    {row}\n"

        except Exception as e:
            print(f"Error retrieving schema details: {e}")
            schema_info_context += f"Error retrieving schema details: {e}\n"

        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

            # The full description, ready to put in the LLM prompt.
            return schema_info_context

if __name__ == "__main__":
    obj = DatabaseUtil({
        "host": "localhost",
        "port": 5432,
        "database": "postgres",
        "user": "postgres",     
        "password": "<redacted>"
    })

    result = obj.schema_details("rideshare")

    with open("schema_details.txt", "w") as f:
        f.write(result)