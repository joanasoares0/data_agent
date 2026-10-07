"""
Database helper for the data agent.

Wraps a PostgreSQL connection and does two jobs for the agent:

1. schema_details(schema_name)
   Turns the structure of a schema (its tables, columns, column types and a
   few sample rows) into plain text. That text goes into the LLM prompt, so
   the model knows which tables and columns exist before it writes SQL.

2. execute_query(query)
   Runs the SQL the LLM wrote and returns the rows as a string.

Example output of schema_details("rideshare"):

    Database Schema: rideshare

    Table: users
      Column: user_id, Data Type: integer
      Column: first_name, Data Type: character varying
      ...
      Sample Data:
        (1, 'Ana', 'Silva', ...)

Running this file directly (`uv run utils/database.py`) writes that text to
schema_details.txt, which is handy for checking what the LLM will see.

Learning notes:
- psycopg2 is the standard Python driver for PostgreSQL.
- A *connection* is the session with the database; a *cursor* is the object
  that sends SQL through that connection and holds the results.
- Each method closes the connection at the end, so a DatabaseUtil object can
  only be used for ONE call. The agent creates a new one each time it needs it.
- The queries are built with f-strings. That's OK here because the schema name
  comes from our own code, but never put untrusted user text into SQL this way
  (SQL injection). psycopg2's sql.Identifier / query parameters are the safe way;
  feed_db.py shows an example.
"""

import psycopg2
import os
import sys
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class DatabaseUtil:

    def __init__(self, db_config):
        # db_config is a dict with the connection settings, e.g.
        # {"host": ..., "port": ..., "database": ..., "user": ..., "password": ...}
        # (same shape as DB_CONFIG in feed_db.py).
        # It can also carry extra libpq settings, e.g.
        # "options": "-c search_path=rideshare,public" (used by execute_sql in the agent).
        self.db_url = db_config

        # Open one connection when the object is created.
        # If it fails (wrong password, database not running, ...) we print the error
        # and keep connection = None instead of crashing at import time.
        try:
            self.connection = psycopg2.connect(**db_config)  # ** unpacks the dict into keyword arguments
        except Exception as e:
            print(f"Error connecting to database: {e}")
            self.connection = None

    def schema_details(self, schema_name):
        """
        Return a text description of every table in `schema_name`:
        its columns (in table order), their data types, and 5 sample rows.
        """

        schema_info_context = ""

        # A cursor is the object that sends SQL to the database and holds the results.
        connection = self.connection
        cursor = connection.cursor()

        # Header line of the description.
        schema_info_context = f"Database Schema: {schema_name}\n"

        try:
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

                # Build the text with `+=` only. (A previous bug did
                # `text += f"{text}..."`, which doubled the whole string on every
                # line and produced a multi-GB file.)
                schema_info_context += f"\nTable: {table_name}\n"

                # 2) For each table, list its columns and their data types.
                #    - filter by table_schema too: the same table name can exist in
                #      other schemas, and we only want this one
                #    - ORDER BY ordinal_position keeps the columns in table order, so
                #      they line up with the values in the sample rows below
                cursor.execute(f"""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = '{schema_name}'
                      AND table_name = '{table_name}'
                    ORDER BY ordinal_position;
                """)
                # Rows look like: [("user_id", "integer"), ("email", "character varying"), ...]
                columns_list = cursor.fetchall()

                for column in columns_list:
                    column_name = column[0]
                    data_type = column[1]
                    schema_info_context += f"  Column: {column_name}, Data Type: {data_type}\n"

                # 3) Add 5 sample rows per table.
                # Real values help the LLM a lot: it learns the exact spelling of
                # categories (e.g. 'credit_card', 'completed'), date formats, etc.
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
            # Don't crash the agent: put the error in the text so it's visible.
            print(f"Error retrieving schema details: {e}")
            schema_info_context += f"Error retrieving schema details: {e}\n"

        finally:
            # `finally` always runs, whether the try block succeeded or failed,
            # so the cursor and connection are always released.
            if cursor:
                cursor.close()
            if connection:
                connection.close()

            # The full description, ready to put in the LLM prompt.
            return schema_info_context

    def execute_query(self, query):
        """
        Execute a SQL query and return the rows as a string,
        e.g. "[('apple_pay',), ('credit_card',)]".

        Returns a string (not a list) because AgentSchema.sql_query_execution_result
        is typed as str, and the result is pasted into an LLM prompt anyway.
        """
        try:
            connection = self.connection
            cursor = connection.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            return str(results)
        except Exception as e:
            # Return the error as text (not None) so the agent state stays a str
            # and the final-answer step can explain the failure to the user.
            print(f"Error executing query: {e}")
            return f"Error executing query: {e}"
        finally:
            # Note: if self.connection is None (connection failed in __init__),
            # `cursor` is never assigned and this block raises NameError,
            # hiding the original error.
            if cursor:
                cursor.close()
            if connection:
                connection.close()

# Run directly to dump the schema description to schema_details.txt.
if __name__ == "__main__":
    obj = DatabaseUtil({
        "host": os.environ['POSTGRES_HOST'],
        "port": int(os.environ['POSTGRES_PORT']),
        "database": os.environ['POSTGRES_DB'],
        "user": os.environ['POSTGRES_USER'],
        "password": os.environ['POSTGRES_PASSWORD'],
    })

    result = obj.schema_details("rideshare")

    with open("schema_details.txt", "w") as f:
        f.write(result)
