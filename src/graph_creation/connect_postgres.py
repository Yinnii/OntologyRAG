import os
import psycopg2 as p
from sqlalchemy import create_engine

POSTGRES_USER = os.getenv("POSTGRES_USER", "openml_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "openml_password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "openml")

# initialize postgresql database connection
def init_postgresql():
    """ Initialize the PostgreSQL database connection."""
    connection = p.connect(
      database=POSTGRES_DB,
      user=POSTGRES_USER,
      password=POSTGRES_PASSWORD,
      host=POSTGRES_HOST,
      port=POSTGRES_PORT
    )

    engine = create_engine(
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    return connection, engine

def close_postgresql(connection):
    """ Close the PostgreSQL database connection."""
    if connection:
        connection.close()
        print("PostgreSQL connection is closed.")
    else:
        print("No PostgreSQL connection to close.")