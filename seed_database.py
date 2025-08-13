import os
import sys
import re
import requests
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def clean_sql_script(sql_script):
    """
    Clean SQL script to be executable via psycopg2.
    
    Args:
        sql_script (str): Raw SQL script
        
    Returns:
        str: Cleaned SQL script
    """
    # Remove problematic commands
    lines = []
    for line in sql_script.split('\n'):
        # Skip psql meta-commands
        if line.strip().startswith('\\'):
            continue
        # Skip DROP DATABASE commands
        if 'DROP DATABASE' in line.upper():
            continue
        # Skip CREATE DATABASE commands
        if 'CREATE DATABASE' in line.upper():
            continue
        # Skip USE DATABASE commands
        if line.strip().upper().startswith('USE '):
            continue
        # Remove N' prefix from strings (SQL Server syntax)
        line = re.sub(r"\bN'", "'", line)
        lines.append(line)
    
    return '\n'.join(lines)


def seed_database():
    """
    Seeds the PostgreSQL database with Chinook data.
    """
    print("Starting Database Seeding Process")
    print("-" * 50)
    
    sql_file_url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_PostgreSql_SerialPKs.sql"
    
    print(f"Downloading SQL script...")
    try:
        response = requests.get(sql_file_url, timeout=30)
        response.raise_for_status()
        postgresql_sql = response.text
        print("SUCCESS: SQL script downloaded")
    except requests.RequestException as e:
        print(f"ERROR: Failed to download SQL script: {e}")
        sys.exit(1)
    
    # Clean the SQL script
    print("Cleaning SQL script...")
    postgresql_sql = clean_sql_script(postgresql_sql)
    
    # Connect to database
    print("Connecting to PostgreSQL database...")
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'user': os.getenv('DB_USER', 'chinook_user'),
        'password': os.getenv('DB_PASSWORD', 'chinook_password'),
        'database': os.getenv('DB_NAME', 'chinook_db')
    }
    
    try:
        conn = psycopg2.connect(**db_config)
        conn.autocommit = False  # Use transactions
        cursor = conn.cursor()
        print(f"SUCCESS: Connected to database '{db_config['database']}'")
    except psycopg2.Error as e:
        print(f"ERROR: Failed to connect to PostgreSQL: {e}")
        sys.exit(1)
    
    try:
        # First, drop existing tables
        print("Cleaning existing tables...")
        cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        tables = cursor.fetchall()
        
        for table in tables:
            cursor.execute(f'DROP TABLE IF EXISTS "{table[0]}" CASCADE')
        
        if tables:
            print(f"Dropped {len(tables)} existing tables")

        # Execute SQL script
        print("Executing SQL script...")
        cursor.execute(postgresql_sql)
        conn.commit()
        print("SUCCESS: SQL script executed")

    except psycopg2.Error as e:
        conn.rollback()
        print(f"ERROR: Database seeding failed: {e}")
        sys.exit(1)
    
    finally:
        cursor.close()
        conn.close()
        print("\nDatabase connection closed")


if __name__ == "__main__":
    seed_database()
