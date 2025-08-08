"""
Database seeding script for PostgreSQL.
Downloads and executes the Chinook PostgreSQL script.
"""

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
    
    # Use the SerialPKs version which should be cleaner
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
        conn.autocommit = True  # Important: autocommit for DDL
        cursor = conn.cursor()
        print(f"SUCCESS: Connected to database '{db_config['database']}'")
    except psycopg2.Error as e:
        print(f"ERROR: Failed to connect to PostgreSQL: {e}")
        sys.exit(1)
    
    # First, drop existing tables
    print("Cleaning existing tables...")
    try:
        cursor.execute("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public'
        """)
        tables = cursor.fetchall()
        
        for table in tables:
            cursor.execute(f'DROP TABLE IF EXISTS "{table[0]}" CASCADE')
        
        if tables:
            print(f"Dropped {len(tables)} existing tables")
    except psycopg2.Error as e:
        print(f"Warning: Could not drop tables: {e}")
    
    # Execute SQL script
    print("Executing SQL script...")
    
    # Split by semicolon but be careful with strings containing semicolons
    # Use a more robust splitting method
    statements = []
    current_statement = []
    in_string = False
    
    for line in postgresql_sql.split('\n'):
        # Track if we're inside a string
        quote_count = line.count("'") - line.count("\\'")
        if quote_count % 2 == 1:
            in_string = not in_string
        
        current_statement.append(line)
        
        # If line ends with semicolon and we're not in a string
        if line.rstrip().endswith(';') and not in_string:
            statements.append('\n'.join(current_statement))
            current_statement = []
    
    # Add any remaining statement
    if current_statement:
        statements.append('\n'.join(current_statement))
    
    success_count = 0
    error_count = 0
    
    for i, statement in enumerate(statements):
        statement = statement.strip()
        if not statement or statement.upper().startswith('SET'):
            continue
        
        try:
            cursor.execute(statement)
            success_count += 1
            if success_count % 10 == 0:
                print(f"  Executed {success_count} statements...")
        except psycopg2.Error as e:
            error_count += 1
            # Only show first 100 chars of error
            error_msg = str(e)[:100]
            if "already exists" not in error_msg.lower():
                # Only show first few errors to avoid spam
                if error_count <= 5:
                    print(f"  Error on statement {i}: {error_msg}")
    
    print(f"SUCCESS: Executed {success_count} statements ({error_count} errors)")
    
    # Verify seeding
    print("\nVerifying database seeding...")
    try:
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
        """)
        table_count = cursor.fetchone()[0]
        
        if table_count > 0:
            print(f"SUCCESS: Created {table_count} tables")
            
            # Check key tables
            tables_to_check = ['Artist', 'Album', 'Track', 'Customer', 'Invoice']
            for table in tables_to_check:
                try:
                    cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
                    row_count = cursor.fetchone()[0]
                    print(f"  {table}: {row_count} rows")
                except psycopg2.Error:
                    pass
        else:
            print("ERROR: No tables were created")
            print("Trying alternative approach...")
            
            # Try downloading and executing a different version
            print("\nAttempting with alternative SQL source...")
            alternative_url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_PostgreSql_AutoIncrementPKs.sql"
            
            try:
                response = requests.get(alternative_url, timeout=30)
                response.raise_for_status()
                alt_sql = clean_sql_script(response.text)
                
                # Try executing as one block
                cursor.execute(alt_sql)
                print("SUCCESS: Alternative script executed")
                
            except Exception as e:
                print(f"Alternative also failed: {e}")
                print("\nManual solution: Download and execute the SQL manually:")
                print("1. Download: https://github.com/lerocha/chinook-database/blob/master/ChinookDatabase/DataSources/Chinook_PostgreSql.sql")
                print("2. Execute in PostgreSQL:")
                print("   docker-compose exec db psql -U chinook_user -d chinook_db -f /path/to/Chinook_PostgreSql.sql")
        
    except psycopg2.Error as e:
        print(f"WARNING: Could not verify seeding: {e}")
    
    finally:
        cursor.close()
        conn.close()
        print("\nDatabase connection closed")


if __name__ == "__main__":
    seed_database()