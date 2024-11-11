# db_connector.py

import sqlite3
from sqlite3 import Error

# Create a connection to the database
def create_connection(db_file):
    """
    Creates a connection to the SQLite database.

    Args:
        db_file (str): The path to the database file.

    Returns:
        sqlite3.Connection: The connection object to the database.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    return conn

# Create the requirements table
def create_requirements_table(conn):
    """
    Creates the requirements table in the database.

    Args:
        conn (sqlite3.Connection): The connection object to the database.
    """
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS requirements (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                label TEXT NOT NULL,
                classification TEXT NOT NULL
            )
        """)
        conn.commit()
    except Error as e:
        print(e)

# Insert a requirement into the database
def insert_requirement(conn, requirement):
    """
    Inserts a requirement into the database.

    Args:
        conn (sqlite3.Connection): The connection object to the database.
        requirement (tuple): A tuple containing the requirement text, label, and classification.
    """
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO requirements (text, label, classification)
            VALUES (?, ?, ?)
        """, requirement)
        conn.commit()
    except Error as e:
        print(e)

# Retrieve all requirements from the database
def retrieve_requirements(conn):
    """
    Retrieves all requirements from the database.

    Args:
        conn (sqlite3.Connection): The connection object to the database.

    Returns:
        list: A list of tuples containing the requirement text, label, and classification.
    """
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM requirements")
        rows = cur.fetchall()
        return rows
    except Error as e:
        print(e)

# Close the database connection
def close_connection(conn):
    """
    Closes the database connection.

    Args:
        conn (sqlite3.Connection): The connection object to the database.
    """
    if conn:
        conn.close()


import sqlite3

# Connect to the database
def connect_to_db(db_name):
    """
    Connects to the SQLite database.

    Args:
        db_name (str): The name of the database.

    Returns:
        sqlite3.Connection: The connection object to the database.
    """
    try:
        conn = sqlite3.connect(db_name)
        return conn
    except sqlite3.Error as e:
        print(e)

# Create the requirements table
def create_requirements_table(conn):
    """
    Creates the requirements table in the database.

    Args:
        conn (sqlite3.Connection): The connection object to the database.
    """
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS requirements (
                id INTEGER PRIMARY KEY,
                requirement_text TEXT NOT NULL,
                classification TEXT NOT NULL,
                similarity REAL
            )
        """)
        conn.commit()
    except sqlite3.Error as e:
        print(e)

# Insert a requirement into the database
def insert_requirement(conn, requirement):
    """
    Inserts a requirement into the database.

    Args:
        conn (sqlite3.Connection): The connection object to the database.
        requirement (tuple): A tuple containing the requirement text, classification, and similarity score.
    """
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO requirements (requirement_text, classification, similarity)
            VALUES (?, ?, ?)
        """, requirement)
        conn.commit()
    except sqlite3.Error as e:
        print(e)
# db_connector.py (updated)

# ...

def retrieve_ambiguous_requirements(conn):
    """
    Retrieves ambiguous requirements from the database.

    Args:
        conn (sqlite3.Connection): The connection object to the database.

    Returns:
        list: A list of ambiguous requirements.
    """
    try:
        cur = conn.cursor()
        cur.execute("SELECT requirement_text FROM requirements WHERE similarity > 0.5")  # Assume similarity > 0.5 indicates ambiguity
        ambiguous_requirements = [row[0] for row in cur.fetchall()]
        return ambiguous_requirements
    except sqlite3.Error as e:
        print(e)

# Example usage
if __name__ == "__main__":
    db_name = "requirements.db"
    conn = connect_to_db(db_name)
    create_requirements_table(conn)
    # Insert a requirement
    requirement = ("This is a sample requirement", "Functional", 0.8)
    insert_requirement(conn, requirement)
    conn.close()
# Example usage
if __name__ == "__main__":
    # Create a connection to the database
    db_file = "requirements.db"
    conn = create_connection(db_file)

    # Create the requirements table
    create_requirements_table(conn)

    # Insert a requirement into the database
    requirement = ("This is a sample functional requirement.", "Functional", "Classified")
    insert_requirement(conn, requirement)

    # Retrieve all requirements from the database
    requirements = retrieve_requirements(conn)
    for row in requirements:
        print(row)

    # Close the database connection
    close_connection(conn)

