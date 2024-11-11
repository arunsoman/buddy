# db_connector.py

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

# Example usage
if __name__ == "__main__":
    db_name = "requirements.db"
    conn = connect_to_db(db_name)
    create_requirements_table(conn)
    # Insert a requirement
    requirement = ("This is a sample requirement", "Functional", 0.8)
    insert_requirement(conn, requirement)
    conn.close()
