import sqlite3
import os
import pandas as pd
from typing import Dict, List, Tuple, Optional


def get_conn(db_path: str):
    d = os.path.dirname(db_path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def create_tables(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trans_date_trans_time TEXT,
            merchant TEXT,
            category TEXT,
            amt REAL,
            first TEXT,
            last TEXT
        )
        """
    )
    conn.commit()


def migrate_csv_to_db(conn: sqlite3.Connection, csv_path: str):
    # If CSV doesn't exist, nothing to do
    if not os.path.exists(csv_path):
        return
    cur = conn.cursor()
    cur.execute('SELECT COUNT(1) as c FROM transactions')
    row = cur.fetchone()
    if row and row['c'] > 0:
        # already has data, skip automatic migration to avoid duplicates
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
    insert_sql = (
        'INSERT INTO transactions (trans_date_trans_time, merchant, category, amt, first, last)'
        ' VALUES (?, ?, ?, ?, ?, ?)'
    )
    for _, r in df.iterrows():
        dt = r.get('trans_date_trans_time', None)
        merchant = r.get('merchant', None)
        category = r.get('category', None)
        try:
            amt = float(r.get('amt', 0)) if pd.notna(r.get('amt', None)) else 0.0
        except Exception:
            amt = 0.0
        first = r.get('first', None)
        last = r.get('last', None)
        cur.execute(insert_sql, (dt, merchant, category, amt, first, last))
    conn.commit()


def init_db(db_path: str, csv_path: Optional[str] = None):
    conn = get_conn(db_path)
    create_tables(conn)
    if csv_path:
        migrate_csv_to_db(conn, csv_path)
    return conn


def insert_transaction(conn: sqlite3.Connection, row: Dict):
    sql = (
        'INSERT INTO transactions (trans_date_trans_time, merchant, category, amt, first, last)'
        ' VALUES (?, ?, ?, ?, ?, ?)'
    )
    dt = row.get('trans_date_trans_time')
    merchant = row.get('merchant')
    category = row.get('category')
    try:
        amt = float(row.get('amt', 0))
    except Exception:
        amt = 0.0
    first = row.get('first')
    last = row.get('last')
    cur = conn.cursor()
    cur.execute(sql, (dt, merchant, category, amt, first, last))
    conn.commit()
    return cur.lastrowid


def fetch_all(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute('SELECT * FROM transactions ORDER BY trans_date_trans_time')
    return cur.fetchall()


def fetch_for_user(conn: sqlite3.Connection, first: str, last: str) -> List[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(
        'SELECT * FROM transactions WHERE first = ? AND last = ? ORDER BY trans_date_trans_time',
        (first, last),
    )
    return cur.fetchall()


def list_users(conn: sqlite3.Connection) -> List[Tuple[str, str]]:
    cur = conn.cursor()
    cur.execute('SELECT DISTINCT first, last FROM transactions WHERE first IS NOT NULL OR last IS NOT NULL')
    return [(r['first'], r['last']) for r in cur.fetchall()]
