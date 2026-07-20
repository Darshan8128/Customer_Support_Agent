"""
database.py — Unified SQLite database module for Date AI Chatbot

Provides persistent storage for:
  - Orders (migrated from hardcoded MOCK_ORDERS dict)
  - Chat sessions & messages (persist across browser restarts)

Uses a single `chatbot.db` file. Existing support_tickets.db and
agent_logs.db remain untouched.
"""

import os
import json
import uuid
import sqlite3
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "chatbot.db")


# ============================================================
# Connection helper
# ============================================================

def _get_conn() -> sqlite3.Connection:
    """Returns a connection with row_factory set for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ============================================================
# Schema initialization
# ============================================================

def init_db():
    """Creates all tables and seeds mock orders if the orders table is empty."""
    conn = _get_conn()
    cursor = conn.cursor()

    # --- Orders table ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id       TEXT PRIMARY KEY,
            customer_name  TEXT NOT NULL,
            product        TEXT NOT NULL,
            status         TEXT NOT NULL,
            order_date     TEXT NOT NULL,
            shipped_date   TEXT,
            delivered_date TEXT,
            tracking_number TEXT,
            carrier        TEXT,
            extra_json     TEXT,
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL
        )
    """)

    # --- Chat sessions table ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id  TEXT PRIMARY KEY,
            title       TEXT DEFAULT 'New Chat',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
    """)

    # --- Chat messages table ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            agent_reasoning TEXT,
            audio_path      TEXT,
            timestamp       TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                ON DELETE CASCADE
        )
    """)

    conn.commit()

    # Seed orders if table is empty
    cursor.execute("SELECT COUNT(*) as cnt FROM orders")
    if cursor.fetchone()["cnt"] == 0:
        _seed_orders(conn)

    conn.close()


# ============================================================
# Order seeding (one-time migration from hardcoded dict)
# ============================================================

_SEED_ORDERS = [
    {
        "order_id": "ORD-1001",
        "customer_name": "Alice Johnson",
        "product": "Date AI Pro License (Annual)",
        "status": "delivered",
        "order_date": "2026-06-15",
        "shipped_date": "2026-06-17",
        "delivered_date": "2026-06-20",
        "tracking_number": "TRK-98765432",
        "carrier": "FedEx",
        "extra_json": json.dumps({}),
    },
    {
        "order_id": "ORD-1002",
        "customer_name": "Bob Smith",
        "product": "Date AI Enterprise Suite",
        "status": "shipped",
        "order_date": "2026-07-10",
        "shipped_date": "2026-07-12",
        "delivered_date": None,
        "tracking_number": "TRK-11223344",
        "carrier": "UPS",
        "extra_json": json.dumps({"estimated_delivery": "2026-07-22"}),
    },
    {
        "order_id": "ORD-1003",
        "customer_name": "Carol Davis",
        "product": "Date AI Starter Plan (Monthly)",
        "status": "processing",
        "order_date": "2026-07-18",
        "shipped_date": None,
        "delivered_date": None,
        "tracking_number": None,
        "carrier": None,
        "extra_json": json.dumps({"estimated_ship_date": "2026-07-23"}),
    },
    {
        "order_id": "ORD-1004",
        "customer_name": "David Lee",
        "product": "Date AI API Credits (10,000)",
        "status": "cancelled",
        "order_date": "2026-07-05",
        "shipped_date": None,
        "delivered_date": None,
        "tracking_number": None,
        "carrier": None,
        "extra_json": json.dumps({
            "cancelled_date": "2026-07-06",
            "cancellation_reason": "Customer requested cancellation",
            "refund_status": "refunded",
        }),
    },
    {
        "order_id": "ORD-1005",
        "customer_name": "Eva Martinez",
        "product": "Date AI Pro License + Training Package",
        "status": "delivered",
        "order_date": "2026-06-01",
        "shipped_date": "2026-06-03",
        "delivered_date": "2026-06-07",
        "tracking_number": "TRK-55667788",
        "carrier": "DHL",
        "extra_json": json.dumps({}),
    },
    {
        "order_id": "ORD-1006",
        "customer_name": "Frank Wilson",
        "product": "Date AI Custom Integration",
        "status": "on_hold",
        "order_date": "2026-07-14",
        "shipped_date": None,
        "delivered_date": None,
        "tracking_number": None,
        "carrier": None,
        "extra_json": json.dumps({
            "hold_reason": "Awaiting technical requirements from customer",
            "support_contact": "integration-team@dateai.com",
        }),
    },
    {
        "order_id": "ORD-1007",
        "customer_name": "Grace Chen",
        "product": "Date AI Enterprise Suite (2-Year)",
        "status": "shipped",
        "order_date": "2026-07-16",
        "shipped_date": "2026-07-18",
        "delivered_date": None,
        "tracking_number": "TRK-99887766",
        "carrier": "FedEx",
        "extra_json": json.dumps({"estimated_delivery": "2026-07-24"}),
    },
    {
        "order_id": "ORD-1008",
        "customer_name": "Henry Brown",
        "product": "Date AI Starter Plan (Annual)",
        "status": "refund_pending",
        "order_date": "2026-06-20",
        "shipped_date": None,
        "delivered_date": "2026-06-25",
        "tracking_number": None,
        "carrier": None,
        "extra_json": json.dumps({
            "refund_requested_date": "2026-07-15",
            "refund_reason": "Product did not meet expectations",
            "refund_status": "under review — estimated 5-7 business days",
        }),
    },
]


def _seed_orders(conn: sqlite3.Connection):
    """Inserts the 8 original mock orders into the database."""
    now = datetime.now().isoformat()
    cursor = conn.cursor()
    for order in _SEED_ORDERS:
        cursor.execute(
            """INSERT INTO orders
               (order_id, customer_name, product, status, order_date,
                shipped_date, delivered_date, tracking_number, carrier,
                extra_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                order["order_id"],
                order["customer_name"],
                order["product"],
                order["status"],
                order["order_date"],
                order.get("shipped_date"),
                order.get("delivered_date"),
                order.get("tracking_number"),
                order.get("carrier"),
                order.get("extra_json", "{}"),
                now,
                now,
            ),
        )
    conn.commit()


# ============================================================
# Order CRUD
# ============================================================

def get_order(order_id: str) -> Optional[dict]:
    """
    Retrieves an order by ID. Returns a flat dict merging
    column data with extra_json fields, or None if not found.
    """
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    # Build the order dict
    order = {
        "order_id": row["order_id"],
        "customer_name": row["customer_name"],
        "product": row["product"],
        "status": row["status"],
        "order_date": row["order_date"],
        "shipped_date": row["shipped_date"],
        "delivered_date": row["delivered_date"],
        "tracking_number": row["tracking_number"],
        "carrier": row["carrier"],
    }

    # Merge any extra fields from extra_json
    try:
        extra = json.loads(row["extra_json"] or "{}")
        order.update(extra)
    except (json.JSONDecodeError, TypeError):
        pass

    return order


def list_orders() -> list[dict]:
    """Returns all orders as a list of dicts."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders ORDER BY order_date DESC")
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        order = {
            "order_id": row["order_id"],
            "customer_name": row["customer_name"],
            "product": row["product"],
            "status": row["status"],
            "order_date": row["order_date"],
        }
        results.append(order)
    return results


def upsert_order(
    order_id: str,
    customer_name: str,
    product: str,
    status: str,
    order_date: str,
    shipped_date: str = None,
    delivered_date: str = None,
    tracking_number: str = None,
    carrier: str = None,
    extra: dict = None,
) -> dict:
    """Inserts or updates an order. Returns the order dict."""
    now = datetime.now().isoformat()
    extra_json = json.dumps(extra or {})

    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO orders
           (order_id, customer_name, product, status, order_date,
            shipped_date, delivered_date, tracking_number, carrier,
            extra_json, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(order_id) DO UPDATE SET
             customer_name = excluded.customer_name,
             product = excluded.product,
             status = excluded.status,
             order_date = excluded.order_date,
             shipped_date = excluded.shipped_date,
             delivered_date = excluded.delivered_date,
             tracking_number = excluded.tracking_number,
             carrier = excluded.carrier,
             extra_json = excluded.extra_json,
             updated_at = excluded.updated_at
        """,
        (
            order_id, customer_name, product, status, order_date,
            shipped_date, delivered_date, tracking_number, carrier,
            extra_json, now, now,
        ),
    )
    conn.commit()
    conn.close()

    return get_order(order_id)


# ============================================================
# Chat session management
# ============================================================

def create_session(title: str = "New Chat") -> str:
    """Creates a new chat session. Returns the session_id."""
    session_id = uuid.uuid4().hex[:12]
    now = datetime.now().isoformat()

    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO chat_sessions (session_id, title, created_at, updated_at)
           VALUES (?, ?, ?, ?)""",
        (session_id, title, now, now),
    )
    conn.commit()
    conn.close()
    return session_id


def list_sessions(limit: int = 20) -> list[dict]:
    """Returns recent chat sessions ordered by last activity."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT session_id, title, created_at, updated_at
           FROM chat_sessions
           ORDER BY updated_at DESC
           LIMIT ?""",
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_session_messages(session_id: str) -> list[dict]:
    """Returns all messages for a session in chronological order."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT role, content, agent_reasoning, audio_path, timestamp
           FROM chat_messages
           WHERE session_id = ?
           ORDER BY id ASC""",
        (session_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    messages = []
    for row in rows:
        msg = {
            "role": row["role"],
            "content": row["content"],
        }
        if row["agent_reasoning"]:
            try:
                msg["agent_reasoning"] = json.loads(row["agent_reasoning"])
            except (json.JSONDecodeError, TypeError):
                pass
        if row["audio_path"]:
            msg["audio"] = row["audio_path"]
        messages.append(msg)

    return messages


def save_message(
    session_id: str,
    role: str,
    content: str,
    agent_reasoning: dict = None,
    audio_path: str = None,
):
    """Saves a single message and updates the session's updated_at and title."""
    now = datetime.now().isoformat()
    reasoning_json = json.dumps(agent_reasoning, default=str) if agent_reasoning else None

    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute(
        """INSERT INTO chat_messages
           (session_id, role, content, agent_reasoning, audio_path, timestamp)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (session_id, role, content, reasoning_json, audio_path, now),
    )

    # Update session timestamp
    cursor.execute(
        "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
        (now, session_id),
    )

    # Auto-title from the first user message
    if role == "user":
        cursor.execute(
            "SELECT title FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if row and row["title"] == "New Chat":
            # Use first 50 chars of user message as title
            title = content[:50].strip()
            if len(content) > 50:
                title += "…"
            cursor.execute(
                "UPDATE chat_sessions SET title = ? WHERE session_id = ?",
                (title, session_id),
            )

    conn.commit()
    conn.close()


def delete_session(session_id: str):
    """Deletes a chat session and all its messages."""
    conn = _get_conn()
    cursor = conn.cursor()
    # Messages deleted by CASCADE
    cursor.execute(
        "DELETE FROM chat_sessions WHERE session_id = ?",
        (session_id,),
    )
    conn.commit()
    conn.close()
