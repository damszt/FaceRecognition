"""
database.py — Supabase PostgreSQL abstraction for Face Recognition app.

Handles attendance logging and querying.
Falls back to local CSV files when Supabase is not configured (for local development).
"""
import os
import json
import csv
from datetime import datetime

# Supabase config
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Detect if Supabase is configured
_supabase_client = None


def _get_client():
    """Lazy-init Supabase client."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    if not SUPABASE_URL or not SUPABASE_KEY:
        return None

    try:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return _supabase_client
    except Exception as e:
        print(f"Supabase init error: {e}")
        return None


def is_configured():
    """Check if Supabase is properly configured."""
    return bool(SUPABASE_URL and SUPABASE_KEY)


# ============================================================
# Local CSV Fallback (for development without Supabase)
# ============================================================

LOCAL_ATTENDANCE_PREFIX = "attendance_"


def _local_log_attendance(name, confidence, details=None):
    """Log attendance to local CSV file."""
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{LOCAL_ATTENDANCE_PREFIX}{today_str}.csv"

        # Check if already logged
        if os.path.exists(filename):
            with open(filename, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0] == name:
                        return False

        timestamp = datetime.now().strftime("%H:%M:%S")
        confidence_str = f"{confidence:.2f}"
        details_json = json.dumps(details) if details else "[]"

        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, timestamp, confidence_str, details_json])

        return True
    except Exception as e:
        print(f"Local logging error: {e}")
        return False


def _local_get_attendance_logs(date_str=None):
    """Get attendance logs from local CSV file."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    filename = f"{LOCAL_ATTENDANCE_PREFIX}{date_str}.csv"
    logs = []

    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        confidence = row[2] if len(row) > 2 else "N/A"
                        details = json.loads(row[3]) if len(row) > 3 else []
                        logs.append({
                            "name": row[0],
                            "timestamp": row[1],
                            "confidence": confidence,
                            "details": details,
                        })
        except Exception as e:
            print(f"Error reading local logs: {e}")

    return logs


# ============================================================
# Supabase Operations
# ============================================================

def log_attendance(name, confidence, details=None):
    """
    Log attendance record.
    Returns True if logged, False if already present or error.
    """
    client = _get_client()
    if client is None:
        return _local_log_attendance(name, confidence, details)

    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H:%M:%S")

        # Check if already logged today
        check = (
            client.table("attendance")
            .select("id")
            .eq("person_name", name)
            .eq("date", today_str)
            .execute()
        )
        if check.data:
            return False  # Already logged today

        # Insert new record
        record = {
            "person_name": name,
            "date": today_str,
            "time": time_str,
            "confidence": round(confidence, 2),
            "details": details or [],
        }
        result = client.table("attendance").insert(record).execute()
        return bool(result.data)
    except Exception as e:
        print(f"Supabase log_attendance error: {e}")
        # Fallback to local
        return _local_log_attendance(name, confidence, details)


def get_attendance_logs(date_str=None):
    """
    Get attendance records for a given date.
    Returns list of dicts with name, timestamp, confidence, details.
    """
    client = _get_client()
    if client is None:
        return _local_get_attendance_logs(date_str)

    try:
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        result = (
            client.table("attendance")
            .select("person_name, time, confidence, details")
            .eq("date", date_str)
            .order("time")
            .execute()
        )

        logs = []
        for row in result.data or []:
            logs.append({
                "name": row["person_name"],
                "timestamp": row["time"],
                "confidence": str(row["confidence"]),
                "details": row.get("details", []),
            })
        return logs
    except Exception as e:
        print(f"Supabase get_logs error: {e}")
        return _local_get_attendance_logs(date_str)


def get_database_info():
    """Get debug info about database status."""
    client = _get_client()
    info = {
        "supabase_configured": is_configured(),
        "supabase_connected": client is not None,
    }

    if client:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            result = (
                client.table("attendance")
                .select("id", count="exact")
                .eq("date", today)
                .execute()
            )
            info["today_records"] = result.count if result.count is not None else len(result.data or [])
        except Exception as e:
            info["supabase_error"] = str(e)

    return info
