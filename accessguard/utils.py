# utils.py
import socket, platform
from datetime import datetime
import requests
import pandas as pd
from config import LOGINS_CSV
import pandas as pd
import requests
from datetime import datetime, timedelta

# =============================
# 📍 1. Get Geolocation by IP
# =============================

def get_ip():
    try:
        return socket.gethostbyname(socket.gethostname())
    except:
        return "0.0.0.0"

def get_device_info():
    return f"{platform.system()} {platform.release()} ({platform.machine()})"

def get_browser_info():
    return "Streamlit-App"

def get_hour():
    return datetime.now().hour


def get_geolocation(ip_address: str) -> dict:
    """
    Returns the geolocation info (city, region, country) for a given IP address.
    Uses the 'ipapi.co' free API.
    """
    try:
        response = requests.get(f"https://ipapi.co/{ip_address}/json/")
        if response.status_code == 200:
            data = response.json()
            return {
                "ip": ip_address,
                "city": data.get("city", "Unknown"),
                "region": data.get("region", "Unknown"),
                "country": data.get("country_name", "Unknown"),
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude")
            }
        else:
            return {"ip": ip_address, "error": "API request failed"}
    except Exception as e:
        return {"ip": ip_address, "error": str(e)}

# ==================================
# 🕒 2. Login Attempts in Last Hour
# ==================================
def login_attempts_in_last_hour(username: str, log_path: str = "data/logins.csv") -> int:
    """
    Count how many login attempts a specific user made in the last 1 hour.
    """
    try:
        df = pd.read_csv(log_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_attempts = df[
            (df["username"] == username) & (df["timestamp"] > one_hour_ago)
        ]
        return len(recent_attempts)
    except FileNotFoundError:
        return 0
    except Exception:
        return 0

# =========================
# 📁 3. Load Logins Dataset
# =========================
def load_logins(log_path: str = "data/logins.csv") -> pd.DataFrame:
    """
    Load login logs from a CSV file for ML training or analysis.
    If file doesn't exist, return an empty DataFrame.
    """
    try:
        df = pd.read_csv(log_path)
        return df
    except FileNotFoundError:
        columns = ["timestamp", "username", "ip", "device", "browser", "risk_score", "status"]
        return pd.DataFrame(columns=columns)


def login_attempts_in_last_hour(username):
    logins = pd.read_csv(LOGINS_CSV)
    if logins.empty:
        return 0

    logins["Timestamp"] = pd.to_datetime(logins["Timestamp"], errors="coerce")
    one_hour_ago = datetime.now() - pd.Timedelta(hours=1)
    mask = (logins["Username"] == username) & (logins["Timestamp"] >= one_hour_ago)
    return mask.sum()
