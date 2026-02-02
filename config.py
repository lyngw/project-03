import os
from dotenv import load_dotenv

load_dotenv()

# Try st.secrets first (Streamlit Cloud), then .env
def _get_secret(key, default=""):
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)

DART_API_KEY = _get_secret("DART_API_KEY")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
