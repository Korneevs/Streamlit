import os
from dotenv import load_dotenv
import json
from pathlib import Path

def load_credentials():
    """Load Google API credentials from environment variables or file."""
    load_dotenv()  # Load environment variables from .env file
    
    # Try to get credentials from environment variable
    creds_path = os.getenv('GOOGLE_CREDENTIALS_PATH')
    if not creds_path:
        raise ValueError("GOOGLE_CREDENTIALS_PATH not set in .env file")
    
    creds_path = Path(creds_path)
    if not creds_path.exists():
        raise FileNotFoundError(f"Credentials file not found at {creds_path}")
    
    with open(creds_path) as f:
        return json.load(f)

def get_token_path():
    """Get the path for storing OAuth token."""
    return Path.home() / '.google_token.json' 