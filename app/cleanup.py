import os
import shutil
import time
from datetime import datetime, timedelta
from config import SESSION_DATA_BASE, SESSION_PERSIST_BASE

# Configuration: Delete sessions older than this many hours
# Note: This is a fallback. Ideally, sessions are cleaned when browser closes.
# Set to 1 hour to quickly reclaim storage from abandoned sessions.
SESSION_EXPIRY_HOURS = 1


def cleanup_old_sessions(expiry_hours=SESSION_EXPIRY_HOURS):
    """
    Remove session directories that haven't been modified in the specified hours.
    This helps reduce storage costs by removing ephemeral user data.
    """
    current_time = time.time()
    expiry_seconds = expiry_hours * 3600
    
    cleaned_count = 0
    
    # Clean up data directories
    if os.path.exists(SESSION_DATA_BASE):
        for session_id in os.listdir(SESSION_DATA_BASE):
            session_path = os.path.join(SESSION_DATA_BASE, session_id)
            if os.path.isdir(session_path):
                # Check last modification time
                last_modified = os.path.getmtime(session_path)
                age_seconds = current_time - last_modified
                
                if age_seconds > expiry_seconds:
                    try:
                        shutil.rmtree(session_path)
                        print(f"Cleaned up data session: {session_id} (age: {age_seconds/3600:.1f} hours)")
                        cleaned_count += 1
                    except Exception as e:
                        print(f"Error cleaning data session {session_id}: {e}")
    
    # Clean up persist directories
    if os.path.exists(SESSION_PERSIST_BASE):
        for session_id in os.listdir(SESSION_PERSIST_BASE):
            session_path = os.path.join(SESSION_PERSIST_BASE, session_id)
            if os.path.isdir(session_path):
                # Check last modification time
                last_modified = os.path.getmtime(session_path)
                age_seconds = current_time - last_modified
                
                if age_seconds > expiry_seconds:
                    try:
                        shutil.rmtree(session_path)
                        print(f"Cleaned up persist session: {session_id} (age: {age_seconds/3600:.1f} hours)")
                        cleaned_count += 1
                    except Exception as e:
                        print(f"Error cleaning persist session {session_id}: {e}")
    
    if cleaned_count > 0:
        print(f"Total sessions cleaned: {cleaned_count}")
    else:
        print("No old sessions to clean up")
    
    return cleaned_count


if __name__ == "__main__":
    cleanup_old_sessions()
