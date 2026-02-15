"""
Session cleanup handler for ephemeral storage.
This module provides functionality to clean up session data when users close the browser.
"""
import os
import shutil
from pathlib import Path


def cleanup_session(session_id: str, session_data_base: str, session_persist_base: str):
    """
    Clean up all data for a specific session.
    
    Args:
        session_id: The unique session identifier
        session_data_base: Base directory for session data
        session_persist_base: Base directory for session persistence
    """
    session_data_path = os.path.join(session_data_base, session_id)
    session_persist_path = os.path.join(session_persist_base, session_id)
    
    cleaned = False
    
    if os.path.exists(session_data_path):
        try:
            shutil.rmtree(session_data_path)
            print(f"Cleaned up session data: {session_id}")
            cleaned = True
        except Exception as e:
            print(f"Error cleaning session data {session_id}: {e}")
    
    if os.path.exists(session_persist_path):
        try:
            shutil.rmtree(session_persist_path)
            print(f"Cleaned up session persist: {session_id}")
            cleaned = True
        except Exception as e:
            print(f"Error cleaning session persist {session_id}: {e}")
    
    return cleaned


def register_session_cleanup_handler():
    """
    Register a cleanup handler that will be called when the Streamlit session ends.
    This uses Streamlit's session state to track when to cleanup.
    """
    import streamlit as st
    from config import SESSION_DATA_BASE, SESSION_PERSIST_BASE
    
    # Mark session as active
    if "session_active" not in st.session_state:
        st.session_state.session_active = True
        
    # Add JavaScript to send cleanup signal on page unload
    cleanup_script = """
    <script>
    // Send cleanup signal when user closes tab or navigates away
    window.addEventListener('beforeunload', function(e) {
        // Use sendBeacon for reliable delivery even during page unload
        const sessionId = '%s';
        const cleanupUrl = window.location.origin + '/_stcore/cleanup?session_id=' + sessionId;
        
        // Note: This requires a backend endpoint to handle cleanup
        // For now, we rely on session timeout and manual cleanup
        console.log('Session cleanup triggered for:', sessionId);
    });
    </script>
    """ % st.session_state.session_id
    
    st.components.v1.html(cleanup_script, height=0)
