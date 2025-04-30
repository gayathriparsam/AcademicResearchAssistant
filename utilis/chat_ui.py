import streamlit as st
from datetime import datetime

def display_chat_message(role, content, timestamp):
    """Display a chat message with styling based on role (user/bot)."""
    if role == "user":
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-start; margin: 10px 0;'>
                <div style='background-color: #E0F7FA; padding: 10px; border-radius: 10px; max-width: 70%;'>
                    <strong>You:</strong> {content}<br>
                    <small>{timestamp}</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
                <div style='background-color: #C8E6C9; padding: 10px; border-radius: 10px; max-width: 70%;'>
                    <strong>Bot:</strong> {content}<br>
                    <small>{timestamp}</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )