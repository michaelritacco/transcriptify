import streamlit as st
from initialization import initialize

st.set_page_config(
    page_title='Home',   
)

st.title('YouTubeGPT')

st.write("""
    Welcome to YouTubeGPT! In this app you can:

    - Transcribe your favourite YouTube video into proper English with TranscribeGPT
    - Ask general questions about any topic with our ChatGPT plugin
    - Use QAGPT to ask questions and get answers from your own PDF documents with references

    Please select one of the options in the sidebar to use this application.
    """)

initialize()
