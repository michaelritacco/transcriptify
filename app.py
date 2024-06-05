import streamlit as st
from initialization import initialize

st.set_page_config(
    page_title='Home',   
)

st.title('YouTubeGPT')

st.write('Welcome to YouTubeGPT! Please select one of the options in the sidebar to use this application.')

# Perform initializations
initialize()
