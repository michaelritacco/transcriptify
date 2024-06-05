import nltk
import streamlit as st

# Cache the download to avoid repeated downloads
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')

# Function to perform all necessary initializations
def initialize():
    download_nltk_resources()