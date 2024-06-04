import nltk
import youtubegpt as YtGPT

nltk.download('punkt')

import streamlit as st
import pandas as pd
import numpy as np
import os

st.title('YouTubeGPT')
st.write('Paste a YouTube URL below')

youtube_video = st.text_input(label="YouTube URLs",  
                              placeholder="Ex: https://www.youtube.com/watch?v=c_hO_fjmMnk", 
                              key="youtube_user_input")


button_ind = st.button("*Generate Output*", 
                       type='secondary', 
                       help="Click to generate output based on information")

col1, col2 = st.columns(2)

col1.header('Raw Transcription:')
col2.header('GPT Transcription:')

if button_ind:
    transcript = YtGPT.get_youtube_transcript(youtube_video)
    col1.download_button('Download Raw Transcription', transcript)
    col1.write(transcript)

    formatted_transcript = YtGPT.process_transcript(transcript)
    col2.download_button('Download ChatGPT Transcription', formatted_transcript)
    col2.write(formatted_transcript)