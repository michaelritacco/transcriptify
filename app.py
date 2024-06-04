from langchain import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import YoutubeLoader
from openai import OpenAI
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt')

import streamlit as st

import pandas as pd
import numpy as np

import os

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=OPENAI_API_KEY,
)

# def load_LLM(openai_api_key):
#     llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name='gpt-4o')
#     return llm

def get_youtube_transcript(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    documents = loader.load()
    transcript = ' '.join([doc.page_content for doc in documents])
    return transcript

def process_transcript(transcription):
    # Split the transcription by token limit
    segments = split_text_by_token_limit(transcription, max_tokens=500)

    # Process each segment to format it properly using ChatGPT
    formatted_segments = [process_segment(segment) for segment in segments]

    # Reassemble the formatted segments into the full transcription
    formatted_transcription = reassemble_text(formatted_segments)

    return formatted_transcription

def process_segment(segment):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "user", "content": "Cleanup the following transcript into paragraphs. Make sure it has proper punction and grammer:"},
            {"role": "system", "content": segment}
        ]
    )

    return response.choices[0].message.content

def split_text_by_token_limit(text, max_tokens=100):
    words = word_tokenize(text)
    segments = []
    current_segment = []
    current_token_count = 0

    for word in words:
        current_segment.append(word)
        current_token_count += 1
        if current_token_count >= max_tokens:
            segments.append(' '.join(current_segment))
            current_segment = []
            current_token_count = 0

    if current_segment:
        segments.append(' '.join(current_segment))

    return segments

def reassemble_text(segments):
    return "\n\n".join(segments)

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
    transcript = get_youtube_transcript(youtube_video)
    col1.download_button('Download Raw Transcription', transcript)
    col1.write(transcript)

    formatted_transcript = process_transcript(transcript)
    col2.download_button('Download ChatGPT Transcription', formatted_transcript)
    col2.write(formatted_transcript)