from langchain import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import YoutubeLoader
from openai import OpenAI
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import os

nltk.download('punkt')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(
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
    global client
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
