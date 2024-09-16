from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from openai import OpenAI
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import os
from langchain_core.prompts import ChatPromptTemplate
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(api_key=OPENAI_API_KEY, model='gpt-4o')

prompt = ChatPromptTemplate.from_messages(
    [
        ('user', 'Cleanup the following transcript into paragraphs. Make sure it has proper punction and grammer.'),
        ('system', '{text}'),
    ]
)

chain = prompt | llm
splitter = RecursiveCharacterTextSplitter(chunk_size=3500)

def get_youtube_transcript(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    documents = loader.load()
    transcript = ' '.join([doc.page_content for doc in documents])
    return transcript

def process_transcript(transcription):
    chunks = splitter.split_text(transcription)

    cleaned_transcript = process_chunk_parallel(chunks)
    return cleaned_transcript

def process_chunk(chunk):
    prompt.format_messages(text=chunk)
    response = chain.invoke(chunk).content
    return response

def process_chunk_parallel(chunks):
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_chunk, chunks))
    joined_results = '\n'.join(results)
    return joined_results

