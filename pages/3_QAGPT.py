from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma

from operator import itemgetter
import streamlit as st
import tempfile
import os
import pandas as pd

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.title('QARAG Chatbot')

st.write('Please upload a PDF document and begin asking questions.')

@st.cache_resource(ttl='1h')
def configure_retriever(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()

    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, 'wb') as f:
            f.write(file.getvalue())
        loader = PyMuPDFLoader(temp_filepath)
        docs.extend(loader.load())

        # Split into document chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

        doc_chunks = text_splitter.split_documents(docs)

        # Ceate document embeddings and store in Vector DB
        embeddings_model = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(doc_chunks, embeddings_model)

        # Define retriver object
        retriever = vectordb.as_retriever()
        return retriever

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=''):
        self.container = container
        self.text = initial_text
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

uploaded_files = st.sidebar.file_uploader(
    label='Upload PDF Files', type=['pdf'],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info('Please upload PDF documents to continue.')
    st.stop()

retrivever = configure_retriever(uploaded_files=uploaded_files)

chatgpt = ChatOpenAI(temperature=0.1, streaming=True)

qa_template = """
    Use only the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know,
    don't try to make up an answer. Keep the answer as concise as possible.

    {context}

    Question: {question}
"""

qa_prompt = ChatPromptTemplate.from_template(qa_template)

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

qa_rag_chain = (
    {
        'context': itemgetter('question')
            |
        retrivever
            |
        format_docs,
        'question': itemgetter('question')
    }
        |
    qa_prompt
        |
    chatgpt
)

streamlit_msg_history = StreamlitChatMessageHistory(key='langchain_messages')

if len(streamlit_msg_history.messages) == 0:
    streamlit_msg_history.add_ai_message('Please ask me any questions about your documents.')

for msg in streamlit_msg_history.messages:
    st.chat_message(msg.type).write(msg.content)

class PostMessageHandler(BaseCallbackHandler):
    def __init__(self, msg: st.write):
        BaseCallbackHandler.__init__(self)
        self.msg = msg
        self.sources = []

    def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
        source_ids = []
        for d in documents:
            metadata = {
                'source': d.metadata['source'],
                'page': d.metadata['page'],
                'content': d.page_content[:200]
            }
            idx = (metadata['source'], metadata['page'])
            if idx not in source_ids:
                source_ids.append(idx)
                self.sources.append(metadata)

    def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
        if len(self.sources):
            st.markdown('__Sources:__' + '\n')
            # st.dataframe(data=pd.DataFrame(self.sources[:3]), width=1000)
            source_df = pd.DataFrame(self.sources[:3])
            source_df.columns = ['Source', 'Page', 'Content Snippet']
            source_df['Source'] = source_df['Source'].apply(lambda x: x.split('/')[-1])
            st.dataframe(source_df, width=1000)

if user_prompt := st.chat_input():
    st.chat_message('human').write(user_prompt)
    with st.chat_message('ai'):
        stream_handler = StreamHandler(st.empty())

        sources_container = st.write('')

        pm_handler = PostMessageHandler(sources_container)

        config = {'callbacks': [stream_handler, pm_handler]}

        response = qa_rag_chain.invoke({'question': user_prompt}, config)

        






