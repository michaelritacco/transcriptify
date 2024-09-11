import youtubegpt as YtGPT
import streamlit as st

st.title('Transcribe GPT')

st.write("Paste a YouTube URL below to get your transcription. Please note runtime will somewhat varry depending on the length of YouTube video.")

youtube_video = st.text_input(label="YouTube URL",  
                              placeholder="Ex: https://www.youtube.com/watch?v=c_hO_fjmMnk", 
                              key="youtube_user_input")

# Cache the transcription functions
@st.cache_data(show_spinner=False)
def get_transcripts(video_url):
    raw_transcript = YtGPT.get_youtube_transcript(video_url)
    formatted_transcript = YtGPT.process_transcript(raw_transcript)
    return raw_transcript, formatted_transcript

# Initialize session state to store the transcripts
if 'raw_transcript' not in st.session_state:
    st.session_state['raw_transcript'] = None
if 'formatted_transcript' not in st.session_state:
    st.session_state['formatted_transcript'] = None

# Button to trigger the transcription process
button_ind = st.button("*Generate Output*", 
                       type='secondary', 
                       help="Click to generate output based on information")

col1, col2 = st.columns(2)
col1.header('Raw Transcription:')
col2.header('GPT Transcription:')


# Process only if the button is clicked
if button_ind and youtube_video:
    try:
        # Get the cached transcripts (or compute if not cached yet)
        raw_transcript, formatted_transcript = get_transcripts(youtube_video)
        
        # Store results in session state so they persist after interactions
        st.session_state['raw_transcript'] = raw_transcript
        st.session_state['formatted_transcript'] = formatted_transcript
    except Exception as e:
        st.warning('Please enter a valid YouTube URL.')

# Display the transcripts if they exist in session state
if st.session_state['raw_transcript']:
    col1.download_button(
        label='Download Raw Transcription', 
        data=st.session_state['raw_transcript'],
        file_name='Raw Transcription.txt',
    )
    col1.write(st.session_state['raw_transcript'])

if st.session_state['formatted_transcript']:
    col2.download_button(
        label='Download ChatGPT Transcription', 
        data=st.session_state['formatted_transcript'],
        file_name='GPT Transcription.txt'
    )
    col2.write(st.session_state['formatted_transcript'])
