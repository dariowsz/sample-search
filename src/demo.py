import subprocess
from pathlib import Path

import librosa
import streamlit as st

from database import ChromaDatabase
from models import MSClap
from search import SearchEngine
from settings import env

# Set page layout to wide mode
st.set_page_config(layout="wide", page_title="Sample Search")


@st.cache_resource
def initialize_search_engine():
    return SearchEngine(
        vector_db=ChromaDatabase(
            collection_name=env.VECTOR_DB_COLLECTION_NAME,
            host=env.VECTOR_DB_HOST,
            port=env.VECTOR_DB_PORT,
        ),
        encoder_model=MSClap(
            weights_path=env.ENCODER_MODEL_WEIGHTS_PATH, device=env.ENCODER_MODEL_DEVICE
        ),
    )


search_engine = initialize_search_engine()


st.title("Sample Search 🎧")
st.subheader("AI powered semantic search engine for music samples retrieval")
st.divider()

tab1, tab2 = st.tabs(["Search", "Index New Samples"])

# Search tab
with tab1:
    col1, _, col2, _ = st.columns([0.45, 0.05, 0.45, 0.05])

    # Text input
    with col1:
        st.text_input(
            "Enter search query", key="search_query", placeholder="e.g. Happy melody"
        )
        if st.button("Search"):
            search_query = st.session_state.search_query
            with st.spinner("Searching for samples..."):
                st.session_state.results = search_engine.query(search_query)

    # Audio player
    with col2:
        if (
            "search_query" in st.session_state
            and st.session_state.search_query
            and "results" in st.session_state
        ):
            if len(st.session_state.results) > 0:
                for result in st.session_state.results:
                    st.write(f"Playing: {result.split('/')[-1]}")
                    audio_data, sr = librosa.load(result)

                    col_audio, col_button = st.columns([0.7, 0.3])
                    with col_audio:
                        st.audio(audio_data, sample_rate=int(sr))
                    with col_button:
                        if st.button(f"Show in Finder", key=f"finder_{result}"):
                            subprocess.run(["open", "-R", result])
            else:
                st.write("No results found")
        else:
            st.write("Results will appear here")

# Index Tab
with tab2:
    st.header("Index New Samples")

    col1, _, col2, _ = st.columns([0.45, 0.05, 0.45, 0.05])

    with col2:
        # Display already indexed folders
        st.subheader("Already Indexed Folders:")
        try:
            indexed_dirs = search_engine.get_indexed_dirs()
            if indexed_dirs:
                container = st.container()
                for idx, folder in enumerate(indexed_dirs):
                    with container:
                        col_a, col_b = st.columns([0.8, 0.2])
                        with col_a:
                            st.text(folder.strip())
                        with col_b:
                            if st.button("Remove", key=f"remove_indexed_{idx}"):
                                search_engine.remove_indexed_dir(folder.strip())
                                st.rerun()
            else:
                st.info("No folders have been indexed yet")
        except FileNotFoundError:
            st.info("No folders have been indexed yet")

    with col1:
        # Initialize session state for folder paths if not exists
        if "folder_paths" not in st.session_state:
            st.session_state.folder_paths = []

        # Add folder input
        new_folder = st.text_input("Enter folder path")
        if st.button("Add Folder"):
            if new_folder and Path(new_folder).exists():
                if new_folder not in st.session_state.folder_paths:
                    st.session_state.folder_paths.append(new_folder)
                    st.success(f"Added: {new_folder}")
                else:
                    st.warning("Folder already added!")
            else:
                st.error("Invalid folder path!")

        # Display and manage added folders
        st.subheader("Selected Folders:")
        for idx, folder in enumerate(st.session_state.folder_paths):
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.text(folder)
            with col2:
                if st.button("Remove", key=f"remove_{idx}"):
                    st.session_state.folder_paths.pop(idx)
                    st.rerun()

        # Index button
        if st.session_state.folder_paths:
            if st.button("Index Samples"):
                with st.spinner("Indexing samples. This might take a while..."):
                    try:
                        count = search_engine.index_dirs(st.session_state.folder_paths)
                        st.session_state.folder_paths = []
                        st.toast(f"Successfully indexed {count} samples!", icon="✅")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during indexing: {str(e)}")
        else:
            st.info("Add folders to index")
