import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from find_songs import DEFAULT_DATASET_PATH, EMBEDDING_MODEL, NVIDIA_MODEL, PlaylistCurator, PROJECT_DIR
from playlist_gen import DEFAULT_AUTH_PATH, create_youtube_music_playlist


PAGE_TITLE = "Playlist Curator"
DEFAULT_PROMPT = "I want music for a late-night drive in the mountains. Atmospheric, slightly melancholic, not too slow."
LANGUAGE_OPTIONS = ["Auto", "English", "Hindi", "Tamil", "Telugu", "Malayalam", "Korean"]


st.set_page_config(page_title=PAGE_TITLE, page_icon=":material/library_music:", layout="wide")


@st.cache_resource(show_spinner="Loading playlist curator and FAISS indexes...")
def load_curator(dataset_path: str, embedding_model: str, llm_model: str, llm_timeout: int) -> PlaylistCurator:
    return PlaylistCurator(
        dataset_path=dataset_path,
        embedding_model=embedding_model,
        llm_model=llm_model,
        llm_timeout=llm_timeout,
    )


def make_playlist_title(prompt: str) -> str:
    cleaned = " ".join(prompt.strip().split())
    if not cleaned:
        cleaned = "Curated playlist"
    return f"AI Curator - {cleaned[:48]}"


def playlist_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def render_track_card(row: pd.Series, rank: int) -> None:
    artwork = row.get("artwork_url")
    track_url = row.get("track_url")

    with st.container(border=True):
        cols = st.columns([0.8, 4.2, 1.1, 1.1])
        with cols[0]:
            if isinstance(artwork, str) and artwork.startswith("http"):
                st.image(artwork, width=72)
            else:
                st.caption(f"#{rank}")
        with cols[1]:
            title = str(row.get("track_name", "Unknown track"))
            artist = str(row.get("artist_name", "Unknown artist"))
            album = str(row.get("album_name", "Unknown album"))
            if isinstance(track_url, str) and track_url.startswith("http"):
                st.markdown(f"**{rank}. [{title}]({track_url})**")
            else:
                st.markdown(f"**{rank}. {title}**")
            st.caption(f"{artist} · {album}")
        with cols[2]:
            st.metric("Popularity", int(row.get("popularity", 0)))
        with cols[3]:
            score = row.get("final_score", 0)
            st.metric("Fit", f"{float(score):.2f}")


st.title(PAGE_TITLE)
st.caption("Describe a moment, tune the retrieval, preview the songs, then create a YouTube Music playlist.")

with st.sidebar:
    st.header("Retrieval")
    language_choice = st.selectbox("Language", LANGUAGE_OPTIONS, index=0)
    n_songs = st.slider("Playlist size", min_value=5, max_value=50, value=25, step=5)
    popular_ratio = st.slider("Popular songs", min_value=0, max_value=100, value=70, step=5)
    candidate_k = st.slider("Candidate pool", min_value=250, max_value=8000, value=2000, step=250)

    st.header("Models")
    embedding_model = st.text_input("Embedding model", value=EMBEDDING_MODEL)
    llm_model = st.text_input("Nvidia NIM model", value=NVIDIA_MODEL)
    llm_timeout = st.slider("LLM timeout (seconds)", min_value=10, max_value=120, value=60, step=5)

    st.header("YouTube Music")
    create_on_ytmusic = st.checkbox("Create playlist on YouTube Music", value=False)
    privacy_status = st.selectbox("Privacy", ["PRIVATE", "UNLISTED", "PUBLIC"], index=0)

    auth_payload: str | Path | dict | None = None
    if create_on_ytmusic:
        st.caption(
            "Requires YouTube Music authentication (`browser.json` or `oauth.json`). "
            "See the [ytmusicapi setup guide](https://ytmusicapi.readthedocs.io/en/stable/setup.html) to create your own auth file."
        )
        auth_mode = st.radio("Auth source", ["Upload auth JSON", "File path or raw JSON"], horizontal=True)
        if auth_mode == "Upload auth JSON":
            uploaded_auth = st.file_uploader(
                "Upload auth JSON",
                type=["json"],
                help="Upload browser.json or oauth.json created with ytmusicapi",
            )
            if uploaded_auth is not None:
                try:
                    auth_payload = json.loads(uploaded_auth.getvalue().decode("utf-8"))
                except Exception as e:
                    st.error(f"Invalid JSON file: {e}")
            elif DEFAULT_AUTH_PATH.exists():
                st.caption(f"Using default local auth file: `{DEFAULT_AUTH_PATH.name}`")
                auth_payload = DEFAULT_AUTH_PATH
            else:
                st.info("Upload your `browser.json` or `oauth.json` above to publish playlists.")
        else:
            auth_text = st.text_area(
                "Path or raw JSON",
                value=str(DEFAULT_AUTH_PATH) if DEFAULT_AUTH_PATH.exists() else "",
                placeholder="Relative/absolute path or paste raw JSON here...",
                height=80,
            )
            if auth_text.strip():
                auth_payload = auth_text.strip()

prompt = st.text_area("Prompt", value=DEFAULT_PROMPT, height=120, placeholder="Describe the mood, scene, language, pace, or activity...")

col_a, col_b = st.columns([1, 1])
with col_a:
    playlist_title = st.text_input("Playlist title", value=make_playlist_title(prompt))
with col_b:
    playlist_description = st.text_input(
        "Playlist description",
        value=f"Generated from: {prompt[:120]}",
    )

generate = st.button("Generate Playlist", type="primary", use_container_width=True)

if generate:
    if not prompt.strip():
        st.error("Give me a prompt first.")
        st.stop()

    language = None if language_choice == "Auto" else language_choice
    dataset = DEFAULT_DATASET_PATH
    if not dataset.exists():
        st.error(f"Dataset not found: {dataset}")
        st.stop()

    with st.spinner("Finding songs with hybrid semantic + audio retrieval..."):
        curator = load_curator(str(dataset), embedding_model, llm_model, llm_timeout)
        playlist = curator.generate_playlist(
            query=prompt,
            language=language,
            n_songs=n_songs,
            popular_ratio=popular_ratio / 100,
            candidate_k=candidate_k,
        )

    st.session_state["playlist"] = playlist
    st.session_state["prompt"] = prompt
    st.session_state["playlist_title"] = playlist_title
    st.session_state["playlist_description"] = playlist_description

playlist = st.session_state.get("playlist")

if playlist is not None:
    if playlist.empty:
        st.warning("No songs matched those filters. Try Auto language or a larger candidate pool.")
        st.stop()

    st.subheader("Generated Playlist")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Tracks", len(playlist))
    metric_cols[1].metric("Avg popularity", f"{playlist['popularity'].mean():.0f}")
    metric_cols[2].metric("Languages", playlist["language"].nunique())
    metric_cols[3].metric("Albums", playlist["album_name"].nunique())

    action_cols = st.columns([1, 1])
    with action_cols[0]:
        st.download_button(
            "Download CSV",
            data=playlist_to_csv(playlist),
            file_name=f"playlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with action_cols[1]:
        st.download_button(
            "Download JSON",
            data=playlist.to_json(orient="records", force_ascii=False, indent=2),
            file_name=f"playlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

    if create_on_ytmusic:
        if st.button("Create YouTube Music Playlist", use_container_width=True):
            if auth_payload is None:
                st.error(
                    "Please upload or specify your YouTube Music auth file (`browser.json` or `oauth.json`) in the sidebar. "
                    "See the [ytmusicapi setup guide](https://ytmusicapi.readthedocs.io/en/stable/setup.html)."
                )
            else:
                with st.spinner("Searching YouTube Music and creating the playlist..."):
                    try:
                        result = create_youtube_music_playlist(
                            playlist,
                            title=st.session_state.get("playlist_title", playlist_title),
                            description=st.session_state.get("playlist_description", playlist_description),
                            privacy_status=privacy_status,
                            auth=auth_payload,
                        )
                    except Exception as exc:
                        st.error(f"Could not create YouTube Music playlist: {exc}")
                    else:
                        st.success(f"Created playlist with {len(result.added_tracks)} tracks.")
                        st.link_button("Open YouTube Music Playlist", result.playlist_url, use_container_width=True)
                        if result.skipped_tracks:
                            st.warning(f"Skipped {len(result.skipped_tracks)} tracks that could not be matched.")
                            with st.expander("Skipped tracks"):
                                st.dataframe(pd.DataFrame(result.skipped_tracks), use_container_width=True)
    else:
        st.info("Enable YouTube Music creation in the sidebar when you are ready to publish the playlist.")

    view = st.radio("View", ["Cards", "Table"], horizontal=True)
    if view == "Table":
        st.dataframe(playlist, use_container_width=True, hide_index=True)
    else:
        for rank, (_, row) in enumerate(playlist.iterrows(), start=1):
            render_track_card(row, rank)
else:
    st.info("Enter a prompt and click Generate Playlist.")
