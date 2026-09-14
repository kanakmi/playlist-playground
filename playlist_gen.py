from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from ytmusicapi import YTMusic


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_AUTH_PATH = PROJECT_DIR / "browser.json"
YOUTUBE_MUSIC_PLAYLIST_URL = "https://music.youtube.com/playlist?list={playlist_id}"


@dataclass
class PlaylistCreationResult:
    playlist_id: str
    playlist_url: str
    added_tracks: list[dict[str, Any]]
    skipped_tracks: list[dict[str, Any]]


def get_ytmusic(auth: str | Path | dict[str, Any] | None = DEFAULT_AUTH_PATH) -> YTMusic:
    if isinstance(auth, dict):
        return YTMusic(auth=auth)

    if isinstance(auth, str) and auth.strip().startswith("{") and auth.strip().endswith("}"):
        return YTMusic(auth=auth.strip())

    if auth is not None:
        auth_path = Path(auth)
        if not auth_path.is_absolute():
            auth_path = (PROJECT_DIR / auth_path).resolve()
        else:
            auth_path = auth_path.resolve()

        if auth_path.exists():
            return YTMusic(str(auth_path))

    raise FileNotFoundError(
        "Missing YouTube Music authentication. Please upload or provide a browser.json or oauth.json file. "
        "See ytmusicapi setup guide: https://ytmusicapi.readthedocs.io/en/stable/setup.html"
    )


def build_search_query(track: pd.Series | dict[str, Any]) -> str:
    track_name = str(track.get("track_name", "")).strip()
    artist_name = str(track.get("artist_name", "")).strip()
    if artist_name:
        primary_artist = artist_name.split(",")[0].strip()
        return f"{track_name} {primary_artist}"
    return track_name


def find_best_video_id(ytmusic: YTMusic, track: pd.Series | dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    query = build_search_query(track)
    if not query:
        return None, None

    results = ytmusic.search(query, filter="songs", limit=5)
    if not results:
        results = ytmusic.search(query, filter="videos", limit=5)

    for result in results:
        video_id = result.get("videoId")
        if video_id:
            return video_id, result
    return None, None


def create_youtube_music_playlist(
    playlist_df: pd.DataFrame,
    title: str,
    description: str = "",
    privacy_status: str = "PRIVATE",
    auth: str | Path | dict[str, Any] | None = DEFAULT_AUTH_PATH,
    auth_path: str | Path | None = None,
) -> PlaylistCreationResult:
    if playlist_df.empty:
        raise ValueError("Cannot create a playlist from an empty dataframe.")

    target_auth = auth_path if auth_path is not None else auth
    ytmusic = get_ytmusic(target_auth)
    playlist_id = ytmusic.create_playlist(
        title=title,
        description=description,
        privacy_status=privacy_status,
    )

    video_ids: list[str] = []
    added_tracks: list[dict[str, Any]] = []
    skipped_tracks: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()

    for _, track in playlist_df.iterrows():
        video_id, match = find_best_video_id(ytmusic, track)
        track_payload = {
            "track_name": track.get("track_name"),
            "artist_name": track.get("artist_name"),
            "album_name": track.get("album_name"),
            "search_query": build_search_query(track),
        }
        if not video_id or video_id in seen_video_ids:
            skipped_tracks.append(track_payload)
            continue

        seen_video_ids.add(video_id)
        video_ids.append(video_id)
        added_tracks.append({**track_payload, "video_id": video_id, "ytmusic_match": match})

    if video_ids:
        ytmusic.add_playlist_items(playlist_id, video_ids, duplicates=False)

    return PlaylistCreationResult(
        playlist_id=playlist_id,
        playlist_url=YOUTUBE_MUSIC_PLAYLIST_URL.format(playlist_id=playlist_id),
        added_tracks=added_tracks,
        skipped_tracks=skipped_tracks,
    )


if __name__ == "__main__":
    sample = pd.DataFrame(
        [
            {
                "track_name": "Wonderwall",
                "artist_name": "Oasis",
                "album_name": "(What's The Story) Morning Glory?",
            }
        ]
    )
    result = create_youtube_music_playlist(sample, title="Curator smoke test", description="Created by playlist_gen.py")
    print(result.playlist_url)
