from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

load_dotenv()


def get_secret(key: str, default: str | None = None) -> str | None:
    """Retrieve secret from Streamlit secrets (local or cloud) with fallback to os.environ / .env."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            val = st.secrets[key]
            if val is not None:
                return str(val)
    except Exception:
        pass
    val = os.getenv(key)
    if val is not None:
        return val
    return default


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = PROJECT_DIR / "artists_datasets" / "language.csv"
DEFAULT_CACHE_DIR = PROJECT_DIR / ".playlist_cache"

HF_TOKEN = get_secret("HF_TOKEN")
NVIDIA_API_KEY = get_secret("NVIDIA_API_KEY")
NVIDIA_MODEL = get_secret("NVIDIA_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
NVIDIA_BASE_URL = get_secret("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Backward compatibility aliases
OLLAMA_MODEL = NVIDIA_MODEL
OLLAMA_CHAT_URL = NVIDIA_BASE_URL

EMBEDDING_MODEL = get_secret(
    "PLAYLIST_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
TEMPO_MAX = 250.0

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "valence",
    "tempo_norm",
]

OUTPUT_COLUMNS = [
    "track_id",
    "track_name",
    "artist_name",
    "album_name",
    "language",
    "year",
    "popularity",
    "final_score",
    "semantic_score",
    "audio_score",
    "track_url",
    "artwork_url",
]


@dataclass
class PlaylistIntent:
    query: str
    language: str | None = None
    moods: list[str] = field(default_factory=list)
    activity: str = ""
    keywords: list[str] = field(default_factory=list)
    semantic_queries: list[str] = field(default_factory=list)
    danceability: float = 0.5
    energy: float = 0.5
    acousticness: float = 0.35
    instrumentalness: float = 0.1
    speechiness: float = 0.08
    liveness: float = 0.12
    valence: float = 0.5
    tempo: float = 105.0

    def audio_vector(self) -> np.ndarray:
        values = [
            self.danceability,
            self.energy,
            self.acousticness,
            self.instrumentalness,
            self.speechiness,
            self.liveness,
            self.valence,
            self.tempo / TEMPO_MAX,
        ]
        vector = np.asarray(values, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vector)
        return vector

    def search_text(self) -> str:
        parts = [
            self.query,
            self.language or "",
            self.activity,
            " ".join(self.moods),
            " ".join(self.keywords),
            " ".join(self.semantic_queries),
        ]
        return " ".join(part for part in parts if part).strip()


def clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return max(low, min(high, number))


def normalize_language(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"any", "all", "none", "null", "unknown"}:
        return None
    return text.title()


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


class NvidiaNimIntentParser:
    def __init__(
        self,
        model: str = NVIDIA_MODEL,
        api_key: str | None = None,
        base_url: str = NVIDIA_BASE_URL,
        timeout: int = 60,
    ):
        self.model = model
        self.api_key = api_key if api_key is not None else get_secret("NVIDIA_API_KEY")
        self.base_url = (base_url or NVIDIA_BASE_URL).rstrip("/")
        self.timeout = timeout

    def parse(self, query: str, language: str | None = None) -> PlaylistIntent:
        prompt = f"""
Analyze this music playlist request and return only one JSON object.

Request: {query}
Explicit language argument: {language or "not provided"}

Infer a playlist retrieval intent. Use audio feature values from 0.0 to 1.0,
except tempo, which is BPM from 40 to 220. Do not select songs. Do not include
markdown.

JSON schema:
{{
  "language": "Hindi | English | Tamil | Telugu | Malayalam | Korean | null",
  "moods": ["short mood words"],
  "activity": "short activity or scene",
  "keywords": ["search words for semantic retrieval"],
  "semantic_queries": ["2-4 rewritten playlist search phrases"],
  "danceability": 0.0,
  "energy": 0.0,
  "acousticness": 0.0,
  "instrumentalness": 0.0,
  "speechiness": 0.0,
  "liveness": 0.0,
  "valence": 0.0,
  "tempo": 100
}}
"""
        fallback = heuristic_intent(query, language)
        if not self.api_key:
            print("Nvidia NIM API key not configured; using heuristic fallback.")
            return fallback

        try:
            raw = self._chat(prompt)
            print(f"Nvidia NIM raw response: {raw}")
        except Exception as exc:
            print(f"Nvidia NIM intent parsing failed; using heuristic fallback: {exc}")
            return fallback

        payload = parse_json_object(raw)
        if not payload:
            return fallback

        print(f"Nvidia NIM intent payload: {payload}")

        explicit_language = normalize_language(language)
        return PlaylistIntent(
            query=query,
            language=explicit_language or normalize_language(payload.get("language")) or fallback.language,
            moods=clean_string_list(payload.get("moods")) or fallback.moods,
            activity=str(payload.get("activity") or fallback.activity).strip(),
            keywords=clean_string_list(payload.get("keywords")) or fallback.keywords,
            semantic_queries=clean_string_list(payload.get("semantic_queries")) or fallback.semantic_queries,
            danceability=clamp(payload.get("danceability"), 0, 1, fallback.danceability),
            energy=clamp(payload.get("energy"), 0, 1, fallback.energy),
            acousticness=clamp(payload.get("acousticness"), 0, 1, fallback.acousticness),
            instrumentalness=clamp(payload.get("instrumentalness"), 0, 1, fallback.instrumentalness),
            speechiness=clamp(payload.get("speechiness"), 0, 1, fallback.speechiness),
            liveness=clamp(payload.get("liveness"), 0, 1, fallback.liveness),
            valence=clamp(payload.get("valence"), 0, 1, fallback.valence),
            tempo=clamp(payload.get("tempo"), 40, 220, fallback.tempo),
        )

    def _chat(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You convert playlist requests into compact JSON retrieval intents.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }
        url = f"{self.base_url}/chat/completions"
        resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Nvidia NIM request failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"No completion choices returned: {data}")
        return choices[0].get("message", {}).get("content", "")


# Backward compatibility alias
OllamaIntentParser = NvidiaNimIntentParser


def clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned[:8]


def heuristic_intent(query: str, language: str | None = None) -> PlaylistIntent:
    text = query.lower()
    intent = PlaylistIntent(
        query=query,
        language=normalize_language(language),
        moods=[],
        activity="",
        keywords=[],
        semantic_queries=[query],
    )

    language_terms = {
        "hindi": "Hindi",
        "bollywood": "Hindi",
        "english": "English",
        "tamil": "Tamil",
        "telugu": "Telugu",
        "malayalam": "Malayalam",
        "korean": "Korean",
        "k-pop": "Korean",
        "kpop": "Korean",
    }
    if not intent.language:
        for term, detected in language_terms.items():
            if term in text:
                intent.language = detected
                break

    cues = [
        ("late night", "night drive", {"energy": 0.45, "valence": 0.42, "tempo": 95, "acousticness": 0.35}),
        ("drive", "driving", {"energy": 0.6, "danceability": 0.55, "tempo": 115}),
        ("rain", "rainy", {"energy": 0.35, "valence": 0.38, "tempo": 85, "acousticness": 0.55}),
        ("melancholic", "sad", "heartbreak", {"energy": 0.32, "valence": 0.22, "tempo": 82}),
        ("atmospheric", "ambient", {"energy": 0.35, "instrumentalness": 0.45, "speechiness": 0.04}),
        ("workout", "gym", {"energy": 0.88, "valence": 0.7, "tempo": 140, "danceability": 0.75}),
        ("party", "dance", {"energy": 0.82, "valence": 0.78, "tempo": 124, "danceability": 0.82}),
        ("romantic", "love", {"energy": 0.45, "valence": 0.62, "tempo": 92, "acousticness": 0.5}),
        ("focus", "study", "work", {"energy": 0.28, "valence": 0.45, "tempo": 80, "instrumentalness": 0.55}),
        ("epic", "war", "cinematic", {"energy": 0.72, "valence": 0.32, "tempo": 118, "instrumentalness": 0.35}),
    ]

    for *terms, attrs in cues:
        if any(term in text for term in terms):
            intent.moods.extend(term for term in terms if term in text)
            for key, value in attrs.items():
                setattr(intent, key, value)

    intent.keywords = list(dict.fromkeys(re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", query)))[:8]
    if not intent.semantic_queries:
        intent.semantic_queries = [query]
    return intent


def normalize_title(title: Any) -> str:
    text = str(title).lower()
    text = re.sub(r"\bfeat\.?\b|\bft\.?\b", "featuring", text)
    patterns = [
        r"\(.*?(remaster|live|acoustic|remix|sped up|slowed|version|edit|karaoke|lofi).*?\)",
        r"\[.*?(remaster|live|acoustic|remix|sped up|slowed|version|edit|karaoke|lofi).*?\]",
        r"\s+-\s+.*?(remaster|live|acoustic|remix|sped up|slowed|version|edit|karaoke|lofi).*",
        r"\s+\(.*?\)\s*$",
        r"\s+\[.*?\]\s*$",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_artist_key(artist_name: Any) -> str:
    artists = str(artist_name).lower().replace(";", ",").split(",")
    primary = artists[0].strip() if artists else str(artist_name).lower().strip()
    return re.sub(r"[^a-z0-9]+", " ", primary).strip()


def duplicate_signature(row: pd.Series) -> tuple[Any, ...]:
    duration = row.get("duration_ms", 0)
    try:
        duration_bucket = round(float(duration) / 10_000)
    except (TypeError, ValueError):
        duration_bucket = 0
    return (
        normalize_title(row.get("track_name")),
        normalize_artist_key(row.get("artist_name")),
        duration_bucket,
        round(float(row.get("energy", 0)), 2),
        round(float(row.get("valence", 0)), 2),
        round(float(row.get("danceability", 0)), 2),
        round(float(row.get("tempo", 0)) / 5),
    )


def feature_label(value: float, low: str, mid: str, high: str) -> str:
    if value < 0.35:
        return low
    if value > 0.68:
        return high
    return mid


def song_document(row: pd.Series) -> str:
    energy = feature_label(row.energy, "low energy", "medium energy", "high energy")
    valence = feature_label(row.valence, "melancholic", "balanced mood", "happy positive")
    dance = feature_label(row.danceability, "not dancey", "moderately danceable", "very danceable")
    acoustic = feature_label(row.acousticness, "electronic produced", "mixed acoustic", "acoustic organic")
    instrumental = "instrumental background" if row.instrumentalness > 0.45 else "vocal song"
    tempo = "slow tempo" if row.tempo < 90 else "fast tempo" if row.tempo > 130 else "mid tempo"
    return (
        f"Track: {row.track_name}. Artist: {row.artist_name}. Album: {row.album_name}. "
        f"Language: {row.language}. Year: {row.year}. Popularity: {row.popularity}. "
        f"Vibe: {energy}, {valence}, {dance}, {acoustic}, {instrumental}, {tempo}."
    )


class PlaylistCurator:
    def __init__(
        self,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = NVIDIA_MODEL,
        llm_timeout: int = 60,
        ollama_model: str | None = None,
        ollama_timeout: int | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.is_absolute():
            self.dataset_path = (PROJECT_DIR / self.dataset_path).resolve()
        else:
            self.dataset_path = self.dataset_path.resolve()

        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_absolute():
            self.cache_dir = (PROJECT_DIR / self.cache_dir).resolve()
        else:
            self.cache_dir = self.cache_dir.resolve()

        self.embedding_model_name = embedding_model
        model = ollama_model or llm_model
        timeout = ollama_timeout if ollama_timeout is not None else llm_timeout
        self.intent_parser = NvidiaNimIntentParser(model=model, timeout=timeout)

        self.df = self._load_dataset()
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.audio_index: faiss.Index
        self.semantic_index: faiss.Index
        self.audio_vectors: np.ndarray
        self._build_or_load_indexes()

    def generate_playlist(
        self,
        query: str,
        language: str | None = None,
        n_songs: int = 25,
        popular_ratio: float = 0.70,
        candidate_k: int = 2000,
    ) -> pd.DataFrame:
        intent = self.intent_parser.parse(query, language=language)
        language = normalize_language(language) or intent.language

        semantic_query = intent.search_text() or query
        semantic_vector = self.embedding_model.encode([semantic_query], normalize_embeddings=True).astype(np.float32)
        semantic_vector = np.ascontiguousarray(semantic_vector)
        audio_vector = intent.audio_vector()

        k = min(candidate_k, len(self.df))
        semantic_scores, semantic_idxs = self.semantic_index.search(semantic_vector, k)
        audio_scores, audio_idxs = self.audio_index.search(audio_vector, k)

        candidate_ids = sorted(set(semantic_idxs[0].tolist()) | set(audio_idxs[0].tolist()))
        candidates: pd.DataFrame = self.df.iloc[candidate_ids].copy()

        semantic_score_map = dict(zip(semantic_idxs[0].tolist(), semantic_scores[0].tolist()))
        audio_score_map = dict(zip(audio_idxs[0].tolist(), audio_scores[0].tolist()))
        candidates["semantic_score"] = candidates.index.map(lambda i: semantic_score_map.get(i, 0.0))
        candidates["audio_score"] = candidates.index.map(lambda i: audio_score_map.get(i, 0.0))

        if language:
            language_mask = candidates["language"].str.lower() == language.lower()
            candidates = candidates.loc[language_mask].copy()

        if candidates.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        candidates["popularity_score"] = candidates["popularity"].fillna(0).clip(0, 100) / 100
        candidates["recency_score"] = self._recency_score(candidates["year"])
        candidates["final_score"] = (
            0.42 * candidates["semantic_score"]
            + 0.33 * candidates["audio_score"]
            + 0.20 * candidates["popularity_score"]
            + 0.05 * candidates["recency_score"]
        )

        candidates = candidates.sort_values("final_score", ascending=False)
        playlist = self._build_diverse_playlist(
            candidates,
            n_songs=n_songs,
            popular_ratio=popular_ratio,
            language=language,
        )
        return playlist.reindex(columns=[col for col in OUTPUT_COLUMNS if col in playlist.columns])

    def _load_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.dataset_path)
        rename_map = {
            "artists": "artist_name",
            "name": "track_name",
            "album": "album_name",
            "id": "track_id",
        }
        df = df.rename(columns={old: new for old, new in rename_map.items() if old in df.columns})

        required = {"track_id", "track_name", "artist_name", "album_name", "popularity", *AUDIO_FEATURES[:-1]}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        if "language" not in df.columns:
            df["language"] = "Unknown"
        if "year" not in df.columns:
            df["year"] = np.nan
        if "track_url" not in df.columns:
            df["track_url"] = ""
        if "artwork_url" not in df.columns:
            df["artwork_url"] = ""

        numeric_cols = [
            "popularity",
            "year",
            "duration_ms",
            "danceability",
            "energy",
            "acousticness",
            "instrumentalness",
            "speechiness",
            "liveness",
            "valence",
            "tempo",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=AUDIO_FEATURES[:-1] + ["tempo"]).copy()
        df["tempo_norm"] = (df["tempo"].clip(0, TEMPO_MAX) / TEMPO_MAX).astype(float)
        df["popularity"] = df["popularity"].fillna(0).clip(0, 100)
        df["artist_name"] = df["artist_name"].fillna("Unknown Artist").astype(str)
        df["track_name"] = df["track_name"].fillna("Unknown Track").astype(str)
        df["album_name"] = df["album_name"].fillna("Unknown Album").astype(str)
        df["language"] = df["language"].fillna("Unknown").astype(str)
        df["semantic_text"] = df.apply(song_document, axis=1)
        return df.reset_index(drop=True)

    def _cache_key(self) -> str:
        stat = self.dataset_path.stat()
        try:
            rel_path = self.dataset_path.relative_to(PROJECT_DIR).as_posix()
        except ValueError:
            rel_path = self.dataset_path.name
        raw = f"{rel_path}:{stat.st_size}:{self.embedding_model_name}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _build_or_load_indexes(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._cache_key()
        audio_path = self.cache_dir / f"{key}.audio.faiss"
        semantic_path = self.cache_dir / f"{key}.semantic.faiss"
        audio_vectors_path = self.cache_dir / f"{key}.audio.npy"

        self.audio_vectors = np.ascontiguousarray(self.df[AUDIO_FEATURES].to_numpy(dtype=np.float32))
        faiss.normalize_L2(self.audio_vectors)

        if audio_path.exists() and semantic_path.exists() and audio_vectors_path.exists():
            self.audio_index = faiss.read_index(str(audio_path))
            self.semantic_index = faiss.read_index(str(semantic_path))
            return

        # Fallback: check if any pre-built index files already exist in cache_dir
        existing_semantics = sorted(self.cache_dir.glob("*.semantic.faiss"))
        for s_path in existing_semantics:
            candidate_key = s_path.name.replace(".semantic.faiss", "")
            a_path = self.cache_dir / f"{candidate_key}.audio.faiss"
            v_path = self.cache_dir / f"{candidate_key}.audio.npy"
            if a_path.exists() and v_path.exists():
                try:
                    self.audio_index = faiss.read_index(str(a_path))
                    self.semantic_index = faiss.read_index(str(s_path))
                    return
                except Exception:
                    pass

        self.audio_index = faiss.IndexFlatIP(len(AUDIO_FEATURES))
        self.audio_index.add(self.audio_vectors)

        semantic_embeddings = self.embedding_model.encode(
            self.df["semantic_text"].tolist(),
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        semantic_embeddings = np.ascontiguousarray(semantic_embeddings)
        self.semantic_index = faiss.IndexFlatIP(semantic_embeddings.shape[1])
        self.semantic_index.add(semantic_embeddings)

        faiss.write_index(self.audio_index, str(audio_path))
        faiss.write_index(self.semantic_index, str(semantic_path))
        np.save(audio_vectors_path, self.audio_vectors)

    @staticmethod
    def _recency_score(years: pd.Series) -> pd.Series:
        numeric_years = pd.to_numeric(years, errors="coerce")
        if numeric_years.notna().sum() == 0:
            return pd.Series(0.5, index=years.index)
        min_year = numeric_years.quantile(0.05)
        max_year = numeric_years.quantile(0.95)
        if max_year <= min_year:
            return pd.Series(0.5, index=years.index)
        return ((numeric_years.fillna(min_year) - min_year) / (max_year - min_year)).clip(0, 1)

    def _build_diverse_playlist(
        self,
        candidates: pd.DataFrame,
        n_songs: int,
        popular_ratio: float,
        language: str | None,
        max_per_album: int = 3,
        max_per_artist: int = 10,
    ) -> pd.DataFrame:
        pool = self.df
        if language:
            pool = pool[pool["language"].str.lower() == language.lower()]
        popular_cutoff = pool["popularity"].quantile(0.70) if not pool.empty else self.df["popularity"].quantile(0.70)

        target_popular = round(n_songs * popular_ratio)
        target_discovery = n_songs - target_popular
        popular = candidates[candidates["popularity"] >= popular_cutoff]
        discovery = candidates[candidates["popularity"] < popular_cutoff]

        selected: list[pd.Series] = []
        artist_counts: defaultdict[str, int] = defaultdict(int)
        album_counts: defaultdict[str, int] = defaultdict(int)
        seen_song_keys: set[tuple[str, str]] = set()
        seen_signatures: set[tuple[Any, ...]] = set()

        def try_add(rows: pd.DataFrame, limit: int | None = None) -> None:
            nonlocal selected
            for _, row in rows.iterrows():
                if len(selected) >= n_songs:
                    return
                if limit is not None and limit <= 0:
                    return

                signature = duplicate_signature(row)
                artist_key = normalize_artist_key(row.artist_name)
                song_key = (normalize_title(row.track_name), artist_key)
                album_key = str(row.album_name).lower().strip()
                if song_key in seen_song_keys or signature in seen_signatures:
                    continue
                if artist_counts[artist_key] >= max_per_artist:
                    continue
                if album_counts[album_key] >= max_per_album:
                    continue

                selected.append(row)
                seen_song_keys.add(song_key)
                seen_signatures.add(signature)
                artist_counts[artist_key] += 1
                album_counts[album_key] += 1
                if limit is not None:
                    limit -= 1

        try_add(popular, target_popular)
        try_add(discovery, target_discovery)
        if len(selected) < n_songs:
            try_add(candidates, None)

        return pd.DataFrame(selected)


_DEFAULT_CURATOR: PlaylistCurator | None = None


def generate_playlist(query: str, language: str | None = None, n_songs: int = 25) -> pd.DataFrame:
    global _DEFAULT_CURATOR
    if _DEFAULT_CURATOR is None:
        _DEFAULT_CURATOR = PlaylistCurator()
    return _DEFAULT_CURATOR.generate_playlist(query=query, language=language, n_songs=n_songs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a hybrid FAISS playlist from a natural-language query.")
    parser.add_argument("query", help="Natural language playlist request.")
    parser.add_argument("--language", default=None, help="Optional language filter, e.g. Hindi or English.")
    parser.add_argument("--n", "--n-songs", dest="n_songs", type=int, default=25)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--llm-model", default=NVIDIA_MODEL, help="Nvidia NIM model identifier.")
    parser.add_argument("--llm-timeout", type=int, default=60, help="Nvidia NIM request timeout in seconds.")
    parser.add_argument("--json", action="store_true", help="Print JSON records instead of a table.")
    args = parser.parse_args()

    curator = PlaylistCurator(
        dataset_path=args.dataset,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        llm_timeout=args.llm_timeout,
    )
    playlist = curator.generate_playlist(args.query, language=args.language, n_songs=args.n_songs)
    if args.json:
        print(playlist.to_json(orient="records", force_ascii=False, indent=2))
    else:
        display_cols = [col for col in ["track_name", "artist_name", "album_name", "language", "popularity", "final_score"] if col in playlist]
        print(playlist[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
