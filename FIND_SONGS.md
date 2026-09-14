# `find_songs.py`

`find_songs.py` is the backend that turns a natural-language playlist request into a ranked, filtered list of songs. The design is intentionally split in two:

1. An **Nvidia NIM model** (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`) interprets the prompt.
2. Python does the retrieval, ranking, deduplication, and diversity filtering.

That split matters. The LLM is good at understanding a sentence like "songs that feel like reading old messages you should have deleted years ago," but the backend should stay in charge of numeric scoring and playlist rules.

## What the file does

The module exposes one main entry point:

```python
generate_playlist(query: str, language: str | None = None, n_songs: int = 25) -> pd.DataFrame
```

Under the hood, it:

1. Loads the dataset from `artists_datasets/language.csv` by default.
2. Builds or reuses FAISS indexes for:
   - audio features
   - semantic text embeddings
3. Calls Nvidia NIM (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`) to convert the prompt into a structured playlist intent.
4. Retrieves a large candidate pool.
5. Scores songs using semantic similarity, audio similarity, popularity, and recency.
6. Applies diversity rules:
   - max 3 songs per album
   - max 10 songs per artist
   - dedupe by normalized song identity
   - 70% popular tracks, 30% discovery tracks
7. Returns the final playlist as a dataframe.

## The important idea

The LLM does **not** directly choose `energy=0.23`, `tempo=78`, or `valence=0.41` in a brittle way.

Instead, it returns a high-level intent object with:

- language
- moods
- activity
- keywords
- semantic rewrites
- coarse audio preferences

That keeps the system much more stable.

## Data flow

```text
User prompt
  -> Nvidia NIM intent parse (nvidia/nemotron-3-nano-omni-30b-a3b-reasoning)
  -> query embeddings + audio query vector
  -> FAISS candidate retrieval
  -> hybrid scoring
  -> popularity split + dedupe + album/artist caps
  -> final playlist dataframe
```

## Core pieces

### 1. `PlaylistIntent`

This dataclass holds the parsed request. It also knows how to build:

- an audio query vector
- a semantic search text string

The audio vector uses:

- `danceability`
- `energy`
- `acousticness`
- `instrumentalness`
- `speechiness`
- `liveness`
- `valence`
- normalized `tempo`

### 2. `NvidiaNimIntentParser`

This is the LLM layer.

It sends a strict JSON-only prompt to Nvidia NIM at:

`https://integrate.api.nvidia.com/v1/chat/completions`

Using the fast `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` model (configurable via `NVIDIA_MODEL`).

The expected output looks like this:

```json
{
  "language": "Hindi",
  "moods": ["nostalgic", "melancholic", "reflective", "sad"],
  "activity": "reflecting",
  "keywords": ["old memories", "regret", "heartbreak", "past love"],
  "semantic_queries": [
    "Hindi nostalgic songs for old memories",
    "sad Hindi songs about past regrets",
    "melancholic Hindi hits"
  ],
  "danceability": 0.2,
  "energy": 0.3,
  "acousticness": 0.8,
  "instrumentalness": 0.1,
  "speechiness": 0.1,
  "liveness": 0.1,
  "valence": 0.2,
  "tempo": 80
}
```

If the Nvidia API times out or returns unusable text, the code automatically falls back to a heuristic parser so the app never crashes.

### 3. FAISS retrieval

There are two indexes:

- semantic index over song text documents
- audio index over normalized audio features

The semantic side helps with prompts like:

- "mountain drive"
- "old messages"
- "rainy evening"

The audio side helps keep the results musically coherent.

FAISS docs:
- [FAISS](https://github.com/facebookresearch/faiss)

### 4. Ranking and filtering

The final score combines:

- semantic similarity
- audio similarity
- popularity
- recency

Then the playlist builder enforces:

- no duplicate songs
- no more than 3 from the same album
- no more than 10 from the same artist
- 70/30 popular vs discovery balance

## Example 1: emotional Hindi prompt

Prompt:

```text
Songs that feel like reading old messages you should have deleted years ago.
```

With `language="Hindi"`, the real Ollama parse I saw was:

```json
{
  "language": "Hindi",
  "moods": ["nostalgic", "melancholic", "reflective", "sad"],
  "activity": "reflecting",
  "keywords": ["old memories", "regret", "heartbreak", "past love"],
  "semantic_queries": [
    "Hindi nostalgic songs for old memories",
    "sad Hindi songs about past regrets",
    "melancholic Hindi hits"
  ],
  "danceability": 0.2,
  "energy": 0.3,
  "acousticness": 0.8,
  "instrumentalness": 0.1,
  "speechiness": 0.1,
  "liveness": 0.1,
  "valence": 0.2,
  "tempo": "slow"
}
```

That intent then becomes the actual retrieval signal.

## Example 2: late-night drive

Prompt:

```text
I want music for a late-night drive in the mountains. Atmospheric, slightly melancholic, not too slow.
```

The backend turns that into a lower-energy, mid-tempo, moody retrieval request. A typical result set from the current code will look like:

```text
track_name                          artist_name                   album_name                    popularity  final_score
Cold Coffee                         Ed Sheeran                  5                             49          0.609837
Vaadi En Trip - From "Lover"        Sean Roldan, ofRO           Vaadi En Trip (From "Lover")  38          0.606718
Someday                             Passenger                   The Boy Who Cried Wolf        43          0.601488
Bedtime Story                       Madonna                     Bedtime Stories               42          0.570778
Lover                               Taylor Swift                Sunset Vibes                  14          0.568146
```

The exact songs will shift with the dataset, cache, and retrieval model, but the shape stays the same.

## Example 3: using the module directly

```python
from find_songs import generate_playlist

playlist = generate_playlist(
    "Bollywood romantic songs for a long drive",
    language="Hindi",
    n_songs=25,
)

print(playlist[["track_name", "artist_name", "album_name", "popularity"]].head())
```

## Example 4: CLI usage

```bash
.venv/bin/python find_songs.py "rainy evening with soft melancholic Hindi songs" --language Hindi --n 25
```

You can also export JSON:

```bash
.venv/bin/python find_songs.py "late-night drive" --language English --json
```

## A few implementation notes

- `NVIDIA_API_KEY` in `.env` authenticates requests to the Nvidia NIM API.
- `NVIDIA_MODEL` sets the Nvidia model identifier (defaults to `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`).
- `NVIDIA_BASE_URL` can override the base endpoint (defaults to `https://integrate.api.nvidia.com/v1`).
- `PLAYLIST_EMBEDDING_MODEL` can override the default sentence transformer.
- `.playlist_cache/` stores FAISS indexes so the expensive embedding step is reused.
- `HF_TOKEN` is loaded from `.env` if you need Hugging Face access for model download.
- The dataset default is `artists_datasets/language.csv`.

## External references

These are the best outside docs for the concepts used here:

- [NVIDIA NIM & API Catalog](https://build.nvidia.com/explore/discover)
- [Sentence-Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)

## If you only remember one thing

The LLM is used to understand the request. The backend is used to make the playlist good.
