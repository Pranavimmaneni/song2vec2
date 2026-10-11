# song2vec 2.0 preflight

A minimal Streamlit app that checks Streamlit Community Cloud can run song2vec 2.0's online stack before the real app is built.

## What it checks

- **CLAP text branch** (`laion/larger_clap_music`, via `ClapTextModelWithProjection`) loads and embeds text queries for the musical space.
- **bge-small** (`BAAI/bge-small-en-v1.5`) loads and embeds text for the lyrical space.
- Both fit in Streamlit Community Cloud's memory limit (roughly 690 MB to 2.7 GB, not guaranteed). The sidebar shows current and peak process memory.
- Audio hosted on Hugging Face plays in `st.audio`.

Audio embeddings are computed offline, so only the text side runs online. Model revisions are pinned in `config.py`. See [DECISIONS.md](DECISIONS.md) for why.

## Run locally

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux), then:

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

## Run the test

The test loads the full CLAP model, so run it locally only. Install pytest first:

```bash
pip install pytest
```

```bash
pytest tests/ -s
```

It checks that the standalone text branch matches the full model's `get_text_features()` (atol 1e-5) over 20 strings and prints the max absolute difference.

## Preflight results

- [ ] Both models load from a fresh start (load time: ___ s)
- [ ] 20 repeated searches without memory steadily increasing (peak: ___ MB)
- [ ] Two browser sessions work at the same time
- [ ] Audio plays, and the app recovers after a reboot
