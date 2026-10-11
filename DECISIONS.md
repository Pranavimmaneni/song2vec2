# Decisions

## 1. Host on Streamlit Community Cloud

**Decision:** Deploy song2vec 2.0 on Streamlit Community Cloud.

**Why:** It is free and deploys straight from a GitHub repo. CPU inference is fine because the app embeds only short text queries online. The memory limit (roughly 690 MB to 2.7 GB, not guaranteed) is the main risk, which is what this preflight measures.

**Alternatives considered:**
- Hugging Face Gradio Spaces: hosting this app there needs a paid plan, and the free ZeroGPU tier needs an account older than 30 days.
- Self-hosting on a VPS: costs money and needs ops work that a hobby project doesn't need.

## 2. Serve only CLAP's text branch

**Decision:** Online, load `ClapTextModelWithProjection` instead of the full `ClapModel`.

**Why:** Audio embeddings are computed offline, so the app only needs to embed text queries. Skipping the audio tower makes the memory footprint and load time much smaller. `tests/test_text_branch.py` checks that the text branch gives the same output as the full model's `get_text_features()`.

**Alternatives considered:**
- Load the full `ClapModel`: simpler, but it wastes memory on an audio encoder that never runs online.
- Precompute embeddings for a fixed query vocabulary: no model needed online, but users couldn't search with free text.

## 3. Pin model revisions to commit hashes

**Decision:** Pin `CLAP_REVISION` and `BGE_REVISION` in `config.py` to exact Hugging Face Hub commit hashes, never `main`.

**Why:** Offline embeddings (audio, lyric summaries) and online query embeddings must come from exactly the same weights and tokenizer. If an upstream update changes the model, similarities would quietly become meaningless.

**Alternatives considered:**
- Track `main`: always up to date, but an upstream push could break search without warning.
- Vendor the weights into the repo: fully reproducible, but bloats the repo and GitHub limits file sizes.

## 4. Load models with `st.cache_resource`

**Decision:** Load both models once per server process, inside a function decorated with `@st.cache_resource`.

**Why:** Streamlit reruns the whole script on every interaction. Without caching, every click would reload the models, which is slow and would push memory past the limit. `cache_resource` shares one copy across all sessions and reruns.

**Alternatives considered:**
- `st.session_state`: one copy per browser session, so memory grows with the number of users.
- Module-level globals: Streamlit doesn't reliably keep them across reruns.
