"""song2vec 2.0 preflight: can Streamlit Community Cloud run the online stack?"""

import os
import time

import psutil
import streamlit as st
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, ClapTextModelWithProjection

import config

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

CLAP_REFERENCE = "upbeat electronic dance music"
BGE_REFERENCE = "a song about heartbreak"


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


@st.cache_resource(show_spinner="Loading models (first run only)...")
def load_models():
    start = time.perf_counter()
    clap_tokenizer = AutoTokenizer.from_pretrained(
        config.CLAP_MODEL_ID, revision=config.CLAP_REVISION
    )
    clap_text = ClapTextModelWithProjection.from_pretrained(
        config.CLAP_MODEL_ID, revision=config.CLAP_REVISION
    ).eval()
    bge = SentenceTransformer(
        config.BGE_MODEL_ID, revision=config.BGE_REVISION, device="cpu"
    )
    load_seconds = time.perf_counter() - start
    return clap_tokenizer, clap_text, bge, load_seconds


def embed_clap(tokenizer, model, texts):
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    with torch.inference_mode():
        embeds = model(**inputs).text_embeds
    return F.normalize(embeds, dim=-1)


def embed_bge(model, texts):
    with torch.inference_mode():
        embeds = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return embeds


st.set_page_config(page_title="song2vec 2.0 preflight", page_icon="🎵")
st.title("song2vec 2.0 preflight")
st.caption("Checks that CLAP's text branch and bge-small run on this host.")

clap_tokenizer, clap_text, bge, load_seconds = load_models()

if "searches" not in st.session_state:
    st.session_state.searches = 0
if "peak_mb" not in st.session_state:
    st.session_state.peak_mb = rss_mb()

with st.form("search"):
    query = st.text_input("Search text", placeholder="e.g. dreamy synth pop about summer")
    submitted = st.form_submit_button("Search")

if submitted and query.strip():
    clap_vecs = embed_clap(clap_tokenizer, clap_text, [query, CLAP_REFERENCE])
    bge_vecs = embed_bge(bge, [query, BGE_REFERENCE])
    st.session_state.searches += 1

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Musical (CLAP)")
        st.metric("Embedding dimension", clap_vecs.shape[1])
        st.metric(f'Cosine vs "{CLAP_REFERENCE}"', f"{(clap_vecs[0] @ clap_vecs[1]).item():.4f}")
    with col2:
        st.subheader("Lyrical (bge-small)")
        st.metric("Embedding dimension", bge_vecs.shape[1])
        st.metric(f'Cosine vs "{BGE_REFERENCE}"', f"{(bge_vecs[0] @ bge_vecs[1]).item():.4f}")
elif submitted:
    st.warning("Type some text first.")

st.divider()
st.subheader("Audio check")
if config.AUDIO_URL.startswith("http"):
    st.audio(config.AUDIO_URL)
    license_md = (
        f"[{config.AUDIO_LICENSE}]({config.AUDIO_LICENSE_URL})"
        if config.AUDIO_LICENSE_URL
        else config.AUDIO_LICENSE
    )
    st.markdown(
        f"**{config.AUDIO_TITLE}** by {config.AUDIO_ARTIST}  \n"
        f"License: {license_md}  \n"
        f"Source: {config.AUDIO_SOURCE}"
    )
else:
    st.info("No audio clip configured yet. Fill in the AUDIO_* values in config.py.")

current_mb = rss_mb()
st.session_state.peak_mb = max(st.session_state.peak_mb, current_mb)

with st.sidebar:
    st.header("Resources")
    st.metric("Process memory (RSS)", f"{current_mb:.0f} MB")
    st.metric("Peak memory this session", f"{st.session_state.peak_mb:.0f} MB")
    st.metric("Searches run", st.session_state.searches)
    st.metric("Model load time", f"{load_seconds:.1f} s")
