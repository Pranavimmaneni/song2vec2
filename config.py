"""Pinned models and audio clip metadata for the song2vec 2.0 preflight."""

# Models. Revisions are commit hashes on the Hugging Face Hub, never "main":
# offline (audio/lyrics) and online (query) embeddings must use identical weights.
CLAP_MODEL_ID = "laion/larger_clap_music"
CLAP_REVISION = "a0b4534a14f58e20944452dff00a22a06ce629d1"

BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

# Audio clip (FMA track 000002)
AUDIO_URL = "https://huggingface.co/datasets/prnv-ima/song2vec2-audio/resolve/main/audio/000002.mp3"
AUDIO_TITLE = "Food"
AUDIO_ARTIST = "AWOL"
AUDIO_LICENSE = "Attribution-NonCommercial-ShareAlike 3.0 International"
# CC 3.0 "International" licenses are published as "Unported".
AUDIO_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/3.0/"
AUDIO_SOURCE = "Free Music Archive (FMA dataset)"
