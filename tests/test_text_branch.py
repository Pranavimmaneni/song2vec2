"""Check that CLAP's standalone text branch matches the full model.

Runs locally only (it loads the full ClapModel, audio tower included).
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, ClapModel, ClapTextModelWithProjection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLAP_MODEL_ID, CLAP_REVISION  # noqa: E402

TEST_STRINGS = [
    "upbeat electronic dance music",
    "slow acoustic folk ballad",
    "heavy metal with distorted guitars",
    "smooth jazz saxophone solo",
    "lo-fi hip hop beats to study to",
    "classical string quartet",
    "melancholic piano piece",
    "aggressive punk rock",
    "dreamy ambient synth pads",
    "funky bass line with brass section",
    "reggae with offbeat guitar skanks",
    "epic orchestral film score",
    "happy children's song with ukulele",
    "dark and moody trap beat",
    "country song with steel guitar and fiddle",
    "solo cello",
    "I've been walking alone down this empty road",
    "dancing all night under the city lights",
    "you broke my heart and left me in the rain",
    "a",
]


def test_text_branch_matches_full_model():
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(CLAP_MODEL_ID, revision=CLAP_REVISION)
    full = ClapModel.from_pretrained(CLAP_MODEL_ID, revision=CLAP_REVISION).eval()
    text = ClapTextModelWithProjection.from_pretrained(
        CLAP_MODEL_ID, revision=CLAP_REVISION
    ).eval()

    inputs = tokenizer(TEST_STRINGS, padding=True, return_tensors="pt")
    with torch.inference_mode():
        # transformers v5: get_text_features() returns an output object whose
        # pooler_output is the L2-normalized projection; text_embeds is not normalized.
        full_out = full.get_text_features(**inputs).pooler_output
        branch_out = F.normalize(text(**inputs).text_embeds, dim=-1)

    max_diff = (full_out - branch_out).abs().max().item()
    print(f"\nMax absolute difference over {len(TEST_STRINGS)} strings: {max_diff:.3e}")
    assert full_out.shape == branch_out.shape
    assert torch.allclose(full_out, branch_out, atol=1e-5), f"max diff {max_diff}"
