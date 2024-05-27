# %%
# This model is not very good at generating captions for music samples.

import research.start as start  # noqa isort:skip

# %%
from msclap import CLAP

# %%
# Load model
clap_model = CLAP(
    "checkpoints/clapcap_weights_2023.pth",
    version="clapcap",
    use_cuda=False,
)

# %%
# Generate audio captions
file_paths = [
    "/Users/dario.wisznewer/Test_Samples/Cymatics - Chanel - 74 BPM E Maj.wav"
]
captions = clap_model.generate_caption(file_paths)
print(captions)
