# %%
import research.start as start  # noqa isort:skip

# %%
from msclap import CLAP

# %%
# Load model
clap_model = CLAP("checkpoints/CLAP_weights_2023.pth", version="2023", use_cuda=False)

# Extract text embeddings
text_embeddings = clap_model.get_text_embeddings(
    ["Drums", "Bongos", "Guitar", "Female voice"]
)

# Extract audio embeddings
audio_embeddings = clap_model.get_audio_embeddings(
    ["/Users/dario.wisznewer/Test_Samples/Cymatics-Bongo4.wav"]
)

# Compute similarity between audio and text embeddings
similarities = clap_model.compute_similarity(audio_embeddings, text_embeddings)

# %%
print(similarities)

# %%
