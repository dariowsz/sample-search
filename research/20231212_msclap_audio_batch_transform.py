# %%
import research.start  # noqa isort:skip

# %%
import torch
from msclap import CLAP
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import SamplesDataset

# %%
# Initialize Dataset
sample_dirs = [
    "/Users/dario.wisznewer/Cymatics",
    "/Users/dario.wisznewer/Samples",
    "/Users/dario.wisznewer/Splice",
]
dataset = SamplesDataset(sample_dirs)
len(dataset)

# %%
# Create DataLoader
batch_size = 32

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
)

# %%
# Initialize model and encode audio
clap_model = CLAP(
    "checkpoints/CLAP_weights_2023.pth",
    version="2023",
    use_cuda=False,
)

# %%
# Encode audio and save embedding batches to files
print("Encoding audio...")
audio_batches = []
for idx, batch in enumerate(tqdm(dataloader)):
    encoded_audio_batch = clap_model.get_audio_embeddings(audio_files=batch)
    audio_batches.append(encoded_audio_batch)
audio_batches = torch.cat(audio_batches)
print(audio_batches.shape)
torch.save(audio_batches, f"output/audio_embeddings/msclap/embeddings.pt")

# %%
