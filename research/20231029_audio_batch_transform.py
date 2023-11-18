# %%
import research.start as start  # noqa isort:skip

# %%
import laion_clap
import torch
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

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# %%
# Initialize model and encode audio
model = laion_clap.CLAP_Module(
    enable_fusion=False,
    amodel="HTSAT-base",
    tmodel="roberta",
)
model.load_ckpt("checkpoints/music_audioset_epoch_15_esc_90.14.pt")

# TODO: Is this necessary?
model.eval()

# %%
# Encode audio and save embedding batches to files
print("Encoding audio...")
for idx, batch in enumerate(tqdm(dataloader)):
    encoded_audio_batch = model.get_audio_embedding_from_filelist(
        x=batch, use_tensor=True
    )
    torch.save(encoded_audio_batch, f"output/audio_embeddings/batch{idx}.pt")

# %%
