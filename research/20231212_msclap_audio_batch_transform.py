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
    num_workers=0,  # FIXME: Multithreading is thorwing an error. I used 8 in this attribute whitout issues before.
)

# %%
# Initialize model and encode audio
clap_model = CLAP(
    "checkpoints/CLAP_weights_2023.pth",
    version="2023",
    use_cuda=False,
)
if torch.backends.mps.is_available():
    clap_model.clap.to(torch.device("mps"))

# %%
# Encode audio and save embedding batches to files
print("Encoding audio...")
audio_batches = []
for idx, batch in enumerate(tqdm(dataloader)):
    preprocessed_audio = clap_model.preprocess_audio(batch, resample=True)
    if torch.backends.mps.is_available():
        preprocessed_audio = preprocessed_audio.to(torch.device("mps"))  # type: ignore
    encoded_audio_batch = clap_model._get_audio_embeddings(preprocessed_audio)
    audio_batches.append(encoded_audio_batch.cpu())
audio_batches = torch.cat(audio_batches)
print(audio_batches.shape)
torch.save(audio_batches, f"output/audio_embeddings/msclap/embeddings2.pt")

# %%
