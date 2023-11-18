# %%
import research.start as start  # noqa isort:skip

# %%
import chromadb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import SamplesDataset

# %%
# Initialize chromadb client
chroma_client = chromadb.HttpClient(host="localhost", port="8000")

# %%
# Get collection or create it if it doesn't exist
try:
    collection = chroma_client.get_collection(name="audio_samples")
except Exception as e:
    collection = chroma_client.create_collection(
        name="audio_samples", metadata={"hnsw:space": "cosine"}
    )

# %%
# Initialize Dataset and create DataLoader
sample_dirs = [
    "/Users/dario.wisznewer/Cymatics",
    "/Users/dario.wisznewer/Samples",
    "/Users/dario.wisznewer/Splice",
]
dataset = SamplesDataset(sample_dirs)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# %%
for idx, batch in enumerate(tqdm(dataloader)):
    tensor = torch.load(f"output/audio_embeddings/batch{idx}.pt")
    collection.add(
        embeddings=tensor.tolist(),
        metadatas=[{"filename": filename} for filename in batch],
        ids=[str(idx) for idx in range((32 * idx), (32 * idx) + len(batch))],
    )

# %%
# Check all embeddings were uploaded
collection.count()

# %%
