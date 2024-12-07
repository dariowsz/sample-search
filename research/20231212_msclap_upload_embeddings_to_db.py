# %%
import research.start as start  # noqa isort:skip

# %%
import chromadb
import torch

from src.datasets import SamplesDataset

# %%
# Initialize chromadb client
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

# %%
# Get collection or create it if it doesn't exist
try:
    collection = chroma_client.get_collection(name="msclap_audio_samples")
except Exception as e:
    collection = chroma_client.create_collection(
        name="msclap_audio_samples", metadata={"hnsw:space": "cosine"}
    )

# %%
# Initialize Dataset and create DataLoader
sample_dirs = [
    "/Users/dario.wisznewer/Cymatics",
    "/Users/dario.wisznewer/Samples",
    "/Users/dario.wisznewer/Splice",
]
dataset = SamplesDataset(sample_dirs)

# %%
tensor = torch.load(f"output/audio_embeddings/msclap/embeddings.pt")
collection.add(
    embeddings=tensor.tolist(),
    metadatas=[{"filename": filename} for filename in dataset],
    ids=[str(idx) for idx in range(0, len(dataset))],
)

# %%
# Check all embeddings were uploaded
collection.count()

# %%
