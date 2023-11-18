# %%
import research.start as start  # noqa isort:skip

# %%
import chromadb
import laion_clap
import librosa
from IPython.display import Audio, display

# %%
# Initialize chromadb client
chroma_client = chromadb.HttpClient(host="localhost", port="8000")

# %%
# Get collection and embedding count
collection = chroma_client.get_collection(name="audio_samples")
collection.count()

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
sample_texts = ["Heavy bass sound", ""]
text_embeddings = model.get_text_embedding(sample_texts, use_tensor=True)
text_embeddings.shape

# %%
sample_count = 5
results = collection.query(
    query_embeddings=text_embeddings[0].tolist(),
    n_results=sample_count,
)

# %%
for i in range(sample_count):
    result_filename: str = results["metadatas"][0][i]["filename"]  # type: ignore
    audio_data, _ = librosa.load(path=result_filename, sr=44100)
    print(result_filename.split("/")[-1])
    display(Audio(audio_data, rate=44100))

# %%
