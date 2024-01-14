# %%
import research.start as start  # noqa isort:skip

# %%
import chromadb
import librosa
from IPython.display import Audio, display
from msclap import CLAP

# %%
# Initialize chromadb client
chroma_client = chromadb.HttpClient(host="localhost", port="8000")

# %%
# Get collection and embedding count
collection = chroma_client.get_collection(name="msclap_audio_samples")
collection.count()

# %%
# Initialize model and encode audio
clap_model = CLAP(
    "checkpoints/CLAP_weights_2023.pth",
    version="2023",
    use_cuda=False,
)

# %%
sample_texts = ["Saxophone"]
text_embeddings = clap_model.get_text_embeddings(sample_texts)
text_embeddings.shape

# %%
sample_count = 5
# TODO: Try storing embeddings in a file, loading in memory and
# using clap_model.compute_similarity(a_embeddings, t_embeddings) function
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
