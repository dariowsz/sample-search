from torch.utils.data import DataLoader

from database import VectorDatabase
from datasets import SamplesDataset
from models import EncoderModel


class SearchEngine:
    def __init__(
        self,
        vector_db: VectorDatabase,
        encoder_model: EncoderModel,
    ):
        self.vector_db = vector_db
        self.encoder_model = encoder_model

    def get_indexed_dirs(self) -> list[str]:
        with open("indexed_sample_dirs.txt", "r") as f:
            return [line.strip() for line in f.readlines()]

    def index_dirs(
        self, samples_dirs: list[str], batch_size: int = 32, num_workers: int = 0
    ) -> int:
        dataset = SamplesDataset(samples_dirs)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        for batch in dataloader:
            embeddings = self.encoder_model.get_audio_embeddings(batch)
            count = self.vector_db.upload_embeddings(
                embeddings=embeddings.cpu().numpy(),
                metadatas=[{"filename": filename} for filename in batch],
                ids=[filename for filename in batch],
            )

        with open("indexed_sample_dirs.txt", "a") as f:
            for dir in samples_dirs:
                f.write(f"{dir}\n")

        return count

    def query(self, query: str, n_results: int = 5) -> list[str]:
        encoded_query = self.encoder_model.get_text_embeddings([query]).cpu().numpy()
        results = self.vector_db.query_embeddings(encoded_query, n_results)
        return [result["filename"] for result in results["metadatas"][0]]

    def remove_indexed_dir(self, dir: str):
        dataset = SamplesDataset([dir])
        self.vector_db.delete_embeddings([filename for filename in dataset])
        with open("indexed_sample_dirs.txt", "r") as f:
            lines = f.readlines()
        with open("indexed_sample_dirs.txt", "w") as f:
            for line in lines:
                if line.strip() != dir:
                    f.write(line)
