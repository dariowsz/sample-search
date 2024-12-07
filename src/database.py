from abc import ABC, abstractmethod
from typing import Any

import chromadb
import numpy as np


class VectorDatabase(ABC):
    @abstractmethod
    def __init__(self, collection_name: str, host: str, port: int):
        """Initialize the vector database"""
        pass

    @abstractmethod
    def upload_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: list[Any] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        """Upload embeddings to the database and return count"""
        pass

    @abstractmethod
    def query_embeddings(self, query_embeddings: np.ndarray, n_results: int = 5) -> Any:
        """Query the database and return filenames"""
        pass

    @abstractmethod
    def delete_embeddings(self, ids: list[str]) -> None:
        """Delete embeddings from the database"""
        pass


class ChromaDatabase(VectorDatabase):
    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 8000,
    ):
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self._initialize_collection(collection_name)

    def _initialize_collection(self, collection_name: str) -> chromadb.Collection:
        try:
            return self.client.get_collection(name=collection_name)
        except Exception:
            return self.client.create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )

    def upload_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: list[Any] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        self.collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
            or [str(idx + self.collection.count()) for idx in range(len(embeddings))],
        )
        return self.collection.count()

    # TODO: type this with a common custom type for every vector database
    def query_embeddings(self, embeddings: np.ndarray, n_results: int = 5) -> Any:
        results = self.collection.query(
            query_embeddings=embeddings, n_results=n_results
        )
        return results

    def delete_embeddings(self, ids: list[str]) -> None:
        self.collection.delete(ids=ids)
