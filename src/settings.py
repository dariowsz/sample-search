from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    ENCODER_MODEL_WEIGHTS_PATH: str
    ENCODER_MODEL_DEVICE: Literal["cuda", "mps", "cpu"]

    TOKENIZERS_PARALLELISM: bool = False

    VECTOR_DB_COLLECTION_NAME: str
    VECTOR_DB_HOST: str
    VECTOR_DB_PORT: int


env = Settings()  # type: ignore
