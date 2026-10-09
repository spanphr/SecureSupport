"""Environment-based settings for SecureSupport (Python 3.11+)."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load optional local configuration without requiring API credentials."""

    model_config = SettingsConfigDict(
        env_prefix="SECURESUPPORT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    debug: bool = False
    openai_api_key: SecretStr | None = None

    embedding_model: str = Field(default="text-embedding-3-small", min_length=1)
    chroma_path: Path = Path(__file__).resolve().parents[1] / "data" / "chroma"
    chat_model: str = Field(default="gpt-4.1-mini", min_length=1)
