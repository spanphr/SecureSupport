"""Environment-based settings for SecureSupport (Python 3.11+)."""

from decimal import Decimal
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
        # Blank values in .env mean "unset". Without this, copying .env.example
        # with empty optional numbers fails validation at startup.
        env_ignore_empty=True,
    )

    environment: Literal["development", "test", "production"] = "development"
    debug: bool = False
    openai_api_key: SecretStr | None = None

    embedding_model: str = Field(default="text-embedding-3-small", min_length=1)
    chroma_path: Path = Path(__file__).resolve().parents[1] / "data" / "chroma"
    chat_model: str = Field(default="gpt-4.1-mini", min_length=1)
    database_path: Path = Path(__file__).resolve().parents[1] / "data" / "support.sqlite3"

    # Retrieval policy. The distance ceiling is deliberately opt-in: cosine distance
    # is not a calibrated relevance probability, so enabling it is a per-corpus decision.
    retrieval_top_k: int = Field(default=3, ge=1, le=10)
    retrieval_max_distance: float | None = Field(default=None, ge=0.0)

    # Conversation budget: bounded history keeps prompts and cost predictable.
    max_history_turns: int = Field(default=6, ge=0, le=20)

    # Explicit pricing in USD per million tokens. Unset means cost is reported as unknown
    # rather than guessed from a hard-coded price table that could silently go stale.
    input_cost_per_million_usd: Decimal | None = Field(default=None, ge=0)
    output_cost_per_million_usd: Decimal | None = Field(default=None, ge=0)

    # One structured JSON line per request; contains no message text or record contents.
    trace_log_enabled: bool = True
