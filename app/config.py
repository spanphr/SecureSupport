"""Environment-based settings for SecureSupport (Python 3.11+)."""

from typing import Literal

from pydantic import SecretStr
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
