"""Runtime configuration loaded from environment variables."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Secrets and runtime settings that must not be committed."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: SecretStr


def load_settings() -> Settings:
    """Load and validate local runtime configuration."""

    return Settings()  # type: ignore[call-arg]
